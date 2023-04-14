import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

INPUT_IMG_SIZE = (224, 224)

class CustomVGG16(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super(CustomVGG16, self).__init__()
        self.feature_extractor = models.vgg16(pretrained=pretrained)
        first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(self.feature_extractor.features))  
        self.feature_extractor.features= nn.Sequential(*first_conv_layer )  
        self.feature_extractor = self.feature_extractor.features[:-1]
        
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(
                kernel_size=(INPUT_IMG_SIZE[0] // 2 ** 5, INPUT_IMG_SIZE[1] // 2 ** 5)
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=self.feature_extractor[-2].out_channels,
                out_features=n_classes,
            ),
        )
        self._freeze_params()

    def _freeze_params(self):
        for param in self.feature_extractor[:23].parameters():
            param.requires_grad = False
        

    def forward(self, x):
        feature_maps = self.feature_extractor(x)
        scores = self.classification_head(feature_maps)
        feature_maps = self.feature_extractor(x)
        scores = self.classification_head(feature_maps)

        if self.training:
            return scores

        else:
            probs = nn.functional.softmax(scores, dim=-1)

            weights = self.classification_head[3].weight
            weights = (
                weights.unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(0)
                .repeat(
                    (
                        x.size(0),
                        1,
                        1,
                        INPUT_IMG_SIZE[0] // 2 ** 4,
                        INPUT_IMG_SIZE[0] // 2 ** 4,
                    )
                )
            )
            feature_maps = feature_maps.unsqueeze(1).repeat((1, probs.size(1), 1, 1, 1))
            location = torch.mul(weights, feature_maps).sum(axis=2)
            location = F.interpolate(location, size=INPUT_IMG_SIZE, mode="bilinear")

            maxs, _ = location.max(dim=-1, keepdim=True)
            maxs, _ = maxs.max(dim=-2, keepdim=True)
            mins, _ = location.min(dim=-1, keepdim=True)
            mins, _ = mins.min(dim=-2, keepdim=True)
            norm_location = (location - mins) / (maxs - mins)

            return probs, norm_location