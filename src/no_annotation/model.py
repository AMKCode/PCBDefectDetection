import torch
import torch.nn as nn
from torchvision import models

class CustomVGG(nn.Module):
    def __init__(self, n_classes=2):
        super(CustomVGG, self).__init__()
        vgg_firstlayer=models.vgg16(pretrained = True).features[0] #load just the first conv layer
        vgg=models.vgg16(pretrained = True).features[1:30] #load upto the classification layers except first conv layer

        w1=vgg_firstlayer.state_dict()['weight'][:,0,:,:]
        w2=vgg_firstlayer.state_dict()['weight'][:,1,:,:]
        w3=vgg_firstlayer.state_dict()['weight'][:,2,:,:]
        w4=w1+w2+w3 # add the three weigths of the channels
        w4=w4.unsqueeze(1)# make it 4 dimensional


        first_conv=nn.Conv2d(1, 64, 3, padding = (1,1)) #create a new conv layer
        first_conv.weigth=torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
        first_conv.bias=torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias


        self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv  layer
        self.feature_extractor =nn.Sequential(vgg)
        self.feature_extractor = models.vgg16(pretrained=True).features[:-1]
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(
                kernel_size=(224 // 2 ** 5, 224 // 2 ** 5) # 224 is the image dimension
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
                        224 // 2 ** 4,
                        224 // 2 ** 4,
                    )
                )
            )
            feature_maps = feature_maps.unsqueeze(1).repeat((1, probs.size(1), 1, 1, 1))
            location = torch.mul(weights, feature_maps).sum(axis=2)
            location = F.interpolate(location, size=(224, 224), mode="bilinear")

            maxs, _ = location.max(dim=-1, keepdim=True)
            maxs, _ = maxs.max(dim=-2, keepdim=True)
            mins, _ = location.min(dim=-1, keepdim=True)
            mins, _ = mins.min(dim=-2, keepdim=True)
            norm_location = (location - mins) / (maxs - mins)

            return probs, norm_location
