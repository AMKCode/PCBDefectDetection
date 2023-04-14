import torch
import torch.nn as nn
from torchvision import models

class CustomResNet18(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)

        # convert model to work on greyscale
        self.resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, n_classes)

    def forward(self, x):
        return self.resnet18(x)