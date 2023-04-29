import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

INPUT_IMG_SIZE = (640, 640)

class CustomRCNN(nn.Module):
    def __init__(self, n_classes=7, pretrained=True):
        super(CustomRCNN, self).__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        self.model.backbone.body.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.transform = GeneralizedRCNNTransform(min_size=(800,), max_size=1333, image_mean=(0.5,), image_std=(0.5,))
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace pre-trained head with new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    def forward(self, image, target):
        self.model.forward(image, target)