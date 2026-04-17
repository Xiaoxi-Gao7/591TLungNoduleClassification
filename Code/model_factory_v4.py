import torch
import torch.nn as nn
from torchvision import models

class DualPathNet(nn.Module):
    def __init__(self):
        super(DualPathNet, self).__init__()
        # path A 128x128 
        self.local_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # path B 256x256 
        self.global_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # EfficientNet-B0 is 1280)
        feature_dim = self.local_backbone.classifier[1].in_features
        
        self.local_backbone.classifier = nn.Identity()
        self.global_backbone.classifier = nn.Identity()

        # combine classification heads
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x_l, x_g):
        # x_l: local image, x_g: global image
        feat_l = self.local_backbone(x_l)
        feat_g = self.global_backbone(x_g)
        
        combined = torch.cat((feat_l, feat_g), dim=1)
        return self.classifier(combined)

def get_model_v4():
    return DualPathNet()