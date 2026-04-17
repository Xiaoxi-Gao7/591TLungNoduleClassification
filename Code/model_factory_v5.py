import torch
import torch.nn as nn
from torchvision import models

class V5_AttentionNet(nn.Module):
    def __init__(self):
        super(V5_AttentionNet, self).__init__()
        # dual path EfficientNet-B0
        self.local_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.global_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        feature_dim = self.local_backbone.classifier[1].in_features
        self.local_backbone.classifier = nn.Identity()
        self.global_backbone.classifier = nn.Identity()

        # Feature Attention Gate
        self.attn_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5), # high Dropout
            nn.Linear(512, 2)
        )

    def forward(self, x_l, x_g):
        f_l = self.local_backbone(x_l)
        f_g = self.global_backbone(x_g)
        
        combined = torch.cat((f_l, f_g), dim=1)
        
        weights = self.attn_gate(combined)
        combined = combined * weights
        
        return self.classifier(combined)

def get_model_v5():
    return V5_AttentionNet()