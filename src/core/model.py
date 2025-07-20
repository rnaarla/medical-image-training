import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetMedical(nn.Module):
    """ResNet model optimized for medical image classification"""
    
    def __init__(self, num_classes=10, pretrained=True, dropout=0.5):
        super(ResNetMedical, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final FC layer for medical classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Initialize custom layers
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=10, pretrained=True, **kwargs):
    """Factory function to create model instance"""
    return ResNetMedical(num_classes=num_classes, pretrained=pretrained, **kwargs)
