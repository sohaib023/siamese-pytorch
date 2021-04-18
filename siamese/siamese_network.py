import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)
        out_features = list(self.backbone.modules())[-1].out_features

        self.cls_head = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(out_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),

            # nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        
        combined_features = torch.cat((feat1, feat2), dim=-1)

        output = self.cls_head(combined_features)
        return output