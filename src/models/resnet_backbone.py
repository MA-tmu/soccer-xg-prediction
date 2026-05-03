
# src/models/resnet_backbone.py

import torch
import torch.nn as nn
import torchvision.models as tvm


class FrozenResNetBackbone(nn.Module):
    """
    Input : x (B,T,3,H,W)
    Output: feat (B,T,D)
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = True, freeze: bool = True):
        super().__init__()

        name = str(name).lower()
        if name == "resnet18":
            weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
            m = tvm.resnet18(weights=weights)
            self.out_dim = 512
        elif name == "resnet50":
            weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
            m = tvm.resnet50(weights=weights)
            self.out_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {name}. Use resnet18/resnet50.")

        self.backbone = nn.Sequential(*list(m.children())[:-1])  # (N,D,1,1)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.freeze = freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,3,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if c != 3:
            raise ValueError(f"FrozenResNetBackbone expects C=3, got C={c}")

        xt = x.reshape(b * t, c, h, w)

        if self.freeze:
            with torch.no_grad():
                feat = self.backbone(xt).flatten(1)
        else:
            feat = self.backbone(xt).flatten(1)

        return feat.view(b, t, self.out_dim)
