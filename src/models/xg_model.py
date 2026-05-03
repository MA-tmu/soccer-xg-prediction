# src/models/xg_model.py

from typing import Any, Dict

import torch
import torch.nn as nn

from src.models.resnet_backbone import FrozenResNetBackbone
from src.models.temporal_head import TemporalTCNHead


class XGModel(nn.Module):
    """
    mode=rgb : x (B,T,3,H,W)
    mode=flow: x (B,T,2,H,W) -> pad to 3ch -> backbone
    """

    def __init__(
        self,
        mode: str,
        backbone: str,
        backbone_pretrained: bool,
        freeze_backbone: bool,
        tcn_channels: int,
        tcn_layers: int,
        tcn_kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        self.mode = str(mode)
        if self.mode not in ("rgb", "flow"):
            raise ValueError(f"mode must be rgb/flow, got: {self.mode}")

        self.backbone = FrozenResNetBackbone(
            name=str(backbone),
            pretrained=bool(backbone_pretrained),
            freeze=bool(freeze_backbone),
        )

        self.temporal = TemporalTCNHead(
            in_dim=self.backbone.out_dim,
            tcn_channels=int(tcn_channels),
            tcn_layers=int(tcn_layers),
            kernel_size=int(tcn_kernel_size),
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected x (B,T,C,H,W), got {tuple(x.shape)}")

        if self.mode == "rgb":
            # (B,T,3,H,W)
            feat = self.backbone(x)
        else:
            # (B,T,2,H,W) -> (B,T,3,H,W)
            if x.shape[2] != 2:
                raise ValueError(f"flow mode expects C=2, got C={x.shape[2]}")
            zeros = torch.zeros_like(x[:, :, :1, :, :])
            x3 = torch.cat([x, zeros], dim=2)
            feat = self.backbone(x3)

        logit, prob = self.temporal(feat)
        return logit, prob


def build_model(cfg: Dict[str, Any], mode: str):
    """
    Used by train.py / eval.py
    """
    m = XGModel(
        mode=str(mode),
        backbone=str(cfg.get("backbone", "resnet18")),
        backbone_pretrained=bool(cfg.get("backbone_pretrained", True)),
        freeze_backbone=bool(cfg.get("freeze_backbone", True)),
        tcn_channels=int(cfg.get("tcn_channels", 256)),
        tcn_layers=int(cfg.get("tcn_layers", 2)),
        tcn_kernel_size=int(cfg.get("tcn_kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    return m
