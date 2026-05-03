# src/models/temporal_head.py

import torch
import torch.nn as nn


class TemporalTCNHead(nn.Module):
    """
    Input : feat (B, T, D)
    Output: logit (B,), prob (B,)

    Config style:
      - tcn_channels: int
      - tcn_layers: int
      - kernel_size: int (odd recommended)
      - dropout: float
    """

    def __init__(
        self,
        in_dim: int,
        tcn_channels: int = 256,
        tcn_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        in_dim = int(in_dim)
        tcn_channels = int(tcn_channels)
        tcn_layers = int(tcn_layers)
        kernel_size = int(kernel_size)
        dropout = float(dropout)

        if tcn_layers <= 0:
            raise ValueError(f"tcn_layers must be > 0, got {tcn_layers}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")

        pad = kernel_size // 2

        layers = []
        prev = in_dim
        for _ in range(tcn_layers):
            layers.append(nn.Conv1d(prev, tcn_channels, kernel_size=kernel_size, padding=pad, bias=False))
            layers.append(nn.BatchNorm1d(tcn_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = tcn_channels

        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 1)

    def forward(self, feat: torch.Tensor):
        if feat.ndim != 3:
            raise ValueError(f"Expected feat (B,T,D), got {tuple(feat.shape)}")

        # (B,T,D) -> (B,D,T)
        x = feat.transpose(1, 2)
        x = self.tcn(x)          # (B,C,T)
        x = x.mean(dim=2)        # (B,C)

        logit = self.classifier(x).squeeze(1)  # (B,)
        prob = torch.sigmoid(logit)
        return logit, prob
