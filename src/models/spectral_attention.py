"""
Spectral channel attention module (SE-Net style).

Applied before patch embedding so the model can learn which spectral
bands carry the most information for detecting plant stress before
committing to a spatial patch representation.

Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention adapted for spectral bands.

    Architecture:
        (B, C, H, W)
          → GlobalAvgPool → (B, C)
          → Linear(C → C//r) + ReLU
          → Linear(C//r → C) + Sigmoid
          → (B, C, 1, 1) attention weights
          → element-wise multiply with input
          → (B, C, H, W)

    Design choice — applied before patch embedding:
        Standard attention would process tokens after embedding.  Applying
        it at the pixel / band level before embedding lets the model weight
        entire spectral bands (e.g., down-weight noisy Green MS, up-weight
        NIR for stress detection) rather than per-spatial-location features.
        This is more interpretable for the thesis and physically motivated.

    Args:
        num_channels:    Number of input spectral bands (default 10).
        reduction_ratio: Bottleneck ratio; hidden dim = max(1, C // r).
    """

    def __init__(self, num_channels: int = 10, reduction_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(1, num_channels // reduction_ratio)
        self.squeeze   = nn.AdaptiveAvgPool2d(1)   # (B, C, H, W) → (B, C, 1, 1)
        self.excitation = nn.Sequential(
            nn.Flatten(),                           # (B, C)
            nn.Linear(num_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x * channel_weights, same shape (B, C, H, W)
        """
        w = self.excitation(self.squeeze(x))       # (B, C)
        return x * w.unsqueeze(-1).unsqueeze(-1)   # broadcast over H, W
