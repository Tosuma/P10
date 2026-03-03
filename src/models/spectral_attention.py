"""
Spectral Attention Module (SAM).

Motivation
----------
Our input tensor has heterogeneous channels: RGB (measured in visible
reflectance), multispectral bands (narrowband reflectance), and computed
vegetation indices (ratios).  Not all channels are equally informative for
detecting plant stress, and their relative importance may vary across field
conditions, crop types, and seasonal growth stages.

SAM learns a soft per-channel weighting (a gating vector) applied before the
ViT patch embedding.  This serves two roles:
  1. The model can suppress uninformative or noisy channels (e.g. blue band
     at high altitude with atmospheric scatter).
  2. The learned weights provide post-hoc interpretability: which spectral
     bands does the anomaly detector rely on most?  This is directly useful
     for the thesis narrative.

Architecture
------------
We follow the channel-attention design from SE-Net (Hu et al. 2018) adapted
for 1-D channel gating on spatial images:
    GlobalAvgPool → FC → ReLU → FC → Sigmoid → channel-wise multiply.
The bottleneck ratio r controls the capacity vs. over-fitting trade-off.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention for spectral band weighting.

    Parameters
    ----------
    num_channels : int
        Total number of input channels (RGB + MS bands + indices).
    reduction_ratio : int
        Bottleneck compression factor for the FC layers.  r=4 is a sensible
        default: expressive enough without introducing too many parameters.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 4):
        super().__init__()
        mid = max(1, num_channels // reduction_ratio)
        self.gate = nn.Sequential(
            # Global average pooling is done in forward(); this operates on (B, C)
            nn.Linear(num_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, num_channels),
            nn.Sigmoid(),             # Output ∈ (0, 1) per channel
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, C, H, W)

        Returns
        -------
        out : torch.Tensor  (B, C, H, W)  — channel-weighted input
        weights : torch.Tensor  (B, C)     — gate values for interpretability
        """
        # Global average pool over spatial dimensions → (B, C)
        gap = x.mean(dim=(-2, -1))
        weights = self.gate(gap)             # (B, C)
        out = x * weights[:, :, None, None]  # Broadcast over H, W
        return out, weights
