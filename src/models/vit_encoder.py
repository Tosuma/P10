"""
Vision Transformer encoder with arbitrary-channel input and fixed 2-D
sinusoidal positional encoding.

Design decisions:
  - Fixed sinusoidal PE (not learned): generalises better across spatial
    positions. He et al. 2022 (MAE) found fixed PE sufficient for
    reconstruction; learned PE can overfit to training patch positions in
    finite datasets like ours (~220k patches).
  - Arbitrary in_chans: timm's default ViT patch embed assumes 3 channels;
    we replace the first Conv2d to accept 10 (RGB + MS + VI).
  - Gradient checkpointing: halves activation memory at the cost of one
    extra forward pass per block; essential to fit ViT-Small on 24 GB GPUs
    with large batch sizes.
  - No CLS token: MAE operates on pure patch sequences; the CLS token is
    not needed and would complicate masking logic.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """
    2-D patch embedding that accepts an arbitrary number of input channels.

    A single Conv2d with kernel_size=patch_size and stride=patch_size splits
    the image into non-overlapping patches and linearly projects each to
    embed_dim, equivalent to per-patch linear projection in the ViT paper.

    Args:
        img_size:   Spatial side length of the square input (default 128).
        patch_size: Spatial side length of each patch (default 16).
        in_chans:   Number of input channels (default 10).
        embed_dim:  Output embedding dimension (default 384 = ViT-Small).
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 10,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)                    # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)    # (B, N, E)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def build_2d_sinusoidal_pe(
    num_patches_h: int,
    num_patches_w: int,
    embed_dim: int,
) -> torch.Tensor:
    """
    Construct fixed 2-D sinusoidal positional encoding.

    Half of embed_dim encodes the row position, half encodes the column
    position.  This factored representation naturally captures 2-D structure
    without requiring any learned parameters.

    Returns:
        (1, num_patches_h * num_patches_w, embed_dim) float32 tensor,
        suitable for addition to patch embeddings.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sinusoidal PE"
    d = embed_dim // 2  # half for row, half for col

    # Standard 1-D sinusoidal for each axis
    def _1d_pe(n_pos: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(n_pos, d_model)
        pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (n_pos, d_model)

    pe_h = _1d_pe(num_patches_h, d)  # (H, d)
    pe_w = _1d_pe(num_patches_w, d)  # (W, d)

    # Expand and concatenate along spatial grid
    pe_h = pe_h.unsqueeze(1).expand(-1, num_patches_w, -1)  # (H, W, d)
    pe_w = pe_w.unsqueeze(0).expand(num_patches_h, -1, -1)  # (H, W, d)

    pe = torch.cat([pe_h, pe_w], dim=-1)           # (H, W, embed_dim)
    pe = pe.view(1, num_patches_h * num_patches_w, embed_dim)
    return pe


# ---------------------------------------------------------------------------
# ViT Encoder
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for multispectral MAE pretraining.

    Supports:
      - Arbitrary input channels via PatchEmbed2D
      - Fixed 2-D sinusoidal PE (see build_2d_sinusoidal_pe)
      - Gradient checkpointing for HPC memory efficiency
      - Optional masking: pass a bool mask to forward() to receive only
        the visible (unmasked) token embeddings

    Default configuration: ViT-Small
      embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 10,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.num_patches
        n_h = n_w = img_size // patch_size

        # Register as buffer (not parameter): moves with .to(device) but not
        # updated by the optimiser.
        pe = build_2d_sinusoidal_pe(n_h, n_w, embed_dim)
        self.register_buffer("pos_embed", pe)   # (1, N, E)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=nn.LayerNorm,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.use_checkpoint = use_checkpoint
        self.embed_dim = embed_dim
        self.num_patches = n_patches

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, C, H, W) input image tensor.
            mask: (B, N) bool tensor. True = masked (removed from sequence).
                  If None, all tokens are kept.
        Returns:
            If mask is None: (B, N, embed_dim)
            If mask is provided: (B, N_visible, embed_dim) — only unmasked tokens.
        """
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, E)

        if mask is not None:
            # Keep only visible (unmasked) tokens
            tokens = tokens[~mask].view(x.size(0), -1, self.embed_dim)

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                tokens = checkpoint(blk, tokens, use_reentrant=False)
            else:
                tokens = blk(tokens)

        return self.norm(tokens)
