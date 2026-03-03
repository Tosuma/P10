"""
Vision Transformer encoder adapted for multi-channel remote sensing input.

Design choices
--------------
1. We use timm's ViT building blocks (PatchEmbed, Block) rather than re-
   implementing them, for correctness and to leverage timm's optimised
   FlashAttention kernels when available.

2. The first layer (PatchEmbed conv) is replaced with a new projection that
   accepts `in_channels` input channels instead of the standard 3, because
   our input has 7–11 channels (RGB + MS + vegetation indices).

3. Positional embeddings are 2-D sinusoidal (fixed) rather than learned.
   Rationale: drone patches come from different image locations during
   training.  Fixed sinusoidal PE generalises better to arbitrary patch
   positions without overfitting to the training image distribution.

4. Gradient checkpointing: enabled via `use_checkpoint=True` to trade
   compute for memory on HPC nodes with limited VRAM.

Supported configs
-----------------
    arch='vit_small_patch16': embed_dim=384, heads=6, depth=12
    arch='vit_base_patch16':  embed_dim=768, heads=12, depth=12

The MAE reconstruction head is defined in mae.py, not here.  The encoder
is responsible only for producing patch tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed


_VIT_CONFIGS = {
    "vit_small_patch16": dict(embed_dim=384, depth=12, num_heads=6,  mlp_ratio=4.0),
    "vit_base_patch16":  dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0),
    "vit_tiny_patch16":  dict(embed_dim=192, depth=12, num_heads=3,  mlp_ratio=4.0),
}


def build_2d_sincos_position_embedding(
    num_patches_h: int,
    num_patches_w: int,
    embed_dim: int,
    temperature: float = 10000.0,
) -> torch.Tensor:
    """
    Build fixed 2-D sinusoidal position embeddings for a grid of patches.

    We separate embed_dim/2 dimensions for height and width, following
    the approach in DINO / MAE.  The CLS token gets a zero embedding.

    Returns
    -------
    pos_embed : torch.Tensor  (1, 1 + num_patches_h * num_patches_w, embed_dim)
    """
    grid_h = torch.arange(num_patches_h, dtype=torch.float32)
    grid_w = torch.arange(num_patches_w, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
    # grid_h, grid_w each shape (num_patches_h, num_patches_w)

    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos PE"
    omega = torch.arange(embed_dim // 4, dtype=torch.float32)
    omega = temperature ** (-omega / (embed_dim // 4))

    out_h = grid_h.flatten()[:, None] * omega[None, :]  # (N, D/4)
    out_w = grid_w.flatten()[:, None] * omega[None, :]  # (N, D/4)

    pos_embed = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)],
        dim=1,
    )  # (N, D)

    # Prepend zero for CLS token
    cls_pe = torch.zeros(1, embed_dim)
    pos_embed = torch.cat([cls_pe, pos_embed], dim=0)  # (1+N, D)
    return pos_embed.unsqueeze(0)  # (1, 1+N, D)


class MultispectralViTEncoder(nn.Module):
    """
    ViT encoder accepting multi-channel raster input.

    Parameters
    ----------
    in_channels : int
        Number of input channels (RGB=3, MS=4, indices=4 → typically 7–11).
    img_size : int
        Spatial side length of input patches (e.g. 128).
    patch_size : int
        ViT patch token size (16 recommended; 8 for smaller patches).
    arch : str
        ViT variant key from _VIT_CONFIGS.
    use_checkpoint : bool
        Enable gradient checkpointing via torch.utils.checkpoint to save VRAM.
    drop_path_rate : float
        Stochastic depth rate for regularisation during pretraining.
    """

    def __init__(
        self,
        in_channels: int = 7,
        img_size: int = 128,
        patch_size: int = 16,
        arch: str = "vit_small_patch16",
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        cfg = _VIT_CONFIGS[arch]
        self.embed_dim = cfg["embed_dim"]
        self.depth = cfg["depth"]
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        # ── Patch embedding ────────────────────────────────────────────────
        # We override timm PatchEmbed's in_chans to accept our channel count.
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=self.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        grid_size = img_size // patch_size
        self.grid_size = grid_size  # e.g. 8 for 128/16

        # ── CLS token ─────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Fixed positional embeddings (not learned) ─────────────────────
        pos_embed = build_2d_sincos_position_embedding(
            grid_size, grid_size, self.embed_dim
        )
        self.register_buffer("pos_embed", pos_embed)  # (1, 1+N, D), no grad

        # ── Transformer blocks ────────────────────────────────────────────
        # Stochastic depth: linearly increases from 0 to drop_path_rate
        dpr = [drop_path_rate * i / (self.depth - 1) for i in range(self.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=cfg["num_heads"],
                mlp_ratio=cfg["mlp_ratio"],
                qkv_bias=True,
                drop_path=dpr[i],
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

    def forward_features(
        self,
        x: torch.Tensor,
        ids_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input patches into token sequences.

        Parameters
        ----------
        x : torch.Tensor  (B, C, H, W)
        ids_keep : torch.Tensor | None  (B, N_vis), dtype long
            Indices of visible (unmasked) patches in noise-shuffle order,
            as returned by _random_masking.  When None, all patches are
            encoded (Stage 2 / feature extraction).  Follows the FAIR MAE
            convention: tokens are gathered with torch.gather so the decoder
            receives them in noise-shuffle order and ids_restore correctly
            undoes the shuffle.

        Returns
        -------
        tokens : torch.Tensor  (B, 1 + num_visible_patches, D)
        """
        B = x.shape[0]
        # Patch embedding → (B, N, D)
        tokens = self.patch_embed(x)

        # Add positional embeddings to patch tokens (skip CLS pe here)
        tokens = tokens + self.pos_embed[:, 1:, :]

        # Keep only visible patches (MAE encoder sees unmasked tokens only).
        # torch.gather preserves noise-shuffle order so that ids_restore in the
        # decoder correctly maps tokens back to their original spatial positions.
        if ids_keep is not None:
            tokens = torch.gather(
                tokens, dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            )

        # Prepend CLS token with its positional embedding
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+N_vis, D)

        # Transformer blocks
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                tokens = checkpoint(blk, tokens, use_reentrant=False)
            else:
                tokens = blk(tokens)

        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass without masking (feature extraction / Stage 2)."""
        return self.forward_features(x)
