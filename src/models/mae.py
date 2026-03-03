"""
Masked Autoencoder (MAE) for multispectral drone imagery.

Implementation follows He et al. 2022 "Masked Autoencoders Are Scalable
Vision Learners" with the following adaptations:
  - Multi-channel input (RGB + MS + vegetation indices) via MultispectralViTEncoder
  - SpectralAttention module before patch embedding
  - Lightweight decoder (fewer layers than encoder) to reconstruct masked patches
  - Target: raw pixel values of the masked patches (no normalisation target
    unlike the original paper which uses per-patch mean/std normalisation —
    we avoid this because spectral channels have very different dynamic ranges
    and per-patch normalisation would destroy the inter-channel ratios that
    encode plant physiology)

Masking strategy
----------------
Random masking at ratio r=0.75 (75% of patches masked per image).  This
forces the encoder to develop a rich internal representation because it
cannot rely on local self-similarity to predict masked content — it must
learn global scene structure.  75% is the optimal ratio reported by He et al.
for ViT-Base; we keep this for consistency.

Loss
----
MSE over the pixel values of the masked patches only.  Including visible
patches in the loss would trivially reduce loss (they are just passed through)
and would not contribute useful gradient signal.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from src.models.spectral_attention import SpectralAttention
from src.models.vit_encoder import MultispectralViTEncoder, _VIT_CONFIGS


class MAEDecoder(nn.Module):
    """
    Lightweight MAE decoder: a shallow Transformer that reconstructs pixel
    values for masked patches from the encoder's output tokens + mask tokens.

    We use a significantly shallower decoder (4 blocks, smaller dim) than the
    encoder because the decoder only operates during pretraining — after
    Stage 1 it is discarded.  A cheap decoder lets the encoder carry the full
    representational burden, as intended by the MAE design.
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        num_patches: int = 64,
        patch_size: int = 16,
        in_channels: int = 7,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        from timm.models.vision_transformer import Block

        self.patch_size = patch_size
        self.in_channels = in_channels
        patch_dim = patch_size * patch_size * in_channels

        # Project encoder dim → decoder dim
        self.proj = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # Learnable mask token (replaces removed patches in decoder input)
        # Learned rather than zero so the decoder knows where masked positions
        # are without requiring positional encoding to disambiguate.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder positional embedding (learnable, simpler than fixed for
        # the decoder since it sees all positions always)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches, decoder_embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
            )
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)

        # Linear projection to pixel space
        self.head = nn.Linear(decoder_embed_dim, patch_dim, bias=True)

    def forward(
        self,
        encoded_tokens: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct all patches (both visible and masked) from encoder output.

        Parameters
        ----------
        encoded_tokens : (B, 1+N_vis, D_enc)  — CLS + visible patch tokens
        mask : (B, N)  — boolean, True = masked patch
        ids_restore : (B, N)  — permutation to undo the masking shuffle

        Returns
        -------
        pred : (B, N, patch_size*patch_size*in_channels)
        """
        B, N_enc, _ = encoded_tokens.shape
        N = mask.shape[1]         # Total number of patches
        N_vis = N_enc - 1         # Subtract CLS token

        # Project to decoder dim
        tokens = self.proj(encoded_tokens)  # (B, 1+N_vis, D_dec)

        # Expand mask token to fill masked positions
        n_masked = N - N_vis
        mask_tokens = self.mask_token.expand(B, n_masked, -1)  # (B, N_mask, D)

        # Unshuffle: place visible tokens and mask tokens back in original order
        # tokens[:, 1:] are the visible patches (strip CLS)
        vis_tokens = tokens[:, 1:, :]               # (B, N_vis, D)
        all_tokens = torch.cat([vis_tokens, mask_tokens], dim=1)  # (B, N, D)

        # ids_restore maps the shuffled order back to original spatial order
        ids_restore_exp = ids_restore.unsqueeze(-1).expand(-1, -1, all_tokens.shape[-1])
        all_tokens = torch.gather(all_tokens, dim=1, index=ids_restore_exp)

        # Re-prepend CLS token
        all_tokens = torch.cat([tokens[:, :1, :], all_tokens], dim=1)  # (B, 1+N, D)

        # Add positional embeddings
        all_tokens = all_tokens + self.pos_embed

        # Decoder Transformer blocks
        for blk in self.blocks:
            all_tokens = blk(all_tokens)
        all_tokens = self.norm(all_tokens)

        # Predict pixel values (drop CLS)
        pred = self.head(all_tokens[:, 1:, :])  # (B, N, patch_dim)
        return pred


class MaskedAutoencoder(nn.Module):
    """
    Full MAE: SpectralAttention → ViT Encoder → MAE Decoder.

    Exposes `encode()` for feature extraction (Stage 2 usage) and
    `forward()` for pretraining (returns loss + predicted pixel values).

    Parameters
    ----------
    in_channels : int
    img_size : int
    patch_size : int
    arch : str
        ViT arch key (see vit_encoder._VIT_CONFIGS)
    masking_ratio : float
        Fraction of patches to mask during pretraining.
    decoder_embed_dim : int
    decoder_depth : int
    use_checkpoint : bool
    sa_reduction_ratio : int
        SpectralAttention bottleneck ratio.
    """

    def __init__(
        self,
        in_channels: int = 7,
        img_size: int = 128,
        patch_size: int = 16,
        arch: str = "vit_small_patch16",
        masking_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        use_checkpoint: bool = True,
        sa_reduction_ratio: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio

        # ── Spectral Attention ────────────────────────────────────────────
        self.spectral_attention = SpectralAttention(
            num_channels=in_channels,
            reduction_ratio=sa_reduction_ratio,
        )

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder = MultispectralViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            arch=arch,
            use_checkpoint=use_checkpoint,
        )
        enc_dim = self.encoder.embed_dim
        num_patches = self.encoder.num_patches

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = MAEDecoder(
            encoder_embed_dim=enc_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            num_patches=num_patches,
            patch_size=patch_size,
            in_channels=in_channels,
        )

    # ── Masking ───────────────────────────────────────────────────────────────

    def _random_masking(
        self, B: int, N: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random masking indices following the FAIR MAE convention.

        Convention (matches facebookresearch/mae):
          - Patches are ranked by random noise (ascending).
          - The first N_vis (lowest noise) are VISIBLE; the rest are MASKED.
          - ids_keep holds the visible patch indices in noise-shuffle order so
            that the decoder can undo the shuffle with ids_restore.

        Returns
        -------
        mask : (B, N) bool — True = masked, in spatial (original) order
        ids_keep : (B, N_vis) long — visible patch indices in noise order
        ids_restore : (B, N) long — inverse permutation to recover spatial order
        """
        n_mask = int(N * self.masking_ratio)
        n_vis = N - n_mask
        noise = torch.rand(B, N, device=device)
        ids_shuffle = noise.argsort(dim=1)          # ascending: low noise → visible
        ids_restore = ids_shuffle.argsort(dim=1)    # inverse permutation

        # Visible patch indices (first n_vis in noise-sorted order)
        ids_keep = ids_shuffle[:, :n_vis]           # (B, N_vis)

        # Spatial-order boolean mask: True = masked
        # scatter False at visible positions; everything else stays True
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_keep, False)

        return mask, ids_keep, ids_restore

    # ── Patch utilities ───────────────────────────────────────────────────────

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, C, H, W) image to (B, N, patch_size²×C) patch tokens.
        Used to compute the reconstruction target.
        """
        p = self.patch_size
        # (B, C, H, W) → (B, N, p*p*C) via einops
        patches = rearrange(
            x, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", p1=p, p2=p
        )
        return patches

    # ── Forward passes ────────────────────────────────────────────────────────

    def encode(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2 feature extraction: encode ALL patches (no masking).

        Parameters
        ----------
        x : (B, C, H, W)
        return_attn_weights : bool
            If True, also return spectral attention weights (B, C) for
            interpretability analysis.

        Returns
        -------
        tokens : (B, 1+N, D)  — CLS + all patch tokens after encoder norm
        [, attn_weights : (B, C)]
        """
        x_att, attn_weights = self.spectral_attention(x)
        tokens = self.encoder.forward_features(x_att, ids_keep=None)
        if return_attn_weights:
            return tokens, attn_weights
        return tokens

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAE pretraining forward pass.

        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        loss : scalar — MSE on masked patches only
        pred : (B, N, patch_dim) — decoder output (all patches)
        mask : (B, N) bool — True = masked positions
        """
        B, C, H, W = x.shape
        N = self.encoder.num_patches
        device = x.device

        # Spectral attention gating
        x_att, _ = self.spectral_attention(x)

        # Random masking — returns visible indices in noise order (FAIR convention)
        mask, ids_keep, ids_restore = self._random_masking(B, N, device)

        # Encode only visible patches (gathered in noise-shuffle order)
        encoded = self.encoder.forward_features(x_att, ids_keep=ids_keep)

        # Decode: reconstruct all N patches
        pred = self.decoder(encoded, mask, ids_restore)

        # Compute loss: MSE on masked patches only
        target = self._patchify(x)       # (B, N, patch_dim) — original pixels
        loss = self._mae_loss(pred, target, mask)

        return loss, pred, mask

    @staticmethod
    def _mae_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss on masked positions only.

        We deliberately do NOT normalise the target per-patch (unlike the
        original MAE paper) because inter-channel ratios carry agronomic
        meaning that normalisation would destroy.
        """
        diff = (pred - target) ** 2   # (B, N, patch_dim)
        loss_per_patch = diff.mean(dim=-1)  # (B, N)
        # Average only over masked patches
        loss = (loss_per_patch * mask.float()).sum() / mask.float().sum()
        return loss
