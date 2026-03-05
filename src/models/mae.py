"""
Masked Autoencoder (MAE) with multispectral ViT backbone and spectral attention.

Follows He et al., "Masked Autoencoders Are Scalable Vision Learners",
CVPR 2022, with the following thesis-specific adaptations:

  - 10-channel input (RGB + MS + VI) instead of 3-channel RGB.
  - SpectralAttention module before patch embedding to learn per-band weights.
  - Fixed sinusoidal PE for spatial generalisation.
  - norm_pix_loss=False: we do NOT normalise each masked patch to zero mean /
    unit std before computing the reconstruction loss.  Rationale: per-patch
    normalisation would destroy inter-channel spectral ratios (e.g., NDVI is
    NIR/R — normalising by local statistics removes this ratio information).
  - Masking ratio 0.75 following He et al. 2022.

The MAE is trained in Stage 1.  After pretraining, only the encoder (and
SpectralAttention) are kept.  The decoder is discarded.  The frozen encoder
is used as a feature extractor for the Stage 2 normalising flow.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from src.models.spectral_attention import SpectralAttention
from src.models.vit_encoder import ViTEncoder, build_2d_sinusoidal_pe


# ---------------------------------------------------------------------------
# MAE Model
# ---------------------------------------------------------------------------

class MAEModel(nn.Module):
    """
    Full Masked Autoencoder: encoder + lightweight decoder.

    Forward pass:
        imgs → SpectralAttention → PatchEmbed → random mask (75%) →
        ViT Encoder (visible tokens only) → Decoder (all positions,
        masked positions use mask token embedding) → linear head →
        MSE loss on masked patches.

    Inference / feature extraction:
        Use encode() which skips masking and decoder entirely.

    Args:
        encoder:             Pre-built ViTEncoder.
        spectral_attention:  Pre-built SpectralAttention.
        decoder_embed_dim:   Decoder hidden dimension (default 256).
        decoder_depth:       Number of decoder transformer blocks (default 4).
        decoder_num_heads:   Decoder attention heads (default 8).
        mlp_ratio:           MLP expansion factor in decoder (default 4.0).
        mask_ratio:          Fraction of patches to mask (default 0.75).
        norm_pix_loss:       Normalise target patches before MSE (default False).
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        spectral_attention: SpectralAttention,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.spectral_attn = spectral_attention
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        E = encoder.embed_dim
        patch_size = encoder.patch_embed.patch_size
        in_chans   = encoder.patch_embed.proj.in_channels
        self.patch_dim = patch_size * patch_size * in_chans   # target dim per masked patch

        # Decoder input projection: encoder_dim → decoder_dim
        self.decoder_embed = nn.Linear(E, decoder_embed_dim, bias=True)

        # Learnable mask token (one vector substituted for each masked position)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional embedding (same fixed PE, different dim)
        n_h = n_w = encoder.patch_embed.img_size // patch_size
        dec_pe = build_2d_sinusoidal_pe(n_h, n_w, decoder_embed_dim)
        self.register_buffer("decoder_pos_embed", dec_pe)   # (1, N, dec_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Split image into patches and flatten each patch.

        Args:
            imgs: (B, C, H, W)
        Returns:
            (B, N, patch_dim) where patch_dim = P*P*C
        """
        B, C, H, W = imgs.shape
        p = self.encoder.patch_embed.patch_size
        h = H // p
        w = W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum("bchpwq->bhwcpq", x)
        x = x.reshape(B, h * w, C * p * p)
        return x

    def _random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask mask_ratio fraction of patch tokens.

        Strategy: assign uniform random noise to each token, then sort.
        The first N*(1-mask_ratio) tokens (lowest noise) are kept visible.
        This avoids any spatial bias in which patches are masked.

        Args:
            x: (B, N, D) token embeddings
        Returns:
            x_visible:   (B, N_vis, D) — kept tokens
            mask:        (B, N) bool — True where masked
            ids_restore: (B, N) long — argsort to restore original order
        """
        B, N, D = x.shape
        n_keep = int(N * (1.0 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)          # (B, N) uniform
        ids_shuffle  = torch.argsort(noise, dim=1)          # ascending: low = keep
        ids_restore  = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :n_keep]                  # (B, n_keep)
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Build bool mask: True = masked (removed)
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_keep, False)

        return x_visible, mask, ids_restore

    # ------------------------------------------------------------------
    # Forward / encode / reconstruct
    # ------------------------------------------------------------------

    def forward(
        self, imgs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full MAE forward pass: encode visible tokens, decode all positions,
        compute MSE loss on masked patches only.

        Args:
            imgs: (B, C, H, W)
        Returns:
            loss: scalar MSE on masked patches
            pred: (B, N_masked, patch_dim) reconstructed patch pixels
            mask: (B, N) bool — True where masked (where loss is computed)
        """
        # 1. Spectral attention: reweight bands
        imgs = self.spectral_attn(imgs)

        # 2. Patch embed + add PE
        tokens = self.encoder.patch_embed(imgs) + self.encoder.pos_embed   # (B, N, E)

        # 3. Random masking
        tokens_vis, mask, ids_restore = self._random_masking(tokens, self.mask_ratio)

        # 4. Encode visible tokens
        for blk in self.encoder.blocks:
            tokens_vis = blk(tokens_vis)
        tokens_vis = self.encoder.norm(tokens_vis)                          # (B, N_vis, E)

        # 5. Project to decoder dim
        tokens_vis = self.decoder_embed(tokens_vis)                         # (B, N_vis, dec_E)

        # 6. Restore full sequence: insert mask tokens at masked positions.
        # Visible tokens are placed back at their original positions; masked
        # positions receive the learnable mask_token embedding.
        B, N_vis, dec_E = tokens_vis.shape
        N = mask.shape[1]
        full = self.mask_token.expand(B, N, -1).clone()                    # (B, N, dec_E)
        full[~mask] = tokens_vis.reshape(-1, dec_E)

        full = full + self.decoder_pos_embed                               # (B, N, dec_E)

        # 7. Decode
        for blk in self.decoder_blocks:
            full = blk(full)
        full = self.decoder_norm(full)
        pred_all = self.decoder_pred(full)                                 # (B, N, patch_dim)

        # 8. Loss: MSE on masked patches only
        target = self._patchify(imgs)                                      # (B, N, patch_dim)

        if self.norm_pix_loss:
            # Per-patch normalisation (not used by default — see module docstring)
            mean = target.mean(dim=-1, keepdim=True)
            std  = target.var(dim=-1, keepdim=True).add_(1e-6).sqrt()
            target = (target - mean) / std

        pred_masked   = pred_all[mask]                                     # (N_masked_total, patch_dim)
        target_masked = target[mask]
        loss = (pred_masked - target_masked).pow(2).mean()

        return loss, pred_masked, mask

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Full encoder forward pass WITHOUT masking.

        Used by the Stage 2 flow model and evaluation to extract rich
        per-patch feature representations from the trained encoder.

        Args:
            imgs: (B, C, H, W)
        Returns:
            (B, N, embed_dim) — features for all N patches
        """
        x = self.spectral_attn(imgs)
        tokens = self.encoder.patch_embed(x) + self.encoder.pos_embed
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        return self.encoder.norm(tokens)

    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode + mask + decode + un-patchify for visual inspection.

        Returns a reconstructed image where masked patches are predicted
        and visible patches are copied from the input.

        Args:
            imgs: (B, C, H, W)
        Returns:
            (B, C, H, W) reconstructed image (float32, same scale as input)
        """
        loss, pred_masked, mask = self.forward(imgs)

        B = imgs.shape[0]
        in_c = self.encoder.patch_embed.proj.in_channels
        p = self.encoder.patch_embed.patch_size

        target = self._patchify(imgs)                   # (B, N, patch_dim)
        target[mask] = pred_masked.float()              # overwrite masked positions

        # Un-patchify
        N = target.shape[1]
        h = w = int(N ** 0.5)
        x = target.reshape(B, h, w, in_c, p, p)
        x = torch.einsum("bhwcpq->bchpwq", x)
        x = x.reshape(B, in_c, h * p, w * p)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ARCH_CONFIGS = {
    "vit_small_patch16": dict(embed_dim=384, depth=12, num_heads=6,  patch_size=16),
    "vit_base_patch16":  dict(embed_dim=768, depth=12, num_heads=12, patch_size=16),
}


def build_mae(
    arch: str = "vit_small_patch16",
    in_chans: int = 10,
    img_size: int = 128,
    use_checkpoint: bool = True,
    mask_ratio: float = 0.75,
    decoder_embed_dim: int = 256,
    decoder_depth: int = 4,
    decoder_num_heads: int = 8,
    **kwargs,
) -> MAEModel:
    """
    Build an MAEModel from a named architecture string.

    Args:
        arch:             One of "vit_small_patch16", "vit_base_patch16".
        in_chans:         Input channels (default 10).
        img_size:         Spatial side length (default 128).
        use_checkpoint:   Enable gradient checkpointing in encoder.
        mask_ratio:       Fraction of patches to mask (default 0.75).
        decoder_embed_dim, decoder_depth, decoder_num_heads: Decoder config.

    Returns:
        MAEModel ready for training.
    """
    if arch not in _ARCH_CONFIGS:
        raise ValueError(f"Unknown arch {arch!r}. Choose from {list(_ARCH_CONFIGS)}")

    cfg = _ARCH_CONFIGS[arch]
    encoder = ViTEncoder(
        img_size=img_size,
        patch_size=cfg["patch_size"],
        in_chans=in_chans,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        use_checkpoint=use_checkpoint,
    )
    spectral_attn = SpectralAttention(num_channels=in_chans)
    return MAEModel(
        encoder=encoder,
        spectral_attention=spectral_attn,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mask_ratio=mask_ratio,
        **kwargs,
    )
