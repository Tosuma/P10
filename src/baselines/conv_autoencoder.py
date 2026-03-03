"""
Baseline 3: Convolutional Autoencoder (CAE) for reconstruction-based anomaly detection.

Approach
--------
Train a standard encoder–decoder convolutional autoencoder on normal (unlabelled)
image patches.  At inference, the reconstruction error (MSE per pixel) is used
as the anomaly score.

Rationale for reconstruction-based AD
  The intuition: an autoencoder trained only on normal data will faithfully
  reconstruct normal patches but struggle with anomalous ones (out-of-
  distribution), resulting in high reconstruction error.

Known limitation (and why MAE + Flow is expected to outperform)
  Autoencoders tend to have a smooth reconstruction loss landscape, which means
  they can sometimes reconstruct anomalies well (under-detect) or fail on
  complex normal textures (over-detect).  The flow model in Stage 2 addresses
  this by explicitly modelling the density of the feature space.

Architecture
  Simple U-Net-inspired encoder–decoder with skip connections.
  Skip connections improve reconstruction fidelity on normal patches
  (sharper reconstructions) but do NOT help anomaly patches — the bottleneck
  remains the information bottleneck that drives the AD behaviour.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _deconv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    5-level encoder–decoder with skip connections.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    base_channels : int
        Feature channels at the first encoder level.  Doubles at each level.
    latent_dim : int
        Bottleneck channel dimension.  Larger = less compression = easier
        reconstruction (weaker AD signal).  Default 128 is a good trade-off.
    """

    def __init__(
        self,
        in_channels: int = 7,
        base_channels: int = 32,
        latent_dim: int = 128,
    ):
        super().__init__()
        c = base_channels

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc1 = _conv_bn_relu(in_channels, c)        # 128 → 128
        self.enc2 = _conv_bn_relu(c, c * 2, stride=2)   # 128 → 64
        self.enc3 = _conv_bn_relu(c * 2, c * 4, stride=2)  # 64 → 32
        self.enc4 = _conv_bn_relu(c * 4, c * 8, stride=2)  # 32 → 16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            _conv_bn_relu(c * 8, latent_dim),
            _conv_bn_relu(latent_dim, latent_dim),
        )

        # ── Decoder ────────────────────────────────────────────────────────
        self.dec4 = _deconv_bn_relu(latent_dim, c * 8)
        self.dec3 = _deconv_bn_relu(c * 8 + c * 8, c * 4)   # + skip from enc4
        self.dec2 = _deconv_bn_relu(c * 4 + c * 4, c * 2)   # + skip from enc3
        self.dec1 = _deconv_bn_relu(c * 2 + c * 2, c)        # + skip from enc2

        # Final reconstruction: sigmoid to match normalised [0,1] input
        self.head = nn.Sequential(
            nn.Conv2d(c + c, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z  = self.bottleneck(e4)
        return z, e1, e2, e3, e4

    def decode(
        self,
        z: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        e4: torch.Tensor,
    ) -> torch.Tensor:
        d4 = self.dec4(z)
        d3 = self.dec3(torch.cat([d4, e4], dim=1))
        d2 = self.dec2(torch.cat([d3, e3], dim=1))
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        out = self.head(torch.cat([d1, e1], dim=1))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, e1, e2, e3, e4 = self.encode(x)
        return self.decode(z, e1, e2, e3, e4)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction error as anomaly score.

        Returns
        -------
        score_map : (B, H, W)  — per-pixel MSE (mean over channels)
        """
        self.eval()
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=1)  # (B, H, W)


# ── Trainer ───────────────────────────────────────────────────────────────────

class CAETrainer:
    """
    Training wrapper for the ConvAutoencoder baseline.

    Parameters
    ----------
    model : ConvAutoencoder
    device : torch.device
    lr : float
    epochs : int
    output_dir : str
    """

    def __init__(
        self,
        model: ConvAutoencoder,
        device: torch.device,
        lr: float = 1e-3,
        epochs: int = 100,
        output_dir: str = "outputs/cae",
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.output_dir = output_dir
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        import os; os.makedirs(output_dir, exist_ok=True)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
        best_val = float("inf")
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                recon = self.model(imgs)
                loss = F.mse_loss(recon, imgs)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        imgs = batch["image"].to(self.device, non_blocking=True)
                        recon = self.model(imgs)
                        val_loss += F.mse_loss(recon, imgs).item()
                val_loss /= len(val_loader)
                self.scheduler.step(val_loss)
                print(f"Epoch {epoch:04d} | train={train_loss:.5f} | val={val_loss:.5f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(self.model.state_dict(),
                               f"{self.output_dir}/cae_best.pth")
            else:
                print(f"Epoch {epoch:04d} | train={train_loss:.5f}")
