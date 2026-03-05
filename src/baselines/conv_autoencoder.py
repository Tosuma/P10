"""
Baseline 3: Convolutional Autoencoder — reconstruction-based anomaly detection.

Trained to minimise reconstruction error on all training patches (assumed
mostly normal).  At test time, patches that the AE cannot reconstruct well
(high MSE) are flagged as anomalous.

Architecture: symmetric 3-level encoder-decoder CNN.
  Encoder:  10 → 32 → 64 → 128 channels, each with 3×3 Conv + BN + ReLU + MaxPool2
  Bottleneck: (B, 128, 16, 16) for 128×128 input
  Decoder: ConvTranspose2d mirrors encoder with Upsample + Conv

All 10 channels are processed jointly so the model learns multi-spectral
reconstruction.  Anomalous patches deviate from the learned spectral-spatial
distribution and thus have high reconstruction error across multiple bands.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    def __init__(self, in_chans: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),   # 128→64
            nn.Conv2d(32, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),       # 64→32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),       # 32→16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 128, 16, 16)


class ConvDecoder(nn.Module):
    def __init__(self, out_chans: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),  nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, out_chans, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)   # (B, C, 128, 128)


class ConvAutoencoder(nn.Module):
    """
    Symmetric convolutional autoencoder for multispectral patch reconstruction.

    Args:
        in_chans: Number of input channels (default 10).
    """

    def __init__(self, in_chans: int = 10) -> None:
        super().__init__()
        self.encoder = ConvEncoder(in_chans)
        self.decoder = ConvDecoder(in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstruction (B, C, H, W)."""
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-patch MSE between input and reconstruction.

        Returns:
            (B,) float32 — mean squared error over all channels and pixels.
        """
        rec = self.forward(x)
        return (rec - x).pow(2).mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ConvAETrainer:
    """
    Training wrapper for ConvAutoencoder.

    Args:
        model:       ConvAutoencoder instance.
        train_loader, val_loader: DataLoaders.
        device:      torch.device.
        lr:          Adam learning rate.
        epochs:      Maximum training epochs.
        patience:    Early stopping patience on val loss.
        output_dir:  Directory to save checkpoints.
    """

    def __init__(
        self,
        model: ConvAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 10,
        output_dir: Path = Path("checkpoints/conv_ae"),
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs
        self.patience     = patience
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self) -> None:
        """Train with early stopping on validation loss."""
        best_val = float("inf")
        no_improve = 0

        for epoch in range(self.epochs):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()
            elapsed    = time.time() - t0

            print(f"[ConvAE {epoch:04d}/{self.epochs}]  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  ({elapsed:.1f}s)")

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), self.output_dir / "best.pth")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    def _train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            imgs = batch["image"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(imgs), imgs)
            loss.backward()
            self.optimizer.step()
            total += loss.item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            imgs = batch["image"].to(self.device)
            total += self.criterion(self.model(imgs), imgs).item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def score_dataset(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, list[str]]:
        """Return (scores, stems) for all patches in loader."""
        self.model.eval()
        all_scores, all_stems = [], []
        for batch in loader:
            imgs = batch["image"].to(self.device)
            s = self.model.anomaly_score(imgs).cpu().numpy()
            all_scores.append(s)
            all_stems.extend(batch["stem"])
        return np.concatenate(all_scores), all_stems

    @classmethod
    def load_model(
        cls, path: Path, device: torch.device, in_chans: int = 10
    ) -> ConvAutoencoder:
        model = ConvAutoencoder(in_chans=in_chans)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        return model
