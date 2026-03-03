"""
Stage 2 Flow Trainer.

Workflow
--------
1. Load frozen MAE encoder (weights from Stage 1 checkpoint).
2. For each training batch:
     a. Extract patch tokens with the frozen encoder (no grad).
     b. Strip CLS token → shape (B, N, D).
     c. Flatten to (B*N, D) — treat each patch as an independent sample.
     d. Compute NLL from the flow model and backpropagate.
3. Log NLL distribution statistics to track training progress.

Why freeze the encoder?
  The MAE encoder has already learned a rich representation of normal
  agricultural scenes.  Fine-tuning it jointly with the flow would allow the
  encoder to collapse toward an easy-to-model distribution, defeating the
  purpose.  Freezing ensures the flow must model the full richness of the
  encoder's feature space.

Anomaly score calibration
  During training on the normal data distribution, we record the mean and std
  of NLL scores and save them in the checkpoint.  At inference, we subtract the
  mean and divide by std (z-score) so that the anomaly score is interpretable:
  z > 2.5 → 2.5 σ above the training distribution → likely anomalous.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models.mae import MaskedAutoencoder
from src.models.flow_model import build_flow_model

logger = logging.getLogger(__name__)


class FlowTrainer:
    """
    Trainer for Stage 2 normalizing flow anomaly model.

    Parameters
    ----------
    mae_checkpoint : str
        Path to the best MAE checkpoint from Stage 1.
    output_dir : str
    device : torch.device
    feature_dim : int
        Must match the encoder embed_dim of the pretrained MAE.
    num_coupling_blocks : int
    subnet_hidden_dim : int
    epochs : int
    lr : float
    weight_decay : float
    use_wandb : bool
    """

    def __init__(
        self,
        mae_model: MaskedAutoencoder,
        mae_checkpoint: str,
        output_dir: str,
        device: torch.device,
        feature_dim: int = 384,
        num_coupling_blocks: int = 8,
        subnet_hidden_dim: int = 512,
        epochs: int = 100,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        use_wandb: bool = True,
        project_name: str = "plant-stress-flow",
    ):
        self.device = device
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Load frozen MAE encoder ────────────────────────────────────────
        state = torch.load(mae_checkpoint, map_location=device)
        mae_model.load_state_dict(state["model"])
        mae_model = mae_model.to(device)
        mae_model.eval()
        # Freeze all encoder parameters — no grad through encoder
        for p in mae_model.parameters():
            p.requires_grad = False
        self.mae_model = mae_model

        # ── Flow model ────────────────────────────────────────────────────
        self.flow = build_flow_model(
            feature_dim=feature_dim,
            num_coupling_blocks=num_coupling_blocks,
            subnet_hidden_dim=subnet_hidden_dim,
        ).to(device)

        # ── Optimiser (flow parameters only) ─────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.flow.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        self.scaler = GradScaler()

        # Statistics for score calibration
        self.score_mean: float = 0.0
        self.score_std: float = 1.0

        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=project_name, config={
                    "epochs": epochs, "lr": lr, "feature_dim": feature_dim,
                    "num_coupling_blocks": num_coupling_blocks,
                })
            except Exception as e:
                logger.warning(f"WandB init failed: {e}")

    # ── Feature extraction ────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run frozen MAE encoder and return patch feature vectors.

        Parameters
        ----------
        images : (B, C, H, W)

        Returns
        -------
        features : (B*N, D)  — N patch tokens per image, flattened into batch
        """
        tokens = self.mae_model.encode(images)  # (B, 1+N, D)
        patch_tokens = tokens[:, 1:, :]         # Strip CLS: (B, N, D)
        B, N, D = patch_tokens.shape
        return patch_tokens.reshape(B * N, D)

    # ── Training loop ─────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        self.flow.train()
        total_nll = 0.0
        n = 0
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            feats = self._extract_features(images)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                nll = self.flow.nll_score(feats).mean()

            self.scaler.scale(nll).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_nll += nll.item()
            n += 1
        return total_nll / max(1, n)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.flow.eval()
        all_nll = []
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            feats = self._extract_features(images)
            with autocast():
                nll = self.flow.nll_score(feats)
            all_nll.append(nll.cpu().float().numpy())
        all_nll_np = np.concatenate(all_nll)
        return float(all_nll_np.mean()), float(all_nll_np.std())

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        best_val_nll = float("inf")
        for epoch in range(self.epochs):
            t0 = time.time()
            train_nll = self._train_epoch(train_loader)
            val_mean, val_std = self._val_epoch(val_loader)
            self.scheduler.step()
            elapsed = time.time() - t0

            log = {
                "flow/train_nll": train_nll,
                "flow/val_nll_mean": val_mean,
                "flow/val_nll_std": val_std,
                "flow/lr": self.scheduler.get_last_lr()[0],
            }
            logger.info(
                f"Epoch {epoch:04d} | train_nll={train_nll:.4f} | "
                f"val_nll={val_mean:.4f}±{val_std:.4f} | {elapsed:.1f}s"
            )
            if self.wandb_run:
                try:
                    import wandb
                    wandb.log({**log, "epoch": epoch})
                except Exception:
                    pass

            if val_mean < best_val_nll:
                best_val_nll = val_mean
                # Update calibration stats from validation split
                self.score_mean = val_mean
                self.score_std  = max(val_std, 1e-6)
                self._save(epoch, val_mean, is_best=True)

        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    def _save(self, epoch: int, val_nll: float, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "flow": self.flow.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_nll": val_nll,
            "score_mean": self.score_mean,
            "score_std":  self.score_std,
        }
        torch.save(state, self.output_dir / f"flow_epoch_{epoch:04d}.pth")
        if is_best:
            torch.save(state, self.output_dir / "flow_best.pth")

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute z-scored anomaly scores for a batch of patches.

        Returns
        -------
        scores : (B, N) float32 — z-scored NLL per patch token
        """
        self.flow.eval()
        images = images.to(self.device)
        feats = self._extract_features(images)   # (B*N, D)
        raw_nll = self.flow.nll_score(feats)     # (B*N,)
        B = images.shape[0]
        N = feats.shape[0] // B
        raw_nll = raw_nll.reshape(B, N)
        # Z-score calibration
        z_scores = (raw_nll - self.score_mean) / self.score_std
        return z_scores.cpu().float()
