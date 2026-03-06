"""
Stage 2: Normalizing flow training on frozen MAE encoder features.

The MAE encoder is FROZEN.  Features are optionally pre-computed and cached
in CPU memory so each training iteration is a pure flow forward/backward,
without re-running the large ViT encoder.  This makes Stage 2 training
significantly faster (5–10×) compared to running the encoder every step.

Design choice — feature caching:
  With ~220k training patches and embed_dim=384, the full feature tensor is
  220000 × 64 × 384 ≈ 21 GB (float32).  This exceeds typical RAM budgets.
  We instead cache flattened per-patch features (220000 × 384) by using
  mean-pooled encoder output (1 vector per 128×128 patch) OR process all
  N=64 tokens independently.  Default: flatten all tokens → 220000×64 rows,
  each 384-dim.  With float32 this is ~21 GB — feasible only on large-RAM
  HPC nodes.  On memory-constrained nodes, set cache_features=False to
  stream from the encoder live.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

from src.models.flow_model import FlowModel
from src.models.mae import MAEModel


class FlowTrainer:
    """
    Trainer for the Stage 2 normalizing flow.

    Args:
        cfg:          OmegaConf DictConfig.  Expected fields:
                        cfg.flow.{cache_features}
                        cfg.training.{flow_epochs, flow_lr}
                        cfg.logging.{use_wandb, project, output_dir}
        flow_model:   FlowModel to train.
        mae_model:    Pre-trained MAEModel (encoder will be frozen).
        train_loader: DataLoader over training patches.
        val_loader:   DataLoader over validation patches.
        device:       torch.device.
    """

    def __init__(
        self,
        cfg,
        flow_model: FlowModel,
        mae_model: MAEModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.flow = flow_model.to(device)
        self.mae  = mae_model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device = device

        # Freeze MAE encoder completely
        for p in self.mae.parameters():
            p.requires_grad_(False)
        self.mae.eval()

        self.output_dir = Path(cfg.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            self.flow.parameters(),
            lr=cfg.training.get("flow_lr", 1e-3),
        )
        scheduler_type = cfg.training.get("scheduler", "cosine")
        epochs = cfg.training.get("flow_epochs", 100)
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-5
            )
        else:
            self.scheduler = None

        self.use_wandb = cfg.logging.get("use_wandb", False)
        if self.use_wandb:
            import wandb
            from omegaconf import OmegaConf
            wandb.init(
                project=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Full Stage 2 training loop."""
        epochs = self.cfg.training.get("flow_epochs", 100)
        cache  = self.cfg.flow.get("cache_features", True)

        # Pre-compute features once if caching is enabled
        train_feats = val_feats = None
        if cache:
            print("Pre-computing training features …")
            train_feats = self._extract_features(self.train_loader).to(self.device)
            print(f"  cached {train_feats.shape[0]:,} patch features on GPU "
                  f"({train_feats.numel() * 4 / 1e6:.0f} MB)")
            print("Pre-computing validation features …")
            val_feats = self._extract_features(self.val_loader).to(self.device)
            print(f"  cached {val_feats.shape[0]:,} patch features on GPU "
                  f"({val_feats.numel() * 4 / 1e6:.0f} MB)")

        best_val = float("inf")
        for epoch in range(epochs):
            t0 = time.time()
            train_m = self._train_epoch(epoch, train_feats)
            val_m   = self._val_epoch(epoch, val_feats)
            if self.scheduler is not None:
                self.scheduler.step()
            elapsed = time.time() - t0

            print(
                f"[{epoch:04d}/{epochs}] "
                f"train_nll={train_m['nll']:.4f}  "
                f"val_nll={val_m['nll']:.4f}  "
                f"time={elapsed:.1f}s"
            )
            if self.use_wandb:
                import wandb
                wandb.log({"train/nll": train_m["nll"], "val/nll": val_m["nll"]}, step=epoch)

            if val_m["nll"] < best_val:
                best_val = val_m["nll"]
                self._save_checkpoint(epoch, fname="best.pth")

        self._save_checkpoint(epochs - 1, fname="final.pth")
        if self.use_wandb:
            import wandb
            wandb.finish()

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_features(self, loader: DataLoader) -> torch.Tensor:
        """
        Run MAE encoder on every batch, mean-pool tokens per patch, return on CPU.

        Mean-pooling (one 384-dim vector per patch) reduces cache size 64×
        vs. storing every token (~330 MB vs ~21 GB), allowing the full cache
        to be moved to GPU once and eliminating per-step CPU→GPU transfers.
        Token-level scoring at inference uses score_patches() directly and is
        unaffected by this choice.

        Returns:
            (N_patches, embed_dim) float32 on CPU.
        """
        all_feats = []
        for batch in loader:
            imgs = batch["image"].to(self.device, non_blocking=True)
            with autocast("cuda"):
                feats = self.mae.encode(imgs)        # (B, N_tok, D)
            all_feats.append(feats.mean(dim=1).cpu())   # (B, D) — pool over tokens
        return torch.cat(all_feats, dim=0)

    def _train_epoch(
        self, epoch: int, cached_feats: Optional[torch.Tensor]
    ) -> dict[str, float]:
        self.flow.train()
        total_nll = 0.0
        n_steps   = 0
        bs = self.cfg.data.get("batch_size", 512)

        if cached_feats is not None:
            # Train from cached feature tensor (already on GPU)
            idx = torch.randperm(cached_feats.shape[0], device=self.device)
            for start in range(0, cached_feats.shape[0], bs):
                chunk = cached_feats[idx[start: start + bs]]   # no .to() — already on GPU
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.flow.flow_loss(chunk)
                loss.backward()
                self.optimizer.step()
                total_nll += loss.item()
                n_steps   += 1
        else:
            # Stream from DataLoader + encoder
            for batch in self.train_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                with torch.no_grad():
                    with autocast("cuda"):
                        feats = self.mae.encode(imgs)     # (B, N, D)
                B, N, D = feats.shape
                flat = feats.reshape(B * N, D).float()

                self.optimizer.zero_grad(set_to_none=True)
                loss = self.flow.flow_loss(flat)
                loss.backward()
                self.optimizer.step()
                total_nll += loss.item()
                n_steps   += 1

        return {"nll": total_nll / max(n_steps, 1)}

    @torch.no_grad()
    def _val_epoch(
        self, epoch: int, cached_feats: Optional[torch.Tensor]
    ) -> dict[str, float]:
        self.flow.eval()
        total_nll = 0.0
        n_steps   = 0
        bs = self.cfg.data.get("batch_size", 512)

        if cached_feats is not None:
            for start in range(0, cached_feats.shape[0], bs):
                chunk = cached_feats[start: start + bs]   # already on GPU
                nll = self.flow.anomaly_score(chunk).mean()
                total_nll += nll.item()
                n_steps   += 1
        else:
            for batch in self.val_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                with autocast("cuda"):
                    feats = self.mae.encode(imgs)
                B, N, D = feats.shape
                flat = feats.reshape(B * N, D).float()
                total_nll += self.flow.anomaly_score(flat).mean().item()
                n_steps   += 1

        return {"nll": total_nll / max(n_steps, 1)}

    def _save_checkpoint(self, epoch: int, fname: Optional[str] = None) -> None:
        state = {"epoch": epoch, "flow": self.flow.state_dict(),
                 "optimizer": self.optimizer.state_dict()}
        fname = fname or f"flow_epoch_{epoch:04d}.pth"
        torch.save(state, self.output_dir / fname)

    @torch.no_grad()
    def compute_anomaly_scores(
        self, loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Compute per-patch token anomaly scores for all data in loader.

        Returns:
            scores: (N_patches, N_tokens) float32 — NLL per token per patch
            image_level_scores: (N_patches,) float32 — max over tokens
            stems:  list[str] of patch stems, length N_patches
        """
        self.flow.eval()
        all_scores  = []
        all_stems   = []

        for batch in loader:
            imgs  = batch["image"].to(self.device, non_blocking=True)
            stems = batch["stem"]
            with autocast("cuda"):
                feats  = self.mae.encode(imgs)               # (B, N, D)
            scores = self.flow.score_patches(feats.float())  # (B, N)
            all_scores.append(scores.cpu())
            all_stems.extend(stems)

        scores_t = torch.cat(all_scores, dim=0)              # (N_patches, N_tokens)
        img_scores = scores_t.max(dim=1).values              # (N_patches,)
        return scores_t, img_scores, all_stems
