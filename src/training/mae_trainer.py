"""
Stage 1: MAE pre-training loop.

Supports multi-GPU DDP (torchrun), mixed-precision (torch.cuda.amp),
cosine LR schedule with linear warmup, and WandB logging.

Learning rate scaling: lr = base_lr × batch_size / 256
This follows the linear scaling rule (Goyal et al. 2017) standard in
self-supervised learning: the effective learning rate should scale with
the total batch size across all GPUs.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.models.mae import MAEModel


class MAETrainer:
    """
    Trainer for Stage 1 MAE pretraining.

    Args:
        cfg:          OmegaConf DictConfig.  Expected fields:
                        cfg.training.{epochs, base_lr, weight_decay,
                                      warmup_epochs, min_lr, clip_grad}
                        cfg.data.batch_size
                        cfg.logging.{use_wandb, project, run_name, output_dir,
                                     save_every, log_every}
        model:        MAEModel (possibly DDP-wrapped).
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        device:       torch.device for this process.
        rank:         Global rank in DDP (0 = primary; -1 = no DDP).
    """

    def __init__(
        self,
        cfg,
        model: MAEModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        rank: int = -1,
    ) -> None:
        self.cfg         = cfg
        self.model       = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.rank         = rank
        self.is_primary   = rank in (-1, 0)

        self.output_dir = Path(cfg.logging.output_dir)
        if self.is_primary:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Effective batch size for LR scaling
        eff_batch = cfg.data.batch_size * max(1, dist.get_world_size() if dist.is_initialized() else 1)
        self.peak_lr = cfg.training.base_lr * eff_batch / 256.0

        self.optimizer = self._build_optimizer()
        self.scaler    = GradScaler()

        # WandB
        self.use_wandb = cfg.logging.get("use_wandb", False) and self.is_primary
        if self.use_wandb:
            import wandb
            wandb.init(
                project=cfg.logging.project,
                name=cfg.logging.get("run_name", None),
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, start_epoch: int = 0) -> None:
        """Run the full training loop."""
        epochs = self.cfg.training.epochs
        best_val_loss = float("inf")
        save_every    = self.cfg.logging.get("save_every", 20)

        for epoch in range(start_epoch, epochs):
            # DDP: re-seed sampler each epoch so shuffling differs per rank
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)
            elapsed = time.time() - t0

            if self.is_primary:
                log = {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/lr":   train_metrics["lr"],
                    "val/loss":   val_metrics["loss"],
                    "epoch_time_s": elapsed,
                }
                print(
                    f"[{epoch:04d}/{epochs}] "
                    f"train={train_metrics['loss']:.4f}  "
                    f"val={val_metrics['loss']:.4f}  "
                    f"lr={train_metrics['lr']:.2e}  "
                    f"time={elapsed:.1f}s"
                )
                if self.use_wandb:
                    import wandb
                    wandb.log(log, step=epoch)

                is_best = val_metrics["loss"] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics["loss"]
                    self._save_checkpoint(epoch, fname="best.pth")

                if (epoch + 1) % save_every == 0:
                    self._save_checkpoint(epoch)

        if self.is_primary:
            self._save_checkpoint(epochs - 1, fname="final.pth")
            if self.use_wandb:
                import wandb
                wandb.finish()

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint state.  Returns epoch to resume from."""
        ckpt = torch.load(path, map_location=self.device)
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {path} (epoch {epoch})")
        return epoch

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_steps    = 0
        current_lr = self.peak_lr
        log_every  = self.cfg.logging.get("log_every", 50)
        steps_per_epoch = len(self.train_loader)

        for step, batch in enumerate(self.train_loader):
            imgs = batch["image"].to(self.device, non_blocking=True)   # (B, 10, H, W)

            # Update LR (cosine schedule with warmup, applied per step)
            current_lr = self._cosine_lr(epoch, step, steps_per_epoch)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                loss, _, _ = self.model(imgs)

            self.scaler.scale(loss).backward()

            if self.cfg.training.get("clip_grad", 1.0) > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.clip_grad)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_steps    += 1

            if self.is_primary and step % log_every == 0:
                print(f"  step {step}/{steps_per_epoch}  loss={loss.item():.4f}  lr={current_lr:.2e}")

        return {"loss": total_loss / max(n_steps, 1), "lr": current_lr}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_steps    = 0

        with autocast():
            for batch in self.val_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                loss, _, _ = self.model(imgs)
                total_loss += loss.item()
                n_steps    += 1

        # All-reduce loss across DDP ranks
        if dist.is_initialized():
            t = torch.tensor(total_loss / max(n_steps, 1), device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            return {"loss": t.item()}

        return {"loss": total_loss / max(n_steps, 1)}

    def _save_checkpoint(self, epoch: int, fname: Optional[str] = None) -> None:
        if not self.is_primary:
            return
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "epoch":     epoch,
            "model":     raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler":    self.scaler.state_dict(),
        }
        fname = fname or f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(state, self.output_dir / fname)

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        AdamW with zero weight decay for bias and LayerNorm parameters.
        This is standard practice in ViT training (Loshchilov & Hutter 2019).
        """
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        groups = [
            {"params": decay,    "weight_decay": self.cfg.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.peak_lr, betas=(0.9, 0.95))

    def _cosine_lr(self, epoch: int, step: int, steps_per_epoch: int) -> float:
        """
        Cosine annealing with linear warmup, updated per step.
        Updates the optimizer learning rate in-place and returns the new lr.
        """
        warmup_steps = self.cfg.training.warmup_epochs * steps_per_epoch
        total_steps  = self.cfg.training.epochs * steps_per_epoch
        global_step  = epoch * steps_per_epoch + step
        min_lr       = self.cfg.training.min_lr

        if global_step < warmup_steps:
            lr = min_lr + (self.peak_lr - min_lr) * global_step / max(warmup_steps, 1)
        else:
            progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = min_lr + 0.5 * (self.peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr
