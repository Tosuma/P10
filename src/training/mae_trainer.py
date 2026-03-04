"""
Stage 1 MAE Pretraining Trainer.

Features
--------
- Mixed precision (torch.cuda.amp) for 2× memory efficiency
- DistributedDataParallel (DDP) — call via torchrun / SLURM srun
- WandB + TensorBoard dual logging
- Cosine LR schedule with linear warmup (standard for MAE)
- Checkpoint save/resume (epoch-level granularity)
- EarlyStopping on validation loss with configurable patience
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.mae import MaskedAutoencoder

logger = logging.getLogger(__name__)


def _cosine_warmup_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> None:
    """
    Manual cosine LR schedule with linear warmup.

    We implement this manually rather than using a scheduler object because
    it must compose correctly with DDP gradient accumulation and gradient
    clipping — both of which interact poorly with PyTorch scheduler state.
    """
    if epoch < warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        lr = float(lr)
    for pg in optimizer.param_groups:
        pg["lr"] = lr


class MAETrainer:
    """
    Encapsulates the MAE pretraining loop.

    Parameters
    ----------
    model : MaskedAutoencoder
    train_loader, val_loader : DataLoaders
    output_dir : str
        Directory for checkpoints and TensorBoard logs.
    device : torch.device
    rank : int
        Local GPU rank (0 for single-GPU).
    world_size : int
    epochs : int
    base_lr : float
    weight_decay : float
    warmup_epochs : int
    grad_clip : float
        Max gradient norm (1.0 is standard for ViT pretraining).
    use_wandb : bool
    project_name : str
        WandB project name.
    save_every : int
        Save checkpoint every N epochs.
    early_stop_patience : int
        Stop if val loss does not improve for this many epochs (0 = disabled).
    """

    def __init__(
        self,
        model: MaskedAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        epochs: int = 200,
        base_lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 40,
        grad_clip: float = 1.0,
        use_wandb: bool = True,
        project_name: str = "plant-stress-mae",
        save_every: int = 10,
        early_stop_patience: int = 20,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.epochs = epochs
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.save_every = save_every
        self.early_stop_patience = early_stop_patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ── Model → DDP ───────────────────────────────────────────────────
        self.model = model.to(device)
        if world_size > 1:
            self.model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        # ── Optimiser ─────────────────────────────────────────────────────
        # Weight decay is NOT applied to biases, LayerNorm, or positional
        # embeddings — this is critical for ViT training stability.
        decay_params, no_decay_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "bias" in name or "norm" in name or "pos_embed" in name or "cls_token" in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=base_lr,
            betas=(0.9, 0.95),  # MAE paper default (slower β2 for larger batches)
        )

        # ── Mixed precision ────────────────────────────────────────────────
        self.scaler = GradScaler("cuda")

        # ── Logging ───────────────────────────────────────────────────────
        self.writer: Optional[SummaryWriter] = None
        if self.is_main:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))

        self.wandb_run = None
        if use_wandb and self.is_main:
            try:
                import wandb
                self.wandb_run = wandb.init(project=project_name, config={
                    "epochs": epochs, "base_lr": base_lr,
                    "weight_decay": weight_decay, "warmup_epochs": warmup_epochs,
                })
            except Exception as e:
                logger.warning(f"WandB init failed: {e}. Continuing without WandB.")

    # ── Training loop ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Update LR at epoch start
        raw_model = self.model.module if self.world_size > 1 else self.model
        _cosine_warmup_lr(self.optimizer, epoch, self.warmup_epochs, self.epochs, self.base_lr)

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                # Call self.model (the DDP wrapper), NOT self.model.module.
                # Calling .module directly bypasses DDP's backward hooks so
                # gradients are never all-reduced across GPUs.
                loss, _, _ = self.model(images)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        raw_model = self.model.module if self.world_size > 1 else self.model

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            with autocast("cuda"):
                loss, _, _ = raw_model(images)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def _log(self, metrics: dict, epoch: int) -> None:
        if not self.is_main:
            return
        for k, v in metrics.items():
            if self.writer:
                self.writer.add_scalar(k, v, epoch)
        if self.wandb_run:
            try:
                import wandb
                wandb.log({**metrics, "epoch": epoch})
            except Exception:
                pass
        msg = (
            f"Epoch {epoch:04d} | "
            + " | ".join(f"{k}={v:.5f}" for k, v in metrics.items())
        )
        logger.info(msg)
        print(msg, flush=True)  # ensure visibility in SLURM output regardless of Hydra log routing

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        if not self.is_main:
            return
        raw_model = self.model.module if self.world_size > 1 else self.model
        state = {
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "val_loss": val_loss,
        }
        ckpt_path = self.output_dir / f"mae_epoch_{epoch:04d}.pth"
        torch.save(state, ckpt_path)
        if is_best:
            torch.save(state, self.output_dir / "mae_best.pth")
        logger.info(f"Saved checkpoint: {ckpt_path}")

    def resume(self, checkpoint_path: str) -> int:
        """Load checkpoint and return the epoch to resume from."""
        state = torch.load(checkpoint_path, map_location=self.device)
        raw_model = self.model.module if self.world_size > 1 else self.model
        raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        logger.info(f"Resumed from epoch {state['epoch']} (val_loss={state['val_loss']:.5f})")
        return state["epoch"] + 1

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run full training loop."""
        start_epoch = 0
        if resume_from:
            start_epoch = self.resume(resume_from)

        best_val_loss = float("inf")
        no_improve_count = 0

        for epoch in range(start_epoch, self.epochs):
            t0 = time.time()

            # DDP: set epoch for reproducible shuffling
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            train_loss = self._train_epoch(epoch)
            val_loss   = self._val_epoch()

            # Reduce val loss across ranks for correct best tracking
            if self.world_size > 1:
                val_tensor = torch.tensor(val_loss, device=self.device)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
                val_loss = val_tensor.item()

            elapsed = time.time() - t0
            lr_now = self.optimizer.param_groups[0]["lr"]

            self._log({
                "loss/train": train_loss,
                "loss/val":   val_loss,
                "lr":         lr_now,
                "time_s":     elapsed,
            }, epoch)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if epoch % self.save_every == 0 or is_best:
                self._save_checkpoint(epoch, val_loss, is_best)

            # Early stopping
            if (
                self.early_stop_patience > 0
                and no_improve_count >= self.early_stop_patience
            ):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.early_stop_patience} epochs)"
                )
                break

        if self.writer:
            self.writer.close()
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
