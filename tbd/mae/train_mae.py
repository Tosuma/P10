"""
Entry point: Stage 1 MAE Pretraining.

Usage (single GPU):
    python train_mae.py

Usage (multi-GPU via torchrun):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train_mae.py

Usage (override config):
    python train_mae.py mae.epochs=100 data.batch_size=32
"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from src.data.dataloader import build_dataloaders
from src.models.mae import MaskedAutoencoder
from src.training.mae_trainer import MAETrainer

log = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise DDP from torchrun environment variables."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


@hydra.main(config_path="configs", config_name="mae", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rank, world_size, device = setup_distributed()

    if rank == 0:
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
        log.info(f"Device: {device} | World size: {world_size}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        rgb_dir=cfg.data.rgb_dir if cfg.data.include_rgb else None,
        ms_dir=cfg.data.ms_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_fraction=cfg.data.val_fraction,
        patch_size=cfg.data.patch_size,
        patches_per_image=cfg.data.patches_per_image,
        include_rgb=cfg.data.include_rgb,
        include_indices=cfg.data.include_indices,
        cache_images=cfg.data.cache_images,
        seed=cfg.data.seed,
        ms_suffixes=list(cfg.data.ms_suffixes),
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MaskedAutoencoder(
        in_channels=cfg.data.in_channels,
        img_size=cfg.data.patch_size,
        patch_size=cfg.mae.patch_size,
        arch=cfg.mae.arch,
        masking_ratio=cfg.mae.masking_ratio,
        decoder_embed_dim=cfg.mae.decoder_embed_dim,
        decoder_depth=cfg.mae.decoder_depth,
        decoder_num_heads=cfg.mae.decoder_num_heads,
        use_checkpoint=cfg.mae.use_checkpoint,
        sa_reduction_ratio=cfg.mae.sa_reduction_ratio,
    )

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Model parameters: {n_params / 1e6:.1f}M")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = MAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=cfg.mae.output_dir,
        device=device,
        rank=rank,
        world_size=world_size,
        epochs=cfg.mae.epochs,
        base_lr=cfg.mae.base_lr,
        weight_decay=cfg.mae.weight_decay,
        warmup_epochs=cfg.mae.warmup_epochs,
        grad_clip=cfg.mae.grad_clip,
        use_wandb=cfg.mae.use_wandb and rank == 0,
        project_name=cfg.mae.wandb_project,
        save_every=cfg.mae.save_every,
        early_stop_patience=cfg.mae.early_stop_patience,
    )

    trainer.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
