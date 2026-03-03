"""
Entry point: Stage 2 FastFlow Training.

Usage:
    python train_flow.py
    python train_flow.py flow.mae_checkpoint=outputs/stage1_mae/mae_best.pth
"""

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.dataloader import build_dataloaders
from src.models.mae import MaskedAutoencoder
from src.training.flow_trainer import FlowTrainer

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="flow", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Device: {device}")

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
    )

    # ── MAE model (for encoder) ───────────────────────────────────────────────
    mae_model = MaskedAutoencoder(
        in_channels=cfg.data.in_channels,
        img_size=cfg.data.patch_size,
        patch_size=cfg.mae.patch_size,
        arch=cfg.mae.arch,
        masking_ratio=cfg.mae.masking_ratio,
        decoder_embed_dim=cfg.mae.decoder_embed_dim,
        decoder_depth=cfg.mae.decoder_depth,
        use_checkpoint=False,  # Not needed for frozen inference
    )

    # ── Flow trainer ──────────────────────────────────────────────────────────
    trainer = FlowTrainer(
        mae_model=mae_model,
        mae_checkpoint=cfg.flow.mae_checkpoint,
        output_dir=cfg.flow.output_dir,
        device=device,
        feature_dim=cfg.flow.feature_dim,
        num_coupling_blocks=cfg.flow.num_coupling_blocks,
        subnet_hidden_dim=cfg.flow.subnet_hidden_dim,
        epochs=cfg.flow.epochs,
        lr=cfg.flow.lr,
        weight_decay=cfg.flow.weight_decay,
        use_wandb=cfg.flow.use_wandb,
        project_name=cfg.flow.wandb_project,
    )

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
