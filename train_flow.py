"""
Stage 2: Normalizing Flow training entry point.

Requires a completed Stage 1 MAE checkpoint.
Single-GPU training (the flow model is small; DDP not needed).

Usage:
    python train_flow.py
    python train_flow.py mae_checkpoint=checkpoints/mae/best.pth
    python train_flow.py flow.cache_features=false logging.use_wandb=false
"""

from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.dataloader import build_dataloaders
from src.models.mae import build_mae
from src.models.flow_model import FlowModel
from src.training.flow_trainer import FlowTrainer


@hydra.main(config_path="configs", config_name="flow", version_base="1.3")
def main(cfg: DictConfig) -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        cfg.data.patch_dir = data_root

    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load pretrained MAE ───────────────────────────────────────────
    mae_ckpt = cfg.get("mae_checkpoint", "checkpoints/mae/best.pth")
    if not os.path.exists(mae_ckpt):
        raise FileNotFoundError(
            f"MAE checkpoint not found: {mae_ckpt}\n"
            "Run train_mae.py first (Stage 1)."
        )

    # Reconstruct the MAE architecture (must match Stage 1 config)
    mae = build_mae(
        arch=cfg.mae.get("arch", "vit_small_patch16"),
        in_chans=cfg.data.in_chans,
        img_size=cfg.data.patch_size,
        use_checkpoint=False,          # no checkpointing needed for frozen encoder
    ).to(device)

    ckpt = torch.load(mae_ckpt, map_location=device)
    state = ckpt.get("model", ckpt)    # handle both raw and wrapped state dicts
    mae.load_state_dict(state, strict=False)
    mae.eval()
    print(f"MAE encoder loaded from {mae_ckpt}")

    # ── Flow model ────────────────────────────────────────────────────
    flow = FlowModel(
        feature_dim=cfg.flow.feature_dim,
        n_coupling_blocks=cfg.flow.n_coupling_blocks,
        clamp=cfg.flow.clamp,
    ).to(device)

    n_params = sum(p.numel() for p in flow.parameters()) / 1e6
    print(f"Flow model parameters: {n_params:.2f}M")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = FlowTrainer(cfg, flow, mae, train_loader, val_loader, device)
    trainer.train()


if __name__ == "__main__":
    main()
