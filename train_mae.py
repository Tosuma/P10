"""
Stage 1: MAE Pre-training entry point.

Multi-GPU launch (recommended):
    torchrun --standalone --nproc_per_node=4 train_mae.py

Single-GPU / debug:
    python train_mae.py data.batch_size=64 mae.use_checkpoint=false

Config overrides follow Hydra syntax:
    torchrun ... train_mae.py mae.mask_ratio=0.8 logging.use_wandb=false

The DATA_ROOT environment variable overrides data.patch_dir on HPC nodes:
    DATA_ROOT=/ceph/.../WeedyRice-patches sbatch scripts/slurm/train_mae.sh
"""

from __future__ import annotations

import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from src.data.dataloader import build_dataloaders
from src.models.mae import build_mae
from src.training.mae_trainer import MAETrainer


def _setup_ddp() -> tuple[int, int, torch.device]:
    """
    Initialise NCCL process group when launched with torchrun.

    Returns:
        local_rank, global_rank, device
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        # Single-GPU / CPU run
        return -1, -1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), torch.device(f"cuda:{local_rank}")


def _teardown_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(config_path="configs", config_name="mae", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Allow DATA_ROOT env var to override patch_dir (HPC convention)
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        cfg.data.patch_dir = data_root

    local_rank, rank, device = _setup_ddp()
    is_primary = rank in (-1, 0)

    if is_primary:
        print(OmegaConf.to_yaml(cfg))

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────
    model = build_mae(
        arch=cfg.mae.arch,
        in_chans=cfg.data.in_chans,
        img_size=cfg.data.patch_size,
        use_checkpoint=cfg.mae.use_checkpoint,
        mask_ratio=cfg.mae.mask_ratio,
        decoder_embed_dim=cfg.mae.decoder_embed_dim,
        decoder_depth=cfg.mae.decoder_depth,
        decoder_num_heads=cfg.mae.decoder_num_heads,
        norm_pix_loss=cfg.mae.get("norm_pix_loss", False),
    ).to(device)

    model = torch.compile(model, mode="reduce-overhead")

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )

    if is_primary:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model parameters: {n_params:.1f}M")

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = MAETrainer(cfg, model, train_loader, val_loader, device, rank=rank)

    # Resume from checkpoint if specified
    start_epoch = 0
    resume_path = cfg.get("resume", None)
    if resume_path:
        start_epoch = trainer.load_checkpoint(resume_path)

    trainer.train(start_epoch=start_epoch)
    _teardown_ddp()


if __name__ == "__main__":
    main()
