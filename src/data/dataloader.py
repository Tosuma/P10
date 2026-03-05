"""
DataLoader factory for WeedyRice patch dataset.

Handles both single-GPU and multi-GPU (DDP) setups transparently.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from src.data.patch_dataset import WeedyRicePatchDataset


def build_dataloaders(
    cfg,
    return_mask: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    When ``torch.distributed`` is initialised (DDP / torchrun), a
    ``DistributedSampler`` is used automatically so each process only sees its
    own shard of data.  Otherwise a standard ``RandomSampler`` is used.

    Expected ``cfg`` fields (OmegaConf DictConfig):
        cfg.data.patch_dir      Path to WeedyRice-patches/
        cfg.data.val_fraction   Fraction of images for validation (default 0.2)
        cfg.data.seed           RNG seed for reproducible split (default 42)
        cfg.data.ms_scale       Divisor for raw MS DNs (default 65535.0)
        cfg.data.batch_size     Batch size PER GPU (default 256)
        cfg.data.num_workers    DataLoader worker processes (default 8)
        cfg.data.pin_memory     Pin CPU memory for faster GPU transfer (default True)
        cfg.data.augment        Enable train-time augmentation (default True)

    Args:
        cfg:         Hydra/OmegaConf config object.
        return_mask: If True both datasets also load and return binary masks.

    Returns:
        train_loader, val_loader
    """
    is_distributed = dist.is_available() and dist.is_initialized()

    common_kwargs = dict(
        patch_dir=cfg.data.patch_dir,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.data.seed,
        ms_scale=cfg.data.ms_scale,
        return_mask=return_mask,
    )

    train_ds = WeedyRicePatchDataset(
        split="train",
        augment=cfg.data.get("augment", True),
        **common_kwargs,
    )
    val_ds = WeedyRicePatchDataset(
        split="val",
        augment=False,
        **common_kwargs,
    )

    if is_distributed:
        train_sampler = DistributedSampler(
            train_ds, shuffle=True, drop_last=True, seed=cfg.data.seed
        )
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = RandomSampler(train_ds)
        val_sampler = SequentialSampler(val_ds)

    loader_kwargs = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.get("pin_memory", True),
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader
