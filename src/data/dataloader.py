"""
DataLoader factory.

Returns train and validation DataLoaders configured for HPC usage:
  - pin_memory=True for faster GPU transfers
  - num_workers from config (typically 8–16 on HPC)
  - persistent_workers=True to avoid fork overhead on repeated epochs
  - DistributedSampler when DDP is active
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.data.patch_dataset import AgriculturalPatchDataset, make_datasets


def build_dataloaders(
    rgb_dir: Optional[str],
    ms_dir: str,
    batch_size: int = 64,
    num_workers: int = 8,
    val_fraction: float = 0.2,
    patch_size: int = 128,
    patches_per_image: int = 32,
    include_rgb: bool = True,
    include_indices: bool = True,
    cache_images: bool = False,
    seed: int = 42,
    ms_suffixes: list[str] | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders.

    Parameters
    ----------
    distributed : bool
        If True, wraps the training dataset in DistributedSampler for DDP.
    rank, world_size : int
        Process rank and total process count for DDP.

    Returns
    -------
    train_loader, val_loader
    """
    train_ds, val_ds = make_datasets(
        rgb_dir=rgb_dir,
        ms_dir=ms_dir,
        val_fraction=val_fraction,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        include_rgb=include_rgb,
        include_indices=include_indices,
        cache_images=cache_images,
        seed=seed,
        ms_suffixes=ms_suffixes,
    )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,     # Avoids incomplete batches breaking DDP sync
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    return train_loader, val_loader


def build_inference_loader(
    rgb_dir: Optional[str],
    ms_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    patch_size: int = 128,
    stride: Optional[int] = None,
    include_rgb: bool = True,
    include_indices: bool = True,
    ms_suffixes: list[str] | None = None,
) -> tuple[DataLoader, "AgriculturalPatchDataset"]:
    """
    Build a dense-grid inference DataLoader over all images.

    Returns both the DataLoader and the underlying Dataset (needed to map
    patch positions back to original image coordinates for heatmap assembly).
    """
    from src.data.patch_dataset import discover_image_records, AgriculturalPatchDataset

    records = discover_image_records(rgb_dir, ms_dir, ms_suffixes)
    dataset = AgriculturalPatchDataset(
        records=records,
        patch_size=patch_size,
        mode="infer",
        include_rgb=include_rgb,
        include_indices=include_indices,
        stride=stride if stride is not None else patch_size // 2,
        cache_images=True,   # Keep full images in RAM during inference
        ms_suffixes=ms_suffixes,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset
