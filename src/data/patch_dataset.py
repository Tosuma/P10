"""
Patch-based dataset for drone imagery.

Why patches?
------------
Drone images are typically 4000×3000+ pixels.  Loading full images into a ViT
would require impractical patch counts.  We extract fixed-size patches (default
128×128) either:
  (a) Randomly during training — effectively infinite augmentation.
  (b) In a dense sliding-window grid at inference — for full-image heatmaps.

Augmentation policy for aerial agricultural data
-------------------------------------------------
We deliberately **exclude** left/right horizontal flips because aerial imagery
has a well-defined nadir orientation and rows in agricultural fields have
meaningful orientation.  We do allow:
  - 90° rotations (fields can be oriented any way relative to flight path)
  - Gaussian blur on RGB (models sensor defocus; do NOT apply to spectral bands)
  - Random brightness/contrast on RGB only (illumination varies; spectral
    reflectance should not be artificially altered as it encodes plant physiology)
  - Random crop within the patch extraction window

Spectral bands and vegetation indices are augmented geometrically only.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from src.data.band_loader import load_image_pair, normalize_channels
from src.data.vegetation_indices import compute_vegetation_indices


class DroneImageRecord:
    """
    Lightweight container for a single drone image file pair.

    Attributes
    ----------
    stem : str
        Shared filename stem (e.g. 'DJI_0001').
    rgb_path : Path | None
    ms_dir : Path
    """
    __slots__ = ("stem", "rgb_path", "ms_dir")

    def __init__(self, stem: str, rgb_path: Optional[Path], ms_dir: Path):
        self.stem = stem
        self.rgb_path = rgb_path
        self.ms_dir = ms_dir


def discover_image_records(
    rgb_dir: Optional[str | Path],
    ms_dir: str | Path,
    ms_suffixes: list[str] | None = None,
) -> list[DroneImageRecord]:
    """
    Scan directories and pair RGB files with their MS counterparts by stem.

    Parameters
    ----------
    rgb_dir : path to RGB images (optional; pass None for MS-only)
    ms_dir  : path to per-band multispectral TIFs
    ms_suffixes : band suffixes used for grouping (default: ['_G','_R','_RE','_NIR'])

    Returns
    -------
    List of DroneImageRecord, sorted by stem for reproducibility.
    """
    if ms_suffixes is None:
        ms_suffixes = ["_G", "_R", "_RE", "_NIR"]

    ms_dir = Path(ms_dir)

    # Collect unique stems from MS directory
    ms_stems: set[str] = set()
    for f in ms_dir.iterdir():
        if f.suffix.lower() not in (".tif", ".tiff"):
            continue
        name = f.stem
        for suf in ms_suffixes:
            if name.endswith(suf):
                ms_stems.add(name[: -len(suf)])
                break

    records: list[DroneImageRecord] = []
    for stem in sorted(ms_stems):
        rgb_path: Optional[Path] = None
        if rgb_dir is not None:
            rgb_dir_ = Path(rgb_dir)
            for ext in (".JPG", ".jpg", ".jpeg", ".png", ".TIF", ".tif"):
                candidate = rgb_dir_ / f"{stem}{ext}"
                if candidate.exists():
                    rgb_path = candidate
                    break
        records.append(DroneImageRecord(stem, rgb_path, ms_dir))

    return records


class AgriculturalPatchDataset(Dataset):
    """
    Unsupervised patch dataset for drone multispectral + RGB imagery.

    Loads full images lazily (cached per-image after first access), extracts
    random patches during training and dense grid patches during inference.

    Parameters
    ----------
    records : list of DroneImageRecord
    patch_size : int
        Side length of square patches (default 128).
    patches_per_image : int
        Number of random patches sampled per image per epoch (training mode).
    mode : 'train' | 'val' | 'infer'
        Controls augmentation and whether patches are random or grid-based.
    ms_suffixes : list of band suffixes
    include_rgb : bool
    include_indices : bool
        Append NDVI, NDRE, SAVI, EVI channels.
    stride : int
        Stride for dense grid extraction in inference mode (default = patch_size//2).
    transform : optional additional callable applied to the final tensor
    cache_images : bool
        If True, caches loaded + normalized images in memory (RAM).
        Recommended on HPC where repeated disk I/O is a bottleneck.
    """

    def __init__(
        self,
        records: list[DroneImageRecord],
        patch_size: int = 128,
        patches_per_image: int = 16,
        mode: str = "train",
        ms_suffixes: list[str] | None = None,
        include_rgb: bool = True,
        include_indices: bool = True,
        stride: Optional[int] = None,
        transform: Optional[Callable] = None,
        cache_images: bool = False,
    ):
        assert mode in ("train", "val", "infer"), f"Unknown mode: {mode}"
        self.records = records
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.mode = mode
        self.ms_suffixes = ms_suffixes or ["_G", "_R", "_RE", "_NIR"]
        self.include_rgb = include_rgb
        self.include_indices = include_indices
        self.stride = stride if stride is not None else patch_size // 2
        self.transform = transform
        self.cache_images = cache_images

        # Cache: stem → (H, W, C) float32 array
        self._image_cache: dict[str, np.ndarray] = {}

        # Shape cache: stem → (H, W) — persisted to disk to avoid repeated
        # rasterio opens on slow network filesystems (Ceph, NFS).
        self._shape_cache_path = Path(ms_dir) / ".shape_cache.json"
        self._shape_cache: dict[str, tuple[int, int]] = self._load_shape_cache()

        # Pre-compute patch index list for val/infer (deterministic)
        if mode in ("val", "infer"):
            self._patch_index = self._build_grid_index()
        else:
            # For train: index is (record_index, patch_within_image)
            self._train_length = len(records) * patches_per_image

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_shape_cache(self) -> dict[str, tuple[int, int]]:
        """Load persisted (stem → (H, W)) mapping from disk, or return empty dict."""
        if self._shape_cache_path.exists():
            try:
                raw = json.loads(self._shape_cache_path.read_text())
                return {k: tuple(v) for k, v in raw.items()}
            except Exception:
                pass
        return {}

    def _save_shape_cache(self) -> None:
        """Persist the shape cache to disk for future runs."""
        try:
            self._shape_cache_path.write_text(
                json.dumps({k: list(v) for k, v in self._shape_cache.items()}, indent=2)
            )
        except Exception:
            pass  # Non-fatal: slow network write or read-only filesystem

    def _load_full_image(self, record: DroneImageRecord) -> np.ndarray:
        """Load, normalize, and optionally append vegetation indices."""
        if record.stem in self._image_cache:
            return self._image_cache[record.stem]

        image, _ = load_image_pair(
            rgb_path=record.rgb_path,
            ms_dir=record.ms_dir,
            stem=record.stem,
            ms_suffixes=self.ms_suffixes,
            include_rgb=self.include_rgb,
            align_to_rgb=self.include_rgb,
        )
        image = normalize_channels(image)

        if self.include_indices:
            rgb_offset = 3 if self.include_rgb else 0
            has_blue = self.include_rgb
            image = compute_vegetation_indices(
                image,
                rgb_offset=rgb_offset,
                has_blue=has_blue,
            )

        if self.cache_images:
            self._image_cache[record.stem] = image

        return image

    def _get_image_shape(self, record: DroneImageRecord) -> tuple[int, int]:
        """
        Return (H, W) by reading rasterio file metadata only — no pixel loading.

        Checks an in-memory shape cache first to avoid repeated rasterio opens
        on slow network filesystems (Ceph/NFS).  The cache is persisted to disk
        via _save_shape_cache() so subsequent runs skip rasterio entirely.

        When include_rgb=True, the final image is aligned to the RGB grid, so
        RGB dimensions are the authoritative source.  Otherwise we fall back to
        the first MS band file.
        """
        if record.stem in self._shape_cache:
            return self._shape_cache[record.stem]

        if self.include_rgb and record.rgb_path is not None:
            with rasterio.open(str(record.rgb_path)) as src:
                shape = src.height, src.width
        else:
            # MS-only: use first available band file
            shape = None
            for suffix in self.ms_suffixes:
                for ext in (".TIF", ".tif", ".tiff", ".TIFF"):
                    candidate = record.ms_dir / f"{record.stem}{suffix}{ext}"
                    if candidate.exists():
                        with rasterio.open(str(candidate)) as src:
                            shape = src.height, src.width
                        break
                if shape is not None:
                    break
            if shape is None:
                raise FileNotFoundError(
                    f"Cannot determine image shape for stem='{record.stem}'"
                )

        self._shape_cache[record.stem] = shape
        return shape

    def _build_grid_index(self) -> list[tuple[int, int, int]]:
        """
        Build a list of (record_idx, top_left_y, top_left_x) for dense tiling.

        Uses rasterio metadata to get image dimensions without loading pixels —
        previously this called _load_full_image() for every val image at startup,
        causing multi-minute hangs when reading large TIFs from Ceph.
        """
        index: list[tuple[int, int, int]] = []
        p = self.patch_size
        s = self.stride
        cache_was_complete = all(r.stem in self._shape_cache for r in self.records)
        for r_idx, record in enumerate(self.records):
            H, W = self._get_image_shape(record)
            for y in range(0, H - p + 1, s):
                for x in range(0, W - p + 1, s):
                    index.append((r_idx, y, x))
        if not cache_was_complete:
            self._save_shape_cache()
        return index

    def _extract_patch(
        self, image: np.ndarray, y: int, x: int
    ) -> np.ndarray:
        """Extract a (patch_size, patch_size, C) crop."""
        p = self.patch_size
        return image[y : y + p, x : x + p].copy()

    def _augment_train(self, patch: np.ndarray) -> np.ndarray:
        """
        Geometric + photometric augmentation for training patches.

        Photometric augmentation is applied only to the RGB channels (first 3)
        because:
        - Spectral reflectance values are physically meaningful; artificially
          shifting them would corrupt the plant-physiology signal.
        - Vegetation indices would also be invalidated by spectral jitter.
        - RGB brightness/contrast jitter models real illumination variation
          between flight passes.
        """
        # ── 90° rotation (k ∈ {0,1,2,3}) ────────────────────────────────────
        k = random.randint(0, 3)
        patch = np.rot90(patch, k=k, axes=(0, 1)).copy()

        # ── Random vertical flip (nadir view: up/down is ambiguous) ──────────
        # Horizontal flip is excluded: field row orientation is meaningful.
        if random.random() < 0.5:
            patch = patch[::-1, :, :].copy()

        # ── Gaussian blur on RGB only (p=0.2) ────────────────────────────────
        if patch.shape[-1] >= 3 and random.random() < 0.2:
            sigma = random.uniform(0.3, 1.2)
            for c in range(3):
                patch[..., c] = cv2.GaussianBlur(patch[..., c], (3, 3), sigma)

        # ── Brightness/contrast jitter on RGB only (p=0.5) ───────────────────
        if patch.shape[-1] >= 3 and random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)   # contrast
            beta  = random.uniform(-0.1, 0.1)  # brightness
            patch[..., :3] = np.clip(alpha * patch[..., :3] + beta, 0.0, 1.0)

        return patch

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self.mode == "train":
            return self._train_length
        return len(self._patch_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.mode == "train":
            record_idx = idx % len(self.records)
            record = self.records[record_idx]
            image = self._load_full_image(record)
            H, W = image.shape[:2]
            p = self.patch_size
            # Random top-left corner
            y = random.randint(0, H - p)
            x = random.randint(0, W - p)
            patch = self._extract_patch(image, y, x)
            patch = self._augment_train(patch)
        else:
            record_idx, y, x = self._patch_index[idx]
            record = self.records[record_idx]
            image = self._load_full_image(record)
            patch = self._extract_patch(image, y, x)

        # (H, W, C) → (C, H, W) float32 tensor
        tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float()

        if self.transform is not None:
            tensor = self.transform(tensor)

        return {
            "image": tensor,
            "record_idx": record_idx,
            "patch_y": y,
            "patch_x": x,
        }


# ── Convenience factory ───────────────────────────────────────────────────────

def make_datasets(
    rgb_dir: Optional[str | Path],
    ms_dir: str | Path,
    val_fraction: float = 0.2,
    patch_size: int = 128,
    patches_per_image: int = 32,
    include_rgb: bool = True,
    include_indices: bool = True,
    cache_images: bool = False,
    seed: int = 42,
    ms_suffixes: list[str] | None = None,
) -> tuple["AgriculturalPatchDataset", "AgriculturalPatchDataset"]:
    """
    Discover all image pairs, split into train/val at the *image* level
    (not patch level) to prevent leakage, and return Dataset objects.

    Split is performed at the image level so that train and val patches come
    from entirely different drone images — this prevents the model from
    memorising local textures that appear in both splits.
    """
    records = discover_image_records(rgb_dir, ms_dir, ms_suffixes)
    assert len(records) > 0, f"No records found in ms_dir={ms_dir}"

    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_records   = shuffled[:n_val]
    train_records = shuffled[n_val:]

    train_ds = AgriculturalPatchDataset(
        records=train_records,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        mode="train",
        include_rgb=include_rgb,
        include_indices=include_indices,
        cache_images=cache_images,
        ms_suffixes=ms_suffixes,
    )
    val_ds = AgriculturalPatchDataset(
        records=val_records,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        mode="val",
        include_rgb=include_rgb,
        include_indices=include_indices,
        cache_images=cache_images,
        ms_suffixes=ms_suffixes,
        stride=patch_size,  # Non-overlapping grid for val efficiency
    )
    return train_ds, val_ds
