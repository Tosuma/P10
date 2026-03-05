"""
WeedyRice patch dataset for MAE pretraining and anomaly detection.

Patches were pre-extracted by utils/patch_weedyrice.py into 128×128 tiles.
This module loads them into a 10-channel tensor suitable for the ViT encoder.

Channel layout (index → band):
    0  R_rgb     from RGB/    .jpg  (originally channel 2 in BGR cv2 image)
    1  G_rgb     from RGB/    .jpg  (channel 1)
    2  B_rgb     from RGB/    .jpg  (channel 0)
    3  G_ms      from Multispectral/ .npy  (band index 0)
    4  R_ms      from Multispectral/ .npy  (band index 1)
    5  RE_ms     from Multispectral/ .npy  (band index 2)
    6  NIR_ms    from Multispectral/ .npy  (band index 3)
    7  NDVI      from NDVI/   .npy
    8  NDRE      from NDRE/   .npy
    9  SAVI      from SAVI/   .npy

Design decisions documented here for thesis defence:
  - Image-level split (80/20): all patches from one drone image stay in the
    same split.  Splitting at patch level would cause train/val leakage because
    spatially adjacent patches share almost identical content.
  - No horizontal flip: rice-field row orientation is meaningful in aerial
    imagery; flipping left-right would create unrealistic orientations.
  - Vertical flip retained: the drone can fly in either direction along a row,
    so a top-bottom flip is a plausible real-world capture.
  - 90° rotations only: avoids interpolation artefacts from arbitrary angles
    and keeps the patch perfectly aligned on the grid.
  - Colour jitter on RGB only: spectral band ratios (NDVI etc.) must not be
    perturbed; jitter applied after channel splitting.
  - ms_scale=65535: DJI Mavic 3 MS raw DNs are 16-bit integers; dividing by
    65535 maps them to [0, 1]. Set ms_scale=1.0 if TIFs already hold
    calibrated reflectance values.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_STEM_RE = re.compile(r"_r\d{3}_c\d{3}$")


def _image_stem_from_patch(patch_stem: str) -> str:
    """Strip trailing `_r{row}_c{col}` from a patch filename stem."""
    return _PATCH_STEM_RE.sub("", patch_stem)


def _collect_patch_stems(rgb_dir: Path) -> list[str]:
    """Return sorted patch stems from the RGB sub-directory."""
    return sorted(p.stem for p in rgb_dir.glob("*.jpg"))


def _split_stems(
    all_patch_stems: list[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split patch stems into train / val at the IMAGE level.

    Returns:
        train_patch_stems, val_patch_stems
    """
    # Group patches by parent image stem
    img_stems: list[str] = sorted(
        {_image_stem_from_patch(s) for s in all_patch_stems}
    )

    rng = random.Random(seed)
    rng.shuffle(img_stems)

    n_val = max(1, int(len(img_stems) * val_fraction))
    val_img_stems = set(img_stems[:n_val])

    train_patches = [s for s in all_patch_stems if _image_stem_from_patch(s) not in val_img_stems]
    val_patches   = [s for s in all_patch_stems if _image_stem_from_patch(s) in val_img_stems]
    return train_patches, val_patches


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_rgb(arr: np.ndarray) -> np.ndarray:
    """uint8 BGR H×W×3 → float32 RGB H×W×3 in [0, 1]."""
    return arr[:, :, ::-1].astype(np.float32) / 255.0


def _norm_ms(arr: np.ndarray, ms_scale: float) -> np.ndarray:
    """float32 H×W×4 → [0, 1] (clamped)."""
    return np.clip(arr / ms_scale, 0.0, 1.0)


def _norm_ndvi(arr: np.ndarray) -> np.ndarray:
    """NDVI/NDRE float32 H×W ∈ [-1, 1] → [0, 1]."""
    return (np.clip(arr, -1.0, 1.0) + 1.0) / 2.0


def _norm_savi(arr: np.ndarray) -> np.ndarray:
    """SAVI float32 H×W ∈ [-1.5, 1.5] → [0, 1]."""
    return (np.clip(arr, -1.5, 1.5) + 1.5) / 3.0


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment(tensor: np.ndarray, seed: int | None = None) -> np.ndarray:
    """
    Apply spatial augmentation to a (C, H, W) float32 array.

    Augmentations applied identically to all channels:
      - Random 90° rotation (k ∈ {0, 1, 2, 3})
      - Random vertical flip with p=0.5

    Colour jitter (brightness/contrast) is applied to RGB channels only
    (channels 0-2) AFTER this function, see _apply_color_jitter.
    """
    rng = random.Random(seed)

    k = rng.randint(0, 3)
    if k:
        # np.rot90 operates on last two axes for (C, H, W)
        tensor = np.rot90(tensor, k=k, axes=(1, 2)).copy()

    if rng.random() < 0.5:
        tensor = np.flip(tensor, axis=1).copy()  # vertical flip

    return tensor


def _apply_color_jitter(
    tensor: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """
    Random brightness + contrast perturbation on RGB channels (0-2) only.

    Spectral bands (channels 3-9) are left untouched to preserve
    inter-channel ratios that are physically meaningful.
    """
    rng = np.random.RandomState(seed)
    out = tensor.copy()

    # brightness: add uniform random offset
    b = rng.uniform(-brightness, brightness)
    out[:3] = np.clip(out[:3] + b, 0.0, 1.0)

    # contrast: scale around 0.5
    c = rng.uniform(1.0 - contrast, 1.0 + contrast)
    out[:3] = np.clip((out[:3] - 0.5) * c + 0.5, 0.0, 1.0)

    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WeedyRicePatchDataset(Dataset):
    """
    PyTorch Dataset for 128×128 pre-extracted WeedyRice patches.

    Each item is a dict containing:
      "image"       (10, H, W) float32 — the multi-channel input
      "mask"        (1,  H, W) float32 — segmentation mask (0/1), optional
      "stem"        str — full patch stem, e.g. "DJI_..._r005_c012"
      "image_stem"  str — parent image stem, e.g. "DJI_..."

    Args:
        patch_dir:    Path to the WeedyRice-patches root directory.
        split:        "train", "val", or "all".
        val_fraction: Fraction of IMAGES (not patches) to hold out for val.
        seed:         Random seed for reproducible image-level split.
        ms_scale:     Divisor for raw multispectral DNs → [0, 1].
        augment:      If True, apply spatial + colour augmentations
                      (only effective for split="train").
        return_mask:  If True, load and return binary segmentation mask.
    """

    def __init__(
        self,
        patch_dir: str | Path,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        ms_scale: float = 65535.0,
        augment: bool = True,
        return_mask: bool = False,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "all"), f"Unknown split: {split!r}"

        self.patch_dir = Path(patch_dir)
        self.split = split
        self.ms_scale = ms_scale
        self.do_augment = augment and split == "train"
        self.return_mask = return_mask

        # Use packed .npz files (single file open per sample) when available.
        # Fall back to per-modality files if Packed/ does not exist.
        self._packed_dir: Path | None = None
        packed_dir = self.patch_dir / "Packed"
        if packed_dir.exists() and any(packed_dir.iterdir()):
            self._packed_dir = packed_dir

        rgb_dir = self.patch_dir / "RGB"
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB sub-directory not found: {rgb_dir}")

        all_stems = _collect_patch_stems(rgb_dir)
        if not all_stems:
            raise RuntimeError(f"No .jpg files found in {rgb_dir}")

        if split == "all":
            self.stems = all_stems
        else:
            train_stems, val_stems = _split_stems(all_stems, val_fraction, seed)
            self.stems = train_stems if split == "train" else val_stems

        # Store split info for external access (e.g. building DataLoader)
        if split != "all":
            train_s, val_s = _split_stems(all_stems, val_fraction, seed)
            train_img_stems = {_image_stem_from_patch(s) for s in train_s}
            val_img_stems   = {_image_stem_from_patch(s) for s in val_s}
            self.train_image_stems: set[str] = train_img_stems
            self.val_image_stems:   set[str] = val_img_stems

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        img_stem = _image_stem_from_patch(stem)

        # --- Load channels (1 file open if packed, 5 if not) -------------
        if self._packed_dir is not None:
            data = np.load(self._packed_dir / f"{stem}.npz")
            rgb_bgr = data["rgb"]
            ms      = _norm_ms(data["ms"], self.ms_scale)
            ndvi    = _norm_ndvi(data["ndvi"])
            ndre    = _norm_ndvi(data["ndre"])
            savi    = _norm_savi(data["savi"])
        else:
            rgb_path = self.patch_dir / "RGB" / f"{stem}.jpg"
            rgb_bgr  = cv2.imread(str(rgb_path))
            if rgb_bgr is None:
                raise FileNotFoundError(f"Cannot read: {rgb_path}")
            ms   = _norm_ms(np.load(self.patch_dir / "Multispectral" / f"{stem}.npy"), self.ms_scale)
            ndvi = _norm_ndvi(np.load(self.patch_dir / "NDVI" / f"{stem}.npy"))
            ndre = _norm_ndvi(np.load(self.patch_dir / "NDRE" / f"{stem}.npy"))
            savi = _norm_savi(np.load(self.patch_dir / "SAVI" / f"{stem}.npy"))

        rgb = _norm_rgb(rgb_bgr)  # (H, W, 3) float32, RGB order, [0,1]

        # --- Stack into (C, H, W) ----------------------------------------
        # Channel order: R, G, B, G_ms, R_ms, RE, NIR, NDVI, NDRE, SAVI
        image = np.concatenate(
            [
                rgb.transpose(2, 0, 1),              # (3, H, W)
                ms.transpose(2, 0, 1),               # (4, H, W)
                ndvi[np.newaxis],                    # (1, H, W)
                ndre[np.newaxis],                    # (1, H, W)
                savi[np.newaxis],                    # (1, H, W)
            ],
            axis=0,
        ).astype(np.float32)                         # (10, H, W)

        # --- Augmentation ------------------------------------------------
        if self.do_augment:
            aug_seed = int(torch.randint(0, 2**31, (1,)).item())
            image = _augment(image, seed=aug_seed)
            image = _apply_color_jitter(image, seed=aug_seed + 1)

        result: dict = {
            "image": torch.from_numpy(image),         # (10, H, W)
            "stem": stem,
            "image_stem": img_stem,
        }

        if self.return_mask:
            mask_path = self.patch_dir / "Masks" / f"{stem}.png"
            mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_arr is None:
                raise FileNotFoundError(f"Cannot read mask: {mask_path}")
            mask = (mask_arr > 127).astype(np.float32)[np.newaxis]  # (1, H, W)
            result["mask"] = torch.from_numpy(mask)

        return result

    # ------------------------------------------------------------------
    # Statistics computation (run once, cache results)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_stats(
        patch_dir: Path | str,
        split_stems: list[str],
        ms_scale: float = 65535.0,
        max_samples: int = 5000,
        seed: int = 0,
        save_path: Optional[Path | str] = None,
    ) -> dict[str, float]:
        """
        Compute per-channel mean and std from a random subsample of patches.

        Call this ONCE on the training split and save the result. Use the
        values for z-score normalisation if desired. The default normalisation
        in __getitem__ uses fixed ranges and does NOT require calling this.

        Returns:
            dict with keys "ch{i}_mean" and "ch{i}_std" for i in 0..9.
        """
        patch_dir = Path(patch_dir)
        rng = random.Random(seed)
        sample = rng.sample(split_stems, min(max_samples, len(split_stems)))

        channel_sums   = np.zeros(10, dtype=np.float64)
        channel_sq_sums = np.zeros(10, dtype=np.float64)
        n_pixels = 0

        for stem in sample:
            rgb_bgr = cv2.imread(str(patch_dir / "RGB" / f"{stem}.jpg"))
            if rgb_bgr is None:
                continue
            rgb = _norm_rgb(rgb_bgr).transpose(2, 0, 1)  # (3, H, W)
            ms  = _norm_ms(np.load(patch_dir / "Multispectral" / f"{stem}.npy"), ms_scale)
            ms  = ms.transpose(2, 0, 1)
            ndvi = _norm_ndvi(np.load(patch_dir / "NDVI" / f"{stem}.npy"))[np.newaxis]
            ndre = _norm_ndvi(np.load(patch_dir / "NDRE" / f"{stem}.npy"))[np.newaxis]
            savi = _norm_savi(np.load(patch_dir / "SAVI" / f"{stem}.npy"))[np.newaxis]
            img = np.concatenate([rgb, ms, ndvi, ndre, savi], axis=0)  # (10, H, W)

            h, w = img.shape[1], img.shape[2]
            channel_sums    += img.reshape(10, -1).sum(axis=1)
            channel_sq_sums += (img ** 2).reshape(10, -1).sum(axis=1)
            n_pixels += h * w

        means = channel_sums / n_pixels
        stds  = np.sqrt(channel_sq_sums / n_pixels - means ** 2)

        stats: dict[str, float] = {}
        for i in range(10):
            stats[f"ch{i}_mean"] = float(means[i])
            stats[f"ch{i}_std"]  = float(max(stds[i], 1e-6))

        if save_path is not None:
            with open(save_path, "w") as fh:
                json.dump(stats, fh, indent=2)

        return stats