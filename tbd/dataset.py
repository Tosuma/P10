"""
Dataset class for the Weedy Rice segmentation experiment.

Supports three input variants:
  1. Real multispectral (4 bands: G, R, RE, NIR as separate TIFs)
  2. Synthetic multispectral (from RGB reconstruction, same 4-band layout)
  3. RGB only

Directory structure expected (WeedyRice-RGBMS-DB):
    data/
    ├── RGB/              # RGB images (*.JPG)
    ├── Multispectral/    # Per-band TIFs (*_G.TIF, *_R.TIF, *_RE.TIF, *_NIR.TIF)
    ├── Synthetic/        # Per-band JPGs (validation_result_<id>_<band>_.JPG, bands 1-4)
    └── Masks/            # Binary segmentation masks (*.png, 0=background, 255=weed)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

BAND_SUFFIXES = ["_G", "_R", "_RE", "_NIR"]
SYNTHETIC_BANDS = [1, 2, 3, 4]  # Band indices in synthetic filenames


class WeedyRiceDataset(Dataset):
    """
    Dataset for weedy rice segmentation.

    Stores image stems (filename without extension/band suffix) so that
    RGB, multispectral, and mask lookups all share the same key.

    Args:
        image_dir: Path to the input images directory.
        mask_dir: Path to the segmentation masks directory.
        input_type: One of 'rgb', 'multispectral', 'synthetic'.
        transform: Albumentations transform pipeline (optional).
        num_classes: Number of segmentation classes.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        input_type: str = "rgb",
        transform: A.Compose = None,
        num_classes: int = 2,
    ):
        assert input_type in ("rgb", "multispectral", "synthetic"), (
            f"input_type must be 'rgb', 'multispectral', or 'synthetic', got '{input_type}'"
        )

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_type = input_type
        self.transform = transform
        self.num_classes = num_classes

        # Collect unique image stems
        # For synthetic: files use different naming, so we map by sorted index
        self._synthetic_map = {}  # mask_stem -> synthetic_id
        if input_type == "multispectral":
            self.image_stems = self._collect_ms_stems(image_dir)
        elif input_type == "synthetic":
            self.image_stems, self._synthetic_map = self._collect_synthetic_stems(
                image_dir, mask_dir
            )
        else:
            self.image_stems = self._collect_rgb_stems(image_dir)

        # Verify masks exist for each stem
        for stem in self.image_stems:
            mask_path = os.path.join(mask_dir, stem + ".png")
            assert os.path.exists(mask_path), (
                f"Mask not found for stem '{stem}': expected {stem}.png in {mask_dir}"
            )

    @staticmethod
    def _collect_rgb_stems(image_dir: str) -> list:
        """Collect stems from RGB directory (strip extension)."""
        stems = []
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                stems.append(os.path.splitext(f)[0])
        return stems

    @staticmethod
    def _collect_ms_stems(image_dir: str) -> list:
        """Collect unique stems from multispectral directory (strip band suffix + extension)."""
        stems = set()
        for f in sorted(os.listdir(image_dir)):
            if not f.lower().endswith((".tif", ".tiff")):
                continue
            name = os.path.splitext(f)[0]
            for suffix in BAND_SUFFIXES:
                if name.endswith(suffix):
                    stems.add(name[: -len(suffix)])
                    break
        return sorted(stems)

    @staticmethod
    def _collect_synthetic_stems(image_dir: str, mask_dir: str) -> tuple:
        """
        Collect synthetic image IDs and map them to mask stems by sorted index.

        Synthetic files: validation_result_<id>_<band>_.JPG (bands 1-4)
        Masks: DJI_DateTime_....png

        Returns:
            (mask_stems, synthetic_map) where synthetic_map = {mask_stem: synthetic_id}
        """
        import re
        # Collect unique synthetic IDs
        synthetic_ids = set()
        for f in os.listdir(image_dir):
            m = re.match(r"validation_result_(\d+)_\d+_\.JPG", f)
            if m:
                synthetic_ids.add(int(m.group(1)))
        synthetic_ids = sorted(synthetic_ids)

        # Collect mask stems (sorted alphabetically — same order used for splits)
        mask_stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(mask_dir)
            if f.lower().endswith((".png", ".tif", ".tiff"))
        ])

        assert len(synthetic_ids) == len(mask_stems), (
            f"Synthetic image count ({len(synthetic_ids)}) != mask count ({len(mask_stems)})"
        )

        synthetic_map = {stem: sid for stem, sid in zip(mask_stems, synthetic_ids)}
        return mask_stems, synthetic_map

    def _load_image(self, stem: str) -> np.ndarray:
        """
        Load image by stem. Returns HxWxC numpy array (float32).

        For multispectral: reads 4 separate band TIFs and stacks them.
        For synthetic: reads 4 separate band JPGs (validation_result_<id>_<band>_.JPG).
        For RGB: reads the JPG directly.
        """
        if self.input_type == "synthetic":
            sid = self._synthetic_map[stem]
            bands = []
            for band_idx in SYNTHETIC_BANDS:
                path = os.path.join(self.image_dir, f"validation_result_{sid}_{band_idx}_.JPG")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Synthetic band file not found: {path}"
                    )
                band = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if band is None:
                    raise FileNotFoundError(f"Could not load band: {path}")
                if band.ndim == 3:
                    band = band[:, :, 0]
                bands.append(band)
            return np.stack(bands, axis=-1).astype(np.float32)
        elif self.input_type == "multispectral":
            bands = []
            for suffix in BAND_SUFFIXES:
                # Try common TIF extensions
                path = None
                for ext in (".TIF", ".tif", ".tiff"):
                    candidate = os.path.join(self.image_dir, f"{stem}{suffix}{ext}")
                    if os.path.exists(candidate):
                        path = candidate
                        break
                if path is None:
                    raise FileNotFoundError(
                        f"Band file not found for stem '{stem}', suffix '{suffix}' in {self.image_dir}"
                    )
                band = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if band is None:
                    raise FileNotFoundError(f"Could not load band: {path}")
                # Each band TIF is stored as (H,W,3) with identical channels — take first
                if band.ndim == 3:
                    band = band[:, :, 0]
                bands.append(band)
            return np.stack(bands, axis=-1).astype(np.float32)
        else:
            # RGB — try common extensions
            img = None
            for ext in (".JPG", ".jpg", ".jpeg", ".png"):
                candidate = os.path.join(self.image_dir, f"{stem}{ext}")
                if os.path.exists(candidate):
                    img = cv2.imread(candidate, cv2.IMREAD_COLOR)
                    break
            if img is None:
                raise FileNotFoundError(f"Could not load RGB image for stem: {stem}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype(np.float32)

    def _load_mask(self, filepath: str) -> np.ndarray:
        """
        Load binary segmentation mask.

        Raw masks use 0=background, 255=weedy rice.
        Remaps to 0=background, 1=weedy rice for CrossEntropyLoss.
        """
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {filepath}")
        # Remap: 0 stays 0, 255 becomes 1
        mask = (mask > 0).astype(np.int64)
        return mask

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Per-channel min-max normalization to [0, 1]."""
        for c in range(image.shape[-1]):
            ch = image[..., c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 0:
                image[..., c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                image[..., c] = 0.0
        return image

    def __len__(self) -> int:
        return len(self.image_stems)

    def __getitem__(self, idx: int):
        stem = self.image_stems[idx]

        image = self._load_image(stem)
        mask = self._load_mask(os.path.join(self.mask_dir, stem + ".png"))

        # Normalize
        image = self._normalize(image)

        # Apply augmentations (albumentations handles HxWxC format)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to torch tensors
        # Image: CxHxW float32
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        # Mask: HxW int64
        mask = torch.from_numpy(mask).long()

        return image, mask


def get_transforms(split: str, image_size: int = 256) -> A.Compose:
    """
    Get augmentation pipeline for a given split.

    Args:
        split: 'train', 'val', or 'test'
        image_size: Target size for resizing
    """
    if split == "train":
        return A.Compose([
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    else:
        return A.Compose([
            A.CenterCrop(height=image_size, width=image_size),
        ])