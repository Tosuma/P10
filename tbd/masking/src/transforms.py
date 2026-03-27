from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationTransform:
    is_train: bool
    brightness: float = 0.0
    contrast: float = 0.0
    noise_std: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, image: np.ndarray, mask: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_train:
            return image, mask

        if self.rng.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        if image.shape[0] == image.shape[1]:
            k = int(self.rng.integers(0, 4))
        else:
            k = int(self.rng.integers(0, 2)) * 2
        if k:
            image = np.rot90(image, k=k, axes=(0, 1))
            mask = np.rot90(mask, k=k, axes=(0, 1))

        if modality == "rgb" and self.brightness > 0:
            brightness_scale = 1.0 + self.rng.uniform(-self.brightness, self.brightness)
            image = image * brightness_scale
        if modality == "rgb" and self.contrast > 0:
            mean = image.mean(axis=(0, 1), keepdims=True)
            contrast_scale = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
            image = (image - mean) * contrast_scale + mean
        if self.noise_std > 0:
            image = image + self.rng.normal(0.0, self.noise_std, size=image.shape)

        return image.copy(), mask.copy()


def build_transforms(config: dict, is_train: bool, seed: int) -> SegmentationTransform:
    aug_cfg = config.get("augmentations", {})
    return SegmentationTransform(
        is_train=is_train,
        brightness=float(aug_cfg.get("brightness", 0.0)),
        contrast=float(aug_cfg.get("contrast", 0.0)),
        noise_std=float(aug_cfg.get("noise_std", 0.0)),
        seed=seed,
    )
