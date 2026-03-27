from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.transforms import build_transforms
from src.utils import parse_path_field, read_csv_rows

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_multichannel(paths: list[str]) -> np.ndarray:
    channels = []
    for path in paths:
        array = np.asarray(Image.open(path), dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        channels.append(array)
    return np.stack(channels, axis=-1)


def load_synthetic(path: str) -> np.ndarray:
    file_path = Path(path)
    if file_path.suffix.lower() == ".npy":
        array = np.load(file_path)
    elif file_path.suffix.lower() == ".npz":
        payload = np.load(file_path)
        first_key = list(payload.keys())[0]
        array = payload[first_key]
    else:
        raise ValueError(f"Unsupported synthetic MSI format: {file_path}")
    if array.ndim != 3:
        raise ValueError(f"Synthetic MSI must be 3D, got shape {array.shape}")
    if array.shape[0] < array.shape[-1]:
        array = np.moveaxis(array, 0, -1)
    return array.astype(np.float32)


def load_mask(path: str) -> np.ndarray:
    mask = np.asarray(Image.open(path), dtype=np.float32)
    return (mask > 0).astype(np.float32)


def crop_hw(image: np.ndarray, left: int, top: int, width: int, height: int) -> np.ndarray:
    return image[top : top + height, left : left + width, ...]


def load_image_for_modality(sample: dict[str, str], modality: str) -> np.ndarray:
    if modality == "rgb":
        return load_rgb(sample["rgb_path"])
    if modality == "synthetic_msi":
        if not sample.get("synthetic_msi_path"):
            raise FileNotFoundError(f"Missing synthetic MSI path for sample {sample['sample_id']}")
        return load_synthetic(sample["synthetic_msi_path"])
    if modality == "real_msi":
        paths = parse_path_field(sample["real_msi_path"])
        if not paths:
            raise FileNotFoundError(f"Missing real MSI path(s) for sample {sample['sample_id']}")
        return load_multichannel(paths)
    if modality == "rgb_real_msi":
        rgb = load_rgb(sample["rgb_path"])
        real_msi = load_image_for_modality(sample, "real_msi")
        return np.concatenate([rgb, real_msi], axis=-1)
    if modality == "rgb_synthetic_msi":
        rgb = load_rgb(sample["rgb_path"])
        synthetic_msi = load_image_for_modality(sample, "synthetic_msi")
        return np.concatenate([rgb, synthetic_msi], axis=-1)
    raise ValueError(f"Unsupported modality: {modality}")


def normalize_image(image: np.ndarray, modality: str, normalization: dict[str, Any]) -> np.ndarray:
    if modality == "rgb":
        mean = np.asarray(normalization.get("mean", IMAGENET_MEAN), dtype=np.float32)
        std = np.asarray(normalization.get("std", IMAGENET_STD), dtype=np.float32)
    else:
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        std = np.asarray(normalization["std"], dtype=np.float32)
    std = np.clip(std, 1e-6, None)
    image = (image - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return image.astype(np.float32)


class WeedSegmentationDataset(Dataset):
    def __init__(
        self,
        sample_manifest_path: str,
        patch_manifest_path: str,
        modality: str,
        normalization: dict[str, Any],
        is_train: bool,
        transform_config: dict[str, Any],
        seed: int,
    ) -> None:
        self.samples = {row["sample_id"]: row for row in read_csv_rows(sample_manifest_path)}
        self.patches = read_csv_rows(patch_manifest_path)
        self.modality = modality
        self.normalization = normalization
        self.transforms = build_transforms(transform_config, is_train=is_train, seed=seed)

    def __len__(self) -> int:
        return len(self.patches)

    def _load_image(self, sample: dict[str, str]) -> np.ndarray:
        return load_image_for_modality(sample, self.modality)

    def __getitem__(self, index: int) -> dict[str, Any]:
        patch = self.patches[index]
        sample = self.samples[patch["sample_id"]]
        left, top = int(patch["x"]), int(patch["y"])
        width, height = int(patch["width"]), int(patch["height"])

        image = crop_hw(self._load_image(sample), left, top, width, height)
        mask = crop_hw(load_mask(sample["mask_path"])[..., None], left, top, width, height)[..., 0]
        image, mask = self.transforms(image, mask, self.modality)
        image = normalize_image(image, self.modality, self.normalization)

        image_tensor = torch.from_numpy(np.moveaxis(image, -1, 0)).float()
        mask_tensor = torch.from_numpy(mask).float()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "metadata": {
                "sample_id": sample["sample_id"],
                "patch_id": patch["patch_id"],
                "origin": patch["origin"],
                "coords": [left, top, width, height],
                "split": patch["split"],
            },
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    metadata = [item["metadata"] for item in batch]
    return {"image": images, "mask": masks, "metadata": metadata}


def compute_channel_stats(sample_manifest_path: str, patch_manifest_path: str, modality: str) -> dict[str, Any]:
    sample_rows = {row["sample_id"]: row for row in read_csv_rows(sample_manifest_path)}
    patch_rows = read_csv_rows(patch_manifest_path)
    total_sum = None
    total_sq_sum = None
    total_pixels = 0

    for patch in patch_rows:
        sample = sample_rows[patch["sample_id"]]
        left, top = int(patch["x"]), int(patch["y"])
        width, height = int(patch["width"]), int(patch["height"])

        image = crop_hw(load_image_for_modality(sample, modality), left, top, width, height)

        flat = image.reshape(-1, image.shape[-1]).astype(np.float64)
        patch_sum = flat.sum(axis=0)
        patch_sq_sum = np.square(flat).sum(axis=0)
        if total_sum is None:
            total_sum = patch_sum
            total_sq_sum = patch_sq_sum
        else:
            total_sum += patch_sum
            total_sq_sum += patch_sq_sum
        total_pixels += flat.shape[0]

    if total_pixels == 0:
        raise RuntimeError("No pixels available to compute channel statistics.")

    mean = total_sum / total_pixels
    var = (total_sq_sum / total_pixels) - np.square(mean)
    std = np.sqrt(np.clip(var, 1e-12, None))
    return {"mean": mean.tolist(), "std": std.tolist()}


def build_dataloader(
    sample_manifest_path: str,
    patch_manifest_path: str,
    modality: str,
    normalization: dict[str, Any],
    batch_size: int,
    num_workers: int,
    is_train: bool,
    transform_config: dict[str, Any],
    seed: int,
) -> DataLoader:
    dataset = WeedSegmentationDataset(
        sample_manifest_path=sample_manifest_path,
        patch_manifest_path=patch_manifest_path,
        modality=modality,
        normalization=normalization,
        is_train=is_train,
        transform_config=transform_config,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
