from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.utils import ensure_dir


def to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        min_value = float(array.min())
        max_value = float(array.max())
        scale = max(max_value - min_value, 1e-6)
        normed = (array - min_value) / scale
        return np.clip(normed * 255.0, 0, 255).astype(np.uint8)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def preview_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3].astype(np.float32)
        rgb_min = rgb.min(axis=(0, 1), keepdims=True)
        rgb_max = rgb.max(axis=(0, 1), keepdims=True)
        rgb = (rgb - rgb_min) / np.clip(rgb_max - rgb_min, 1e-6, None)
        return to_uint8(rgb)
    if image.ndim == 3:
        return np.repeat(to_uint8(image[..., 0])[..., None], 3, axis=-1)
    return np.repeat(to_uint8(image)[..., None], 3, axis=-1)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0), alpha: float = 0.35) -> np.ndarray:
    base = image_rgb.astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    mask_3 = np.repeat((mask > 0)[..., None], 3, axis=-1)
    blended = np.where(mask_3, (1 - alpha) * base + alpha * overlay, base)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_prediction_panel(
    output_path: str | Path,
    image: np.ndarray,
    gt_mask: np.ndarray,
    probability_map: np.ndarray,
    pred_mask: np.ndarray,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    preview = preview_image(image)
    gt_rgb = np.repeat(to_uint8(gt_mask)[..., None], 3, axis=-1)
    prob_rgb = np.repeat(to_uint8(probability_map)[..., None], 3, axis=-1)
    pred_rgb = np.repeat(to_uint8(pred_mask)[..., None], 3, axis=-1)
    overlay = overlay_mask(preview, pred_mask)

    panels = [preview, gt_rgb, prob_rgb, pred_rgb, overlay]
    widths = [panel.shape[1] for panel in panels]
    heights = [panel.shape[0] for panel in panels]
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    offset_x = 0
    for panel in panels:
        canvas.paste(Image.fromarray(panel), (offset_x, 0))
        offset_x += panel.shape[1]
    canvas.save(output_path)
