from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def resolve_target_config(config: dict[str, Any] | None) -> dict[str, Any]:
    target_config = (config or {}).get("target", {})
    mode = str(target_config.get("mode", "binary"))
    halo_radius_px = int(target_config.get("halo_radius_px", 0))
    halo_min_value = float(target_config.get("halo_min_value", 0.35))
    if mode not in {"binary", "fuzzy_halo"}:
        raise ValueError(f"Unsupported target mode: {mode}")
    if halo_radius_px < 0:
        raise ValueError("target.halo_radius_px must be >= 0")
    if not 0.0 <= halo_min_value <= 1.0:
        raise ValueError("target.halo_min_value must be between 0 and 1")
    return {
        "mode": mode,
        "halo_radius_px": halo_radius_px,
        "halo_min_value": halo_min_value,
    }


def _halo_value(distance_px: int, radius_px: int, halo_min_value: float) -> float:
    if radius_px <= 0:
        return 0.0
    if radius_px == 1:
        return float((1.0 + halo_min_value) / 2.0)
    fraction = (distance_px - 1) / max(radius_px - 1, 1)
    near_boundary_value = 1.0 - 1e-3
    return float((1.0 - fraction) * near_boundary_value + fraction * halo_min_value)


def build_target_mask(hard_mask: np.ndarray, target_config: dict[str, Any]) -> np.ndarray:
    hard_mask = (np.asarray(hard_mask, dtype=np.float32) > 0).astype(np.float32)
    if target_config["mode"] != "fuzzy_halo" or target_config["halo_radius_px"] <= 0:
        return hard_mask

    radius_px = int(target_config["halo_radius_px"])
    halo_min_value = float(target_config["halo_min_value"])
    target_mask = hard_mask.copy()
    assigned = hard_mask > 0
    current = torch.from_numpy(hard_mask[None, None, ...])

    for distance_px in range(1, radius_px + 1):
        dilated = F.max_pool2d(current, kernel_size=3, stride=1, padding=1)
        dilated_np = dilated[0, 0].numpy() > 0
        ring = np.logical_and(dilated_np, np.logical_not(assigned))
        if ring.any():
            target_mask[ring] = _halo_value(distance_px, radius_px, halo_min_value)
            assigned = np.logical_or(assigned, ring)
        current = dilated

    return target_mask.astype(np.float32)


def target_views_from_hard_mask(hard_mask: np.ndarray, config: dict[str, Any] | None) -> dict[str, np.ndarray]:
    target_config = resolve_target_config(config)
    relaxed_mask = build_target_mask(hard_mask, target_config)
    return {
        "original": (np.asarray(hard_mask, dtype=np.float32) > 0).astype(np.float32),
        "relaxed": relaxed_mask,
    }
