"""
Band loader: reads and spatially aligns per-band raster files into a unified
multi-channel NumPy array.

Design rationale
----------------
Drone multispectral sensors (e.g. MicaSense RedEdge / Parrot Sequoia) capture
each spectral band as an independent GeoTIFF with its own geotransform.  Slight
misalignment between bands is common due to the multi-lens rig geometry.  We
use rasterio + pyproj to re-project every band onto a common grid (the RGB
image footprint) via affine warping so that every channel corresponds to the
same ground pixel.

Expected directory layout
--------------------------
    root/
    ├── RGB/
    │   └── DJI_<id>.JPG          (or .TIF)
    └── Multispectral/
        ├── DJI_<id>_G.TIF
        ├── DJI_<id>_R.TIF
        ├── DJI_<id>_RE.TIF
        └── DJI_<id>_NIR.TIF

The band order in the returned array is always:
    [B, G, R, RE, NIR]    — i.e. RGB comes first, spectral bands appended.

If only multispectral (no RGB) is required, set `include_rgb=False`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

# Default spectral band suffixes — override via config if your dataset differs
DEFAULT_MS_SUFFIXES: list[str] = ["_G", "_R", "_RE", "_NIR"]


def _load_raster_as_array(path: str | Path) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    """
    Load a single-band or multi-band GeoTIFF/JPEG into a (H, W, C) float32 array.

    Returns
    -------
    array : np.ndarray  shape (H, W, C), dtype float32
    profile : rasterio profile dict (CRS, transform, etc.)
    """
    with rasterio.open(str(path)) as src:
        data = src.read().astype(np.float32)   # shape: (C, H, W)
        profile = src.profile
    return data.transpose(1, 2, 0), profile    # → (H, W, C)


def _reproject_band_to_reference(
    band_path: str | Path,
    ref_profile: rasterio.profiles.Profile,
) -> np.ndarray:
    """
    Reproject a single-band raster to match the spatial extent + resolution
    defined by `ref_profile`.

    This ensures per-pixel alignment between the multispectral bands and the
    reference (usually the RGB image or the first MS band).

    Returns
    -------
    band : np.ndarray  shape (H, W), dtype float32
    """
    dst_crs = ref_profile["crs"]
    dst_transform = ref_profile["transform"]
    dst_width = ref_profile["width"]
    dst_height = ref_profile["height"]

    destination = np.zeros((dst_height, dst_width), dtype=np.float32)

    with rasterio.open(str(band_path)) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return destination


def load_image_pair(
    rgb_path: Optional[str | Path],
    ms_dir: str | Path,
    stem: str,
    ms_suffixes: Sequence[str] = DEFAULT_MS_SUFFIXES,
    include_rgb: bool = True,
    align_to_rgb: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Load a single RGB + multispectral image pair into one aligned tensor.

    Parameters
    ----------
    rgb_path:
        Full path to the RGB file.  Required if `include_rgb=True`.
    ms_dir:
        Directory containing per-band TIF files.
    stem:
        Filename stem shared by all band files (e.g. 'DJI_0001').
    ms_suffixes:
        List of suffixes appended to `stem` for each band (e.g. ['_G', '_R', ...]).
    include_rgb:
        Whether to include the 3-channel RGB as the first channels.
    align_to_rgb:
        If True, reproject all MS bands to match the RGB spatial grid.
        If False, use the first MS band as reference grid.

    Returns
    -------
    image : np.ndarray  shape (H, W, C), dtype float32
        Channels: [R, G, B, G_ms, R_ms, RE, NIR] if include_rgb else [G, R, RE, NIR]
        (exact order depends on ms_suffixes)
    meta : dict
        Rasterio profile of the reference image (for GeoTIFF export later).
    """
    ms_dir = Path(ms_dir)

    # ── Load RGB ──────────────────────────────────────────────────────────────
    if include_rgb:
        assert rgb_path is not None, "rgb_path must be provided when include_rgb=True"
        rgb_array, ref_profile = _load_raster_as_array(rgb_path)
        # If RGB has alpha or extra channels, keep only 3
        rgb_array = rgb_array[..., :3]  # (H, W, 3)
    else:
        ref_profile = None
        rgb_array = None

    # ── Locate MS band files ──────────────────────────────────────────────────
    ms_paths: list[Path] = []
    for suffix in ms_suffixes:
        found = None
        for ext in (".TIF", ".tif", ".tiff", ".TIFF"):
            candidate = ms_dir / f"{stem}{suffix}{ext}"
            if candidate.exists():
                found = candidate
                break
        if found is None:
            raise FileNotFoundError(
                f"Band file missing: stem='{stem}', suffix='{suffix}' in {ms_dir}"
            )
        ms_paths.append(found)

    # ── Set reference profile if no RGB ───────────────────────────────────────
    if ref_profile is None:
        _, ref_profile = _load_raster_as_array(ms_paths[0])
        # Ensure CRS / transform are present; raw JPEGs won't have them
        if ref_profile.get("crs") is None:
            # Fall back: skip reprojection, just load and resize to common shape
            align_to_rgb = False

    # ── Load & align MS bands ─────────────────────────────────────────────────
    ms_bands: list[np.ndarray] = []
    for bp in ms_paths:
        if align_to_rgb and ref_profile.get("crs") is not None:
            band = _reproject_band_to_reference(bp, ref_profile)
        else:
            arr, _ = _load_raster_as_array(bp)
            band = arr[..., 0]  # single-band TIF
            # Resize to match reference if shapes differ
            if rgb_array is not None and band.shape != rgb_array.shape[:2]:
                import cv2
                h, w = rgb_array.shape[:2]
                band = cv2.resize(band, (w, h), interpolation=cv2.INTER_LINEAR)
        ms_bands.append(band)

    ms_array = np.stack(ms_bands, axis=-1)  # (H, W, num_bands)

    # ── Concatenate channels ──────────────────────────────────────────────────
    if include_rgb and rgb_array is not None:
        image = np.concatenate([rgb_array, ms_array], axis=-1)
    else:
        image = ms_array

    return image.astype(np.float32), ref_profile


def normalize_channels(image: np.ndarray, percentile: float = 2.0) -> np.ndarray:
    """
    Robust per-channel normalization to [0, 1] using percentile clipping.

    Using a 2%/98% clip avoids letting extreme outlier pixels (sensor noise,
    overexposed specular reflections) dominate the normalization range —
    important for agricultural drone imagery where sunlit water patches can
    saturate individual bands.

    Parameters
    ----------
    image : np.ndarray  (H, W, C)
    percentile : float  lower/upper clip percentile

    Returns
    -------
    np.ndarray  (H, W, C), dtype float32, values in [0, 1]
    """
    out = image.astype(np.float32).copy()
    for c in range(out.shape[-1]):
        ch = out[..., c]
        lo = np.percentile(ch, percentile)
        hi = np.percentile(ch, 100.0 - percentile)
        if hi - lo > 1e-6:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        else:
            out[..., c] = 0.0
    return out
