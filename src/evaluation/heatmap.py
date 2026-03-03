"""
Heatmap generation: assemble patch-level anomaly scores into full-image maps
and export as GeoTIFF (preserving spatial metadata for GIS usage).

Workflow
--------
1. The inference DataLoader returns (scores, record_idx, patch_y, patch_x) for
   each patch.
2. `assemble_heatmap()` averages overlapping patch scores into a 2-D spatial
   grid aligned with the original image.
3. `export_geotiff()` writes the heatmap as a GeoTIFF using the rasterio profile
   of the original image, so that the anomaly map can be overlaid in QGIS /
   ArcGIS with correct georeferencing.

Why GeoTIFF?
  Drone surveys produce georeferenced images.  Saving the anomaly map as a
  plain PNG discards the spatial reference — the farmer or agronomist cannot
  locate the detected stress regions in the field without GPS context.
  GeoTIFF preserves CRS + affine transform, enabling direct overlay on field
  maps and integration with precision agriculture workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import from_bounds


def assemble_heatmap(
    patch_scores: list[tuple[float, int, int]],
    image_height: int,
    image_width: int,
    patch_size: int,
) -> np.ndarray:
    """
    Assemble overlapping patch scores into a 2-D anomaly map by averaging.

    Parameters
    ----------
    patch_scores : list of (score, y, x) tuples
        One tuple per patch: scalar anomaly score, top-left y, top-left x.
    image_height, image_width : int
        Spatial dimensions of the original image.
    patch_size : int

    Returns
    -------
    heatmap : np.ndarray  (H, W), float32
        Average anomaly score per pixel.  NaN where no patch covers a pixel.
    """
    accumulator = np.zeros((image_height, image_width), dtype=np.float64)
    count       = np.zeros((image_height, image_width), dtype=np.int32)

    for score, y, x in patch_scores:
        y2 = min(y + patch_size, image_height)
        x2 = min(x + patch_size, image_width)
        accumulator[y:y2, x:x2] += score
        count[y:y2, x:x2] += 1

    with np.errstate(invalid="ignore"):
        heatmap = np.where(count > 0, accumulator / count, np.nan)

    return heatmap.astype(np.float32)


def smooth_heatmap(
    heatmap: np.ndarray,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Gaussian smoothing to reduce patch-boundary artefacts in the heatmap.

    Parameters
    ----------
    heatmap : (H, W) float32
    sigma : float
        Gaussian standard deviation in pixels.

    Returns
    -------
    smoothed : (H, W) float32
    """
    from scipy.ndimage import gaussian_filter
    # Fill NaN with mean before smoothing (NaN at image borders)
    filled = heatmap.copy()
    nan_mask = np.isnan(filled)
    filled[nan_mask] = np.nanmean(filled)
    smoothed = gaussian_filter(filled, sigma=sigma).astype(np.float32)
    smoothed[nan_mask] = np.nan
    return smoothed


def normalise_heatmap(
    heatmap: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalise heatmap values to [0, 1] for visualisation.

    Parameters
    ----------
    method : 'minmax' | 'zscore' | 'percentile'
    """
    valid = heatmap[~np.isnan(heatmap)]
    if method == "minmax":
        lo, hi = valid.min(), valid.max()
        out = (heatmap - lo) / max(hi - lo, 1e-6)
    elif method == "zscore":
        out = (heatmap - valid.mean()) / max(valid.std(), 1e-6)
        # Rescale z to [0,1] using tanh saturation at ±3σ
        out = (np.tanh(out / 3) + 1) / 2
    elif method == "percentile":
        lo = np.nanpercentile(heatmap, 2)
        hi = np.nanpercentile(heatmap, 98)
        out = np.clip((heatmap - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")
    return out.astype(np.float32)


def export_geotiff(
    heatmap: np.ndarray,
    output_path: str | Path,
    reference_profile: Optional[dict] = None,
    crs_epsg: int = 4326,
    bounds: Optional[tuple] = None,
) -> None:
    """
    Export heatmap as a single-band GeoTIFF.

    Parameters
    ----------
    heatmap : (H, W) float32
    output_path : str or Path
    reference_profile : rasterio profile dict (from the source image).
        If provided, inherits CRS and affine transform directly.
    crs_epsg : int
        EPSG code to use if no reference_profile (e.g. 4326 for WGS84).
    bounds : (left, bottom, right, top) in CRS units
        Required if no reference_profile.
    """
    H, W = heatmap.shape
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if reference_profile is not None:
        profile = reference_profile.copy()
        profile.update({
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": W,
            "height": H,
            "compress": "lzw",
            "predictor": 3,         # Horizontal predictor for float data
            "nodata": float("nan"),
        })
    else:
        from rasterio.crs import CRS
        if bounds is None:
            bounds = (0, 0, W, H)   # Pixel coordinates as fallback
        transform = from_bounds(*bounds, width=W, height=H)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": W,
            "height": H,
            "crs": CRS.from_epsg(crs_epsg),
            "transform": transform,
            "compress": "lzw",
            "predictor": 3,
            "nodata": float("nan"),
        }

    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(heatmap[np.newaxis, :, :])  # (1, H, W)

    print(f"Exported heatmap → {output_path}")
