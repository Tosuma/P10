"""
Anomaly heatmap generation and GeoTIFF export.

Stitches per-patch, per-token anomaly scores into full-image score maps
and exports them as:
  - PNG: colourised overlay on the RGB drone image
  - GeoTIFF: float32 raster preserving spatial metadata for QGIS / ArcGIS

Spatial layout:
  Full image (after resizing in patch extraction): 2560 × 1920 pixels
  Patch grid: 20 cols × 15 rows of 128×128 patches
  Token grid per patch: 8 × 8 tokens of 16×16 pixels (from ViT patch_size=16)
  Full token grid: (15×8) rows × (20×8) cols = 120 × 160 tokens

Saving as GeoTIFF with the spatial profile from the original rasterio read
preserves the CRS and affine transform, so the anomaly map can be overlaid
on the orthoimage in a GIS application — directly useful for field-level
decision support (future work section of thesis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


class AnomalyHeatmap:
    """
    Stitch patch-level token scores into full-image anomaly maps.

    Args:
        patch_size:  Spatial size of each extracted patch in pixels (default 128).
        token_size:  Spatial size of each ViT token in pixels (default 16,
                     matching patch_size=16 in the ViT encoder).
        image_w:     Width of the full resized image (default 2560 = 20 × 128).
        image_h:     Height of the full resized image (default 1920 = 15 × 128).
        n_cols:      Number of patch columns (default 20).
        n_rows:      Number of patch rows (default 15).
    """

    def __init__(
        self,
        patch_size: int = 128,
        token_size: int = 16,
        image_w: int = 2560,
        image_h: int = 1920,
        n_cols: int = 20,
        n_rows: int = 15,
    ) -> None:
        self.patch_size = patch_size
        self.token_size = token_size
        self.image_w    = image_w
        self.image_h    = image_h
        self.n_cols     = n_cols
        self.n_rows     = n_rows

        self.tokens_per_patch_side = patch_size // token_size   # = 8
        self.score_map_w = n_cols * self.tokens_per_patch_side  # = 160
        self.score_map_h = n_rows * self.tokens_per_patch_side  # = 120

    # ------------------------------------------------------------------

    def stitch(
        self,
        patch_scores: dict[str, np.ndarray],
        image_stem: str,
    ) -> np.ndarray:
        """
        Assemble per-patch token scores into a full-image score map.

        Args:
            patch_scores: dict mapping patch_stem → (N_tokens,) float32.
                          N_tokens = (patch_size / token_size)² = 64.
            image_stem:   Parent image stem used to filter patch_scores.
        Returns:
            score_map: (score_map_h, score_map_w) float32 = (120, 160).
                       NaN where patches are missing.
        """
        import re
        score_map = np.full((self.score_map_h, self.score_map_w), np.nan, dtype=np.float32)
        t = self.tokens_per_patch_side

        for stem, scores in patch_scores.items():
            if not stem.startswith(image_stem):
                continue
            m = re.search(r"_r(\d{3})_c(\d{3})$", stem)
            if m is None:
                continue
            row, col = int(m.group(1)), int(m.group(2))
            token_scores = np.asarray(scores, dtype=np.float32).reshape(t, t)
            r0, r1 = row * t, (row + 1) * t
            c0, c1 = col * t, (col + 1) * t
            score_map[r0:r1, c0:c1] = token_scores

        return score_map

    def to_heatmap_image(
        self,
        score_map: np.ndarray,
        rgb_image: np.ndarray,
        alpha: float = 0.6,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Colourise score_map and blend with the RGB image.

        Args:
            score_map:  (H_scores, W_scores) float32.
            rgb_image:  (H_img, W_img, 3) uint8 RGB image.
            alpha:      Heatmap opacity (0 = transparent, 1 = opaque).
            colormap:   OpenCV colourmap constant (default COLORMAP_JET).
        Returns:
            (H_img, W_img, 3) uint8 BGR overlay image.
        """
        # Normalise scores to [0, 255]
        s = score_map.copy()
        nan_mask = np.isnan(s)
        s[nan_mask] = np.nanmin(s)
        smin, smax = s.min(), s.max()
        if smax > smin:
            s = (s - smin) / (smax - smin)
        s = (s * 255).astype(np.uint8)

        # Resize to image dimensions
        heat = cv2.applyColorMap(s, colormap)                             # BGR, (H_s, W_s, 3)
        heat = cv2.resize(heat, (rgb_image.shape[1], rgb_image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

        # Convert RGB image to BGR for cv2 blending
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) if rgb_image.shape[2] == 3 else rgb_image
        return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)

    def save_geotiff(
        self,
        score_map: np.ndarray,
        output_path: Path,
        reference_profile: Optional[dict] = None,
    ) -> None:
        """
        Save score map as a float32 GeoTIFF.

        The spatial reference (CRS, affine transform) is copied from
        reference_profile if provided, so the exported raster can be loaded
        directly into QGIS / ArcGIS with correct georeferencing.

        Args:
            score_map:         (H, W) float32 anomaly scores.
            output_path:       Destination .tif path.
            reference_profile: rasterio profile dict from the source image.
        """
        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for GeoTIFF export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver":  "GTiff",
            "dtype":   "float32",
            "width":   score_map.shape[1],
            "height":  score_map.shape[0],
            "count":   1,
            "compress": "deflate",
        }
        if reference_profile:
            for k in ("crs", "transform"):
                if k in reference_profile:
                    profile[k] = reference_profile[k]

        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(score_map.astype(np.float32), 1)

    def save_png(
        self,
        score_map: np.ndarray,
        rgb_image: np.ndarray,
        output_path: Path,
        alpha: float = 0.6,
    ) -> None:
        """Save coloured overlay as a PNG file."""
        overlay = self.to_heatmap_image(score_map, rgb_image, alpha)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)
