"""
Baseline 1: NDVI Thresholding.

Classical remote-sensing approach: pixels with NDVI below a threshold are
classified as anomalous (stressed, dead, or bare soil).

Strengths
---------
- Computationally trivial; interpretable.
- Strong prior for vegetation health detection.

Weaknesses
----------
- Single threshold is extremely sensitive to illumination, atmospheric
  conditions, and sensor calibration across different flight days.
- Cannot distinguish between stress types (drought, disease, weed competition).
- Fails in areas of dense canopy where NDVI saturates.

Role in thesis
--------------
This is the simplest possible baseline.  We expect all learned methods to
outperform it.  If they don't, something is wrong with the learning pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class NDVIThreshold:
    """
    NDVI-based anomaly detector.

    The anomaly score for each pixel is defined as:
        score = threshold - NDVI(pixel)

    Positive score → pixel is below threshold (anomalous).
    This continuous formulation allows AUROC computation.

    Parameters
    ----------
    threshold : float
        NDVI value below which a pixel is considered stressed.
        Literature typical ranges: healthy crops 0.4–0.8, stressed < 0.3.
        Default 0.35 is a reasonable field value — tune per-dataset.
    nir_channel : int
        Channel index of the NIR band in the input image array.
    red_channel : int
        Channel index of the Red band.
    """

    def __init__(
        self,
        threshold: float = 0.35,
        nir_channel: int = 6,    # Default assumes [R,G,B,G_ms,R_ms,RE,NIR] ordering
        red_channel: int = 4,
    ):
        self.threshold = threshold
        self.nir_channel = nir_channel
        self.red_channel = red_channel
        self._eps = 1e-8

    def fit(self, X: np.ndarray) -> "NDVIThreshold":
        """No-op: NDVI threshold requires no fitting."""
        return self

    def ndvi(self, image: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel NDVI from a (H, W, C) image array.

        Returns
        -------
        ndvi : np.ndarray  (H, W), float32, range [-1, 1]
        """
        nir = image[..., self.nir_channel].astype(np.float32)
        red = image[..., self.red_channel].astype(np.float32)
        return (nir - red) / (nir + red + self._eps)

    def score(self, image: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel anomaly score.

        Higher score → more likely anomalous.

        Parameters
        ----------
        image : np.ndarray  (H, W, C)

        Returns
        -------
        anomaly_map : np.ndarray  (H, W), float32
        """
        ndvi_map = self.ndvi(image)
        return (self.threshold - ndvi_map).astype(np.float32)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Binary anomaly map: 1 = anomalous, 0 = normal."""
        return (self.score(image) > 0).astype(np.uint8)
