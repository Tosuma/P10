"""
Baseline 1: NDVI Thresholding.

Simplest possible anomaly detection baseline: pixels below an NDVI
threshold are flagged as potentially stressed vegetation.  No learning.

Anomaly score per pixel: −NDVI  (so higher score = more anomalous)
This inverts NDVI so the convention matches the flow model (high score =
more anomalous), making comparison across methods straightforward.

Threshold selection:
  - Fixed: domain knowledge (NDVI < 0.3 ≈ stressed / sparse vegetation)
  - Auto:  Otsu's method on the NDVI histogram partitions the population
           into two classes, finding the threshold that maximises inter-class
           variance — a principled, data-driven choice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import cv2


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Compute Otsu's threshold from a 1-D array of NDVI values.

    Otsu's method maximises inter-class variance between two Gaussian-like
    populations (healthy vs stressed vegetation in the NDVI histogram).
    """
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=n_bins, density=True)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    best_thresh = bin_centres[len(bin_centres) // 2]
    best_var = 0.0

    w0 = 0.0
    for i, (h, c) in enumerate(zip(hist, bin_centres)):
        w0 += h
        w1  = total - w0
        if w0 < 1e-9 or w1 < 1e-9:
            continue
        mu0 = (hist[:i+1] * bin_centres[:i+1]).sum() / w0
        mu1 = (hist[i+1:] * bin_centres[i+1:]).sum() / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = c

    return float(best_thresh)


class NDVIThreshold:
    """
    NDVI-based anomaly detector.

    Args:
        threshold: NDVI threshold below which a pixel is considered anomalous.
                   Passed as None to trigger automatic Otsu selection via
                   NDVIThreshold.from_data().
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    @classmethod
    def from_data(
        cls,
        patch_dir: Path,
        split_stems: list[str],
        max_samples: int = 2000,
        seed: int = 42,
    ) -> "NDVIThreshold":
        """
        Estimate threshold from Otsu's method on a subsample of training NDVI patches.

        Args:
            patch_dir:    Path to WeedyRice-patches/ root.
            split_stems:  Patch stems belonging to the training split.
            max_samples:  Maximum patches to subsample for histogram estimation.
            seed:         RNG seed for reproducible subsampling.
        """
        import random
        rng = random.Random(seed)
        sample = rng.sample(split_stems, min(max_samples, len(split_stems)))

        ndvi_dir = Path(patch_dir) / "NDVI"
        all_vals: list[np.ndarray] = []
        for stem in sample:
            arr = np.load(ndvi_dir / f"{stem}.npy").ravel()
            all_vals.append(arr)

        all_vals_arr = np.concatenate(all_vals)
        thresh = _otsu_threshold(all_vals_arr)
        print(f"[NDVIThreshold] Otsu threshold = {thresh:.4f}")
        return cls(threshold=thresh)

    # ------------------------------------------------------------------

    def score(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel anomaly score.

        Args:
            ndvi: (H, W) float32 NDVI values.
        Returns:
            (H, W) float32 anomaly scores = −NDVI (higher = more anomalous).
        """
        return -ndvi.astype(np.float32)

    def predict_mask(self, ndvi: np.ndarray) -> np.ndarray:
        """Binary mask: 1 where NDVI < threshold (potentially anomalous)."""
        return (ndvi < self.threshold).astype(np.uint8)

    def run_on_patch_dir(
        self,
        patch_dir: Path,
        split_stems: list[str],
        output_dir: Optional[Path] = None,
    ) -> dict[str, float]:
        """
        Score all patches in NDVI/ for the given split.

        Saves per-patch score arrays as .npy files if output_dir is given.

        Returns:
            Summary statistics: mean_score, std_score, fraction_anomalous.
        """
        ndvi_dir = Path(patch_dir) / "NDVI"
        if output_dir is not None:
            out = Path(output_dir) / "ndvi_threshold"
            out.mkdir(parents=True, exist_ok=True)

        all_scores: list[float] = []
        n_anom = 0
        n_total = 0

        for stem in split_stems:
            arr = np.load(ndvi_dir / f"{stem}.npy")
            scores = self.score(arr)
            mask   = self.predict_mask(arr)

            if output_dir is not None:
                np.save(out / f"{stem}.npy", scores)

            all_scores.append(float(scores.mean()))
            n_anom  += int(mask.sum())
            n_total += int(mask.size)

        return {
            "mean_score":         float(np.mean(all_scores)),
            "std_score":          float(np.std(all_scores)),
            "fraction_anomalous": n_anom / max(n_total, 1),
            "threshold":          self.threshold,
        }
