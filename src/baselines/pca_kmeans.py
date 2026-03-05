"""
Baseline 2: PCA + k-means clustering on spectral patch signatures.

Pipeline:
1. Represent each 128×128 patch by its per-channel mean → 10-dim spectral
   signature vector. (Memory: O(10) per patch instead of O(10·128·128).)
2. Fit PCA (n_components=16) on training-set signatures.
3. Fit k-means (n_clusters=8) on PCA-projected training signatures.
4. Anomaly score = Euclidean distance to the nearest cluster centroid.

Design choice — per-patch mean spectral signature:
  Spatial information is discarded in favour of spectral composition.  This
  mirrors the classical remote-sensing workflow of per-pixel / per-patch
  spectral analysis and provides a fair comparison to spatial baselines
  (MAE + flow) that also operate at the patch level.

Limitations:
  - Ignores all spatial texture within a patch.
  - k-means is sensitive to initialisation and k; eight clusters covers
    typical rice-field land-cover classes (healthy rice, weedy rice, soil,
    water, shadows, crop edges) without excessive fragmentation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class PCAKMeans:
    """
    PCA + k-means anomaly detector for multispectral + VI patches.

    Args:
        n_components: PCA components (default 16).
        n_clusters:   k-means clusters (default 8).
        seed:         Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 16,
        n_clusters: int = 8,
        seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_clusters   = n_clusters
        self.seed         = seed
        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=n_components, random_state=seed)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        self._fitted = False

    # ------------------------------------------------------------------

    def _extract_signatures(
        self, patch_dir: Path, stems: list[str]
    ) -> np.ndarray:
        """
        Extract 10-dim per-channel-mean spectral signature for each patch.

        Loads RGB, Multispectral, NDVI, NDRE, SAVI and concatenates per-channel
        means.  Returns (N, 10) float32 array.
        """
        import cv2
        sigs = []
        rgb_dir = patch_dir / "RGB"
        ms_dir  = patch_dir / "Multispectral"
        vi_dirs = {vi: patch_dir / vi for vi in ("NDVI", "NDRE", "SAVI")}

        for stem in stems:
            rgb_bgr = cv2.imread(str(rgb_dir / f"{stem}.jpg"))
            if rgb_bgr is None:
                continue
            rgb = (rgb_bgr[:, :, ::-1].astype(np.float32) / 255.0).mean(axis=(0, 1))  # (3,)
            ms  = np.load(ms_dir / f"{stem}.npy").mean(axis=(0, 1))                    # (4,)
            vi  = np.array([
                np.load(vi_dirs["NDVI"] / f"{stem}.npy").mean(),
                np.load(vi_dirs["NDRE"] / f"{stem}.npy").mean(),
                np.load(vi_dirs["SAVI"] / f"{stem}.npy").mean(),
            ])                                                                           # (3,)
            sigs.append(np.concatenate([rgb, ms, vi]))                                  # (10,)

        return np.stack(sigs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------

    def fit(self, patch_dir: Path, train_stems: list[str]) -> "PCAKMeans":
        """
        Fit scaler, PCA, and k-means on the training split.

        Returns:
            self (for chaining)
        """
        print(f"[PCAKMeans] Extracting signatures from {len(train_stems):,} train patches …")
        X = self._extract_signatures(Path(patch_dir), train_stems)
        X_scaled = self.scaler.fit_transform(X)
        X_pca    = self.pca.fit_transform(X_scaled)
        print(f"  PCA: {self.n_components} components, "
              f"explained variance = {self.pca.explained_variance_ratio_.sum():.3f}")
        self.kmeans.fit(X_pca)
        print(f"  k-means: {self.n_clusters} clusters fitted")
        self._fitted = True
        return self

    def score(
        self, patch_dir: Path, stems: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute anomaly scores (distance to nearest centroid) for given stems.

        Returns:
            scores: (N,) float32
            stems:  filtered list (patches that could be loaded)
        """
        assert self._fitted, "Call fit() before score()"
        X = self._extract_signatures(Path(patch_dir), stems)
        X_s = self.scaler.transform(X)
        X_p = self.pca.transform(X_s)
        # Euclidean distance to nearest centroid
        dists = np.linalg.norm(
            X_p[:, np.newaxis, :] - self.kmeans.cluster_centers_[np.newaxis, :, :],
            axis=-1,
        )  # (N, k)
        return dists.min(axis=1).astype(np.float32), stems[: len(X)]

    def run(
        self,
        patch_dir: Path,
        output_dir: Path,
        train_stems: list[str],
        val_stems: list[str],
    ) -> dict[str, float]:
        """Full pipeline: fit on train, score val, save results."""
        self.fit(patch_dir, train_stems)
        scores, valid_stems = self.score(patch_dir, val_stems)

        output_dir = Path(output_dir) / "pca_kmeans"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "val_scores.npy", scores)

        self.save(output_dir / "model.pkl")
        stats = {
            "mean_score": float(scores.mean()),
            "std_score":  float(scores.std()),
            "min_score":  float(scores.min()),
            "max_score":  float(scores.max()),
        }
        import json
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "PCAKMeans":
        with open(path, "rb") as f:
            return pickle.load(f)
