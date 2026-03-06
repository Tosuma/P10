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
import torch
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

    def _extract_signatures_from_loader(
        self, loader: torch.utils.data.DataLoader
    ) -> np.ndarray:
        """
        Extract per-channel-mean spectral signature from a DataLoader.

        Uses the packed .npz cache and multi-worker loading — much faster than
        reading individual files per patch.  Returns (N, C) float32 array.
        """
        sigs = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"]          # (B, C, H, W)
                sigs.append(imgs.mean(dim=(2, 3)).cpu().numpy())   # (B, C)
        return np.concatenate(sigs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------

    def fit(self, train_loader: torch.utils.data.DataLoader) -> "PCAKMeans":
        """
        Fit scaler, PCA, and k-means on the training split.

        Args:
            train_loader: DataLoader over the training split (no augmentation).
        Returns:
            self (for chaining)
        """
        print("[PCAKMeans] Extracting signatures from training patches …")
        X = self._extract_signatures_from_loader(train_loader)
        print(f"  {X.shape[0]:,} patches, {X.shape[1]} channels")
        X_scaled = self.scaler.fit_transform(X)
        X_pca    = self.pca.fit_transform(X_scaled)
        print(f"  PCA: {self.n_components} components, "
              f"explained variance = {self.pca.explained_variance_ratio_.sum():.3f}")
        self.kmeans.fit(X_pca)
        print(f"  k-means: {self.n_clusters} clusters fitted")
        self._fitted = True
        return self

    def score(
        self, val_loader: torch.utils.data.DataLoader
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute anomaly scores (distance to nearest centroid) for given loader.

        Returns:
            scores: (N,) float32
            stems:  list of patch stems in loader order
        """
        assert self._fitted, "Call fit() before score()"
        X = self._extract_signatures_from_loader(val_loader)
        stems = [s for batch in val_loader for s in batch["stem"]]
        X_s = self.scaler.transform(X)
        X_p = self.pca.transform(X_s)
        dists = np.linalg.norm(
            X_p[:, np.newaxis, :] - self.kmeans.cluster_centers_[np.newaxis, :, :],
            axis=-1,
        )  # (N, k)
        return dists.min(axis=1).astype(np.float32), stems

    def run(
        self,
        output_dir: Path,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """Full pipeline: fit on train, score val, save results."""
        self.fit(train_loader)
        scores, valid_stems = self.score(val_loader)

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
