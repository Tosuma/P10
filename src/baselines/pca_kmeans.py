"""
Baseline 2: PCA + k-means anomaly detection on spectral bands.

Approach
--------
1. Flatten each image into a set of per-pixel feature vectors (spectral bands).
2. Fit PCA to reduce dimensionality (retaining 95% variance).
3. Fit k-means on the PCA-projected data.
4. Anomaly score = distance from each pixel to its nearest cluster centroid.

Why this approach?
  PCA + k-means is a classical unsupervised anomaly detection baseline that
  exploits the spectral redundancy of multispectral imagery.  Healthy
  vegetation pixels form tight clusters in spectral space; stressed or
  anomalous pixels are outliers.

Limitations
  - Distance-to-centroid is not a proper probability score.
  - k-means is sensitive to the number of clusters k (we sweep k).
  - PCA may discard anomaly-relevant variance in higher components.

Role in thesis
--------------
Baseline 2 represents the category of "hand-crafted spectral features +
classical clustering".  It is computationally cheap but makes strong linearity
assumptions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class PCAKMeansAnomalyDetector:
    """
    PCA → k-means anomaly detector operating on per-pixel spectral vectors.

    Parameters
    ----------
    n_clusters : int
        Number of k-means clusters.
    pca_variance : float
        Fraction of variance to retain in PCA (0–1).  None = skip PCA.
    scale : bool
        Standardise features before PCA.  Should be True for mixed-scale
        inputs (RGB 0–255 mixed with NDVI in [-1,1]).
    random_state : int
    """

    def __init__(
        self,
        n_clusters: int = 8,
        pca_variance: float = 0.95,
        scale: bool = True,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance
        self.scale = scale
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = StandardScaler() if scale else None
        self.pca: Optional[PCA] = (
            PCA(n_components=pca_variance, svd_solver="full") if pca_variance else None
        )
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=5,
            batch_size=4096,
        )
        self._fitted = False

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Apply scaler + PCA (must be fitted)."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        return X

    def fit(self, images: list[np.ndarray]) -> "PCAKMeansAnomalyDetector":
        """
        Fit scaler, PCA, and k-means on a list of (H, W, C) image arrays.

        We subsample up to 500k pixels to keep fitting tractable on large datasets.
        """
        pixels = np.concatenate(
            [img.reshape(-1, img.shape[-1]) for img in images], axis=0
        )
        # Subsample for tractability
        max_px = 500_000
        if len(pixels) > max_px:
            idx = np.random.default_rng(self.random_state).choice(
                len(pixels), max_px, replace=False
            )
            pixels = pixels[idx]

        if self.scaler is not None:
            pixels = self.scaler.fit_transform(pixels)
        if self.pca is not None:
            pixels = self.pca.fit_transform(pixels)
            n_components = self.pca.n_components_
            var_explained = self.pca.explained_variance_ratio_.sum()
            print(f"PCA: {n_components} components explain {var_explained:.3f} variance")

        self.kmeans.fit(pixels)
        self._fitted = True
        return self

    def score(self, image: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel anomaly score = distance to nearest centroid.

        Parameters
        ----------
        image : np.ndarray  (H, W, C)

        Returns
        -------
        anomaly_map : np.ndarray  (H, W), float32
        """
        assert self._fitted, "Call .fit() before .score()"
        H, W, C = image.shape
        pixels = image.reshape(-1, C).astype(np.float32)
        pixels_pp = self._preprocess(pixels)
        # Euclidean distance to nearest centroid
        distances = self.kmeans.transform(pixels_pp)  # (N, k)
        min_dist = distances.min(axis=1)              # (N,)
        return min_dist.reshape(H, W).astype(np.float32)

    def predict(self, image: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Binary anomaly prediction.

        If threshold is None, uses the 95th percentile of the score map.
        """
        scores = self.score(image)
        t = threshold if threshold is not None else np.percentile(scores, 95)
        return (scores > t).astype(np.uint8)
