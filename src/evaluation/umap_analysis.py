"""
UMAP dimensionality reduction for encoder feature analysis.

Reduces high-dimensional MAE encoder features to 2D for qualitative
thesis visualisation.  Useful for:
  - Verifying the encoder has learned semantically meaningful structure
    (healthy / stressed vegetation should cluster differently).
  - Identifying clusters of anomalous patches in feature space.
  - Demonstrating that the learned representation separates land-cover
    types (rice, weeds, soil, water) without any labels.

Design choice — cosine metric:
  Cosine distance is scale-invariant and better suited to high-dimensional
  ViT feature vectors than Euclidean distance, which suffers from the
  curse of dimensionality in 384-dim space.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


class UMAPAnalysis:
    """
    Wrapper around UMAP for anomaly feature analysis.

    Args:
        n_components: Output dimensionality (default 2).
        n_neighbors:  UMAP neighbourhood size (default 15).
        min_dist:     Minimum distance in low-dim space (default 0.1).
        metric:       Distance metric (default "cosine").
        seed:         Random state for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        seed: int = 42,
    ) -> None:
        if not _UMAP_AVAILABLE:
            raise ImportError("umap-learn is required: pip install umap-learn")
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        self._fitted = False

    def fit_transform(
        self,
        features: np.ndarray,
        max_samples: int = 10_000,
    ) -> np.ndarray:
        """
        Fit UMAP on (possibly subsampled) features and return 2-D embeddings.

        When N > max_samples, a random subsample is used for fitting but all
        points are projected using the fitted reducer.

        Args:
            features:    (N, D) float32 feature array.
            max_samples: Maximum samples to use for UMAP fitting.
        Returns:
            (N, n_components) float32 2-D embeddings.
        """
        N = features.shape[0]
        if N > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(N, max_samples, replace=False)
            self.reducer.fit(features[idx])
            embedding = self.reducer.transform(features)
        else:
            embedding = self.reducer.fit_transform(features)

        self._fitted = True
        self._last_embedding = embedding
        return embedding.astype(np.float32)

    def plot(
        self,
        embeddings: np.ndarray,
        scores: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
        title: str = "UMAP of MAE Encoder Features",
    ) -> None:
        """
        Scatter plot of 2-D embeddings coloured by anomaly score or label.

        If scores provided: viridis colourmap (bright = high anomaly score).
        If labels provided: tab10 colour per class (0=normal, 1=anomaly).
        If neither:         uniform colour.

        Args:
            embeddings:   (N, 2) 2-D coordinates.
            scores:       (N,) float anomaly scores (optional).
            labels:       (N,) int class labels (optional; takes precedence).
            output_path:  Save path (PNG).  If None, the figure is shown
                          (not recommended on HPC — use output_path).
            title:        Plot title.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        x, y = embeddings[:, 0], embeddings[:, 1]

        if labels is not None:
            unique = np.unique(labels)
            cmap = plt.cm.tab10
            for i, lbl in enumerate(unique):
                mask = labels == lbl
                ax.scatter(x[mask], y[mask], s=2, alpha=0.5,
                           color=cmap(i / max(len(unique), 1)),
                           label=f"Class {lbl}")
            ax.legend(markerscale=5, fontsize=10)
        elif scores is not None:
            sc = ax.scatter(x, y, c=scores, cmap="viridis", s=2, alpha=0.5)
            plt.colorbar(sc, ax=ax, label="Anomaly Score (NLL)")
        else:
            ax.scatter(x, y, s=2, alpha=0.4, color="steelblue")

        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.tight_layout()

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"UMAP plot saved → {path}")
        else:
            plt.show()

    def save(self, path: Path) -> None:
        """Serialise the fitted UMAP reducer to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "UMAPAnalysis":
        with open(path, "rb") as f:
            return pickle.load(f)
