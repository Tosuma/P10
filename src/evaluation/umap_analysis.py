"""
UMAP-based cluster analysis of MAE encoder features.

Purpose
-------
UMAP provides a 2-D embedding of the high-dimensional patch feature space.
This serves two roles in the thesis:
  1. Qualitative validation: do normal and anomalous patches form distinct
     clusters in feature space?  If yes, the MAE encoder has learned a useful
     representation.
  2. Interpretability: what does the flow model's density boundary look like
     in 2-D?  We can overlay anomaly scores as a colour dimension.

We use UMAP rather than t-SNE because:
  - UMAP preserves global structure better (not just local neighbourhood).
  - It is faster on large datasets (N > 50k).
  - The embedding is more reproducible across runs with a fixed seed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def compute_umap_embedding(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
    max_samples: int = 50_000,
) -> np.ndarray:
    """
    Fit a UMAP embedding on high-dimensional features.

    Parameters
    ----------
    features : (N, D) float32
    max_samples : int
        Sub-sample if N > max_samples for tractability.

    Returns
    -------
    embedding : (N', 2) float32  (N' ≤ N due to sub-sampling)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    if len(features) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), max_samples, replace=False)
        features = features[idx]

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        verbose=True,
    )
    return reducer.fit_transform(features).astype(np.float32)


def plot_umap_embedding(
    embedding: np.ndarray,
    scores: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "UMAP of Patch Features",
    cmap: str = "RdYlGn_r",
    point_size: float = 1.5,
    alpha: float = 0.5,
) -> plt.Figure:
    """
    Scatter-plot of the UMAP embedding, coloured by anomaly score or label.

    Parameters
    ----------
    embedding : (N, 2)
    scores : (N,) float — anomaly scores for colour mapping (overrides labels)
    labels : (N,) int — binary labels (0=normal, 1=anomalous)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    if scores is not None:
        norm = Normalize(vmin=np.percentile(scores, 2), vmax=np.percentile(scores, 98))
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=scores,
            cmap=cmap,
            norm=norm,
            s=point_size,
            alpha=alpha,
            rasterized=True,
        )
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Anomaly Score")
    elif labels is not None:
        colours = np.where(labels == 1, "red", "steelblue")
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colours,
            s=point_size,
            alpha=alpha,
            rasterized=True,
        )
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="steelblue", label="Normal"),
            Patch(color="red", label="Anomalous"),
        ], fontsize=9)
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            s=point_size, alpha=alpha, c="steelblue", rasterized=True
        )

    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(title, fontsize=12)
    return fig
