"""
Visualisation utilities: overlay heatmaps on RGB images, generate distribution
plots, and produce thesis-quality figures.

All plotting functions accept optional `ax` arguments so they can be embedded
in larger figure layouts (e.g. multi-panel thesis figures).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def overlay_heatmap_on_rgb(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "RdYlGn_r",  # Red=high anomaly, Green=low anomaly (intuitive)
    ax: Optional[plt.Axes] = None,
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Blend a normalised anomaly heatmap over the RGB image.

    Parameters
    ----------
    rgb : (H, W, 3) float32 in [0, 1]
    heatmap : (H, W) float32 (normalised anomaly scores)
    alpha : float
        Heatmap opacity (0=transparent, 1=opaque).
    cmap : str
        Matplotlib colormap.  RdYlGn_r is reversed so red=anomalous.
    vmin, vmax : float
        Colormap range.  Defaults to heatmap min/max.

    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.get_figure()

    ax.imshow(np.clip(rgb, 0, 1))
    norm = Normalize(
        vmin=vmin if vmin is not None else np.nanmin(heatmap),
        vmax=vmax if vmax is not None else np.nanmax(heatmap),
    )
    hm_rgba = cm.get_cmap(cmap)(norm(heatmap))
    hm_rgba[..., 3] = np.where(np.isnan(heatmap), 0, alpha)
    ax.imshow(hm_rgba)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Anomaly Score")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    return fig


def plot_score_distribution(
    scores_dict: dict[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
    title: str = "Anomaly Score Distribution",
    xlabel: str = "Anomaly Score",
) -> plt.Figure:
    """
    Histogram of anomaly score distributions for one or multiple methods.

    Parameters
    ----------
    scores_dict : dict mapping method_name → 1-D score array
        Overlay multiple methods for qualitative comparison.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    for name, scores in scores_dict.items():
        ax.hist(
            scores.ravel(),
            bins=100,
            density=True,
            alpha=0.6,
            label=name,
        )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    return fig


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    ax: Optional[plt.Axes] = None,
    title: str = "ROC Curves",
) -> plt.Figure:
    """
    Plot ROC curves for multiple methods.

    Parameters
    ----------
    results : dict mapping method_name → (fpr, tpr) arrays
    """
    from sklearn.metrics import auc

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    for name, (fpr, tpr) in results.items():
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR", fontsize=11)
    ax.set_ylabel("TPR", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    return fig


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 200) -> None:
    """Save figure with tight layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {path}")
