"""
Visualisation utilities for anomaly detection results.

All plotting functions:
  - Accept numpy arrays / torch tensors as input.
  - Save to disk (no GUI display required — important for HPC batch jobs).
  - Use matplotlib with the Agg backend (no display server needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for HPC / headless environments
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_score_histogram(
    scores: np.ndarray,
    method_name: str,
    output_path: Path,
    threshold: Optional[float] = None,
    bins: int = 100,
) -> None:
    """
    Plot and save a histogram of anomaly scores.

    Args:
        scores:       (N,) float anomaly scores.
        method_name:  Label used in title and legend.
        output_path:  File path for the saved PNG.
        threshold:    If given, draw a vertical dashed line.
        bins:         Number of histogram bins.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=bins, density=True, alpha=0.7, color="steelblue",
            label=method_name)
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--",
                   label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Anomaly Score Distribution — {method_name}")
    ax.legend()
    save_figure(fig, Path(output_path))


def plot_score_comparison(
    scores_dict: dict[str, np.ndarray],
    output_path: Path,
    bins: int = 100,
) -> None:
    """
    Side-by-side histograms for multiple anomaly detection methods.

    Args:
        scores_dict: {method_name: (N,) float} mapping.
        output_path: Output PNG path.
        bins:        Number of histogram bins.
    """
    n = len(scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colours = plt.cm.tab10(np.linspace(0, 1, n))

    for ax, (name, scores), colour in zip(axes, scores_dict.items(), colours):
        ax.hist(scores, bins=bins, density=True, alpha=0.75, color=colour)
        ax.set_title(name)
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")

    fig.suptitle("Anomaly Score Distributions — Method Comparison", y=1.02)
    fig.tight_layout()
    save_figure(fig, Path(output_path))


def plot_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    masks: torch.Tensor,
    output_path: Path,
    n_samples: int = 8,
    channel_indices: tuple[int, ...] = (0, 1, 2),
) -> None:
    """
    Visualise original / masked / reconstructed patches side by side.

    Shows three rows per sample: original, masked (MAE input), reconstruction.
    Only the channels specified by channel_indices are displayed (default: RGB).

    Args:
        originals:       (N, C, H, W) original patch tensors.
        reconstructions: (N, C, H, W) reconstructed patch tensors.
        masks:           (N, num_patches) bool mask from MAE (True = masked).
        output_path:     Output PNG path.
        n_samples:       Number of patches to display.
        channel_indices: Channel indices to use for RGB visualisation.
    """
    n = min(n_samples, originals.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 8))

    def _to_img(t: torch.Tensor, chs: tuple) -> np.ndarray:
        img = t[list(chs)].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        return img

    for i in range(n):
        orig = _to_img(originals[i], channel_indices)
        recon = _to_img(reconstructions[i], channel_indices)

        axes[0, i].imshow(orig)
        axes[0, i].set_title("Original", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(orig * 0.3)    # darkened = masked approximation
        axes[1, i].set_title("(Masked)", fontsize=8)
        axes[1, i].axis("off")

        axes[2, i].imshow(recon)
        axes[2, i].set_title("Reconstruction", fontsize=8)
        axes[2, i].axis("off")

    fig.suptitle("MAE Reconstructions", y=1.01)
    fig.tight_layout()
    save_figure(fig, Path(output_path))


def plot_heatmap_grid(
    rgb_images: list[np.ndarray],
    heatmaps: list[np.ndarray],
    titles: list[str],
    output_path: Path,
) -> None:
    """
    Grid of (RGB, heatmap overlay) image pairs.

    Args:
        rgb_images: list of (H, W, 3) uint8 RGB images.
        heatmaps:   list of (H, W, 3) uint8 BGR overlay images (from cv2).
        titles:     list of title strings.
        output_path: Output PNG path.
    """
    import cv2
    n = len(rgb_images)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(rgb_images[i])
        axes[0, i].set_title(titles[i], fontsize=9)
        axes[0, i].axis("off")

        # cv2 returns BGR — convert to RGB for matplotlib
        overlay_rgb = cv2.cvtColor(heatmaps[i], cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(overlay_rgb)
        axes[1, i].set_title("Anomaly Heatmap", fontsize=9)
        axes[1, i].axis("off")

    fig.tight_layout()
    save_figure(fig, Path(output_path))
