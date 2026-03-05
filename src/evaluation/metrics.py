"""
Evaluation metrics for anomaly detection.

Since ground-truth labels are unavailable for most drone imagery in this
project, evaluation is primarily:
  - Distributional: anomaly score statistics (mean, std, percentiles)
  - Qualitative: heatmap overlays on original drone images
  - AUROC: computed when manually annotated samples are available (optional)

AUROC interpretation: 0.5 = random baseline; 1.0 = perfect separation.
For unsupervised anomaly detection without labels, AUROC is estimated from
a small manually inspected subset of images.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def compute_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute AUROC using sklearn.

    Args:
        scores: (N,) float — anomaly scores (higher = more anomalous)
        labels: (N,) int  — ground-truth labels (0=normal, 1=anomaly)
    Returns:
        AUROC in [0, 1].
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for AUROC computation")
    return float(roc_auc_score(labels, scores))


def compute_score_statistics(scores: np.ndarray) -> dict[str, float]:
    """
    Compute descriptive statistics of an anomaly score array.

    Args:
        scores: (N,) float32 anomaly scores.
    Returns:
        dict with keys: mean, std, min, max, p5, p25, p50, p75, p95.
    """
    scores = np.asarray(scores, dtype=np.float32)
    return {
        "mean": float(scores.mean()),
        "std":  float(scores.std()),
        "min":  float(scores.min()),
        "max":  float(scores.max()),
        "p5":   float(np.percentile(scores, 5)),
        "p25":  float(np.percentile(scores, 25)),
        "p50":  float(np.percentile(scores, 50)),
        "p75":  float(np.percentile(scores, 75)),
        "p95":  float(np.percentile(scores, 95)),
    }


def anomaly_threshold_at_fpr(
    scores: np.ndarray,
    fpr_target: float = 0.05,
) -> float:
    """
    Return the score threshold corresponding to an approximate false-positive
    rate of fpr_target, assuming all samples in scores are from the
    training/normal distribution.

    In an unsupervised setting (no labels) this estimates the threshold
    at which fpr_target fraction of normal samples would be flagged.

    Args:
        scores:     (N,) anomaly scores from the training/normal split.
        fpr_target: Target FPR (default 0.05 → flag top 5% of normal patches).
    Returns:
        threshold: float
    """
    return float(np.percentile(scores, (1.0 - fpr_target) * 100))


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    """Serialise a metrics dict to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {output_path}")
