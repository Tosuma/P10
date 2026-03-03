"""
Evaluation metrics for the anomaly detection pipeline.

Since our dataset is unlabelled, most evaluation is qualitative or relies on
self-supervised proxies.  The metrics here support:
  (a) Quantitative evaluation if any ground truth labels become available
      (AUROC, AUPRC, optimal threshold F1).
  (b) Distribution analysis: anomaly score histograms, score statistics.
  (c) Score calibration: normalisation, outlier fraction estimation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Area Under the ROC Curve.

    Parameters
    ----------
    scores : (N,) float — higher = more anomalous
    labels : (N,) int  — 1 = anomalous, 0 = normal

    Returns
    -------
    auroc : float
    """
    return float(roc_auc_score(labels, scores))


def compute_auprc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Area Under Precision-Recall Curve."""
    return float(average_precision_score(labels, scores))


def optimal_threshold_f1(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Find the threshold that maximises F1, and return (threshold, F1).

    Useful for converting continuous anomaly scores to binary predictions
    when a small labelled validation set is available.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    precision = tpr / np.maximum(tpr + fpr, 1e-8)
    recall = tpr
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    best_idx = np.argmax(f1)
    return float(thresholds[best_idx]), float(f1[best_idx])


def score_statistics(scores: np.ndarray) -> dict:
    """
    Compute descriptive statistics of anomaly score distribution.

    Useful for tracking training progress (the distribution should tighten
    as the flow model converges) and for setting alert thresholds.

    Returns
    -------
    dict with keys: mean, std, median, p95, p99, skewness
    """
    from scipy.stats import skew
    return {
        "mean":     float(scores.mean()),
        "std":      float(scores.std()),
        "median":   float(np.median(scores)),
        "p95":      float(np.percentile(scores, 95)),
        "p99":      float(np.percentile(scores, 99)),
        "skewness": float(skew(scores.ravel())),
    }


def estimate_anomaly_fraction(
    scores: np.ndarray,
    z_threshold: float = 2.5,
) -> float:
    """
    Estimate the fraction of anomalous patches using z-score thresholding.

    This is the primary unsupervised evaluation metric when no labels exist.
    A z_threshold of 2.5 corresponds to ~1.2% false positive rate under a
    Gaussian, which is aggressive enough to flag real anomalies in fields
    while avoiding excessive noise flags.
    """
    z = (scores - scores.mean()) / max(scores.std(), 1e-6)
    return float((z > z_threshold).mean())
