"""
Segmentation metrics for the experiment.

Primary metrics:
  - mIoU (mean Intersection over Union)
  - Per-class IoU

These are the standard metrics for semantic segmentation and what
reviewers will expect in your thesis.
"""

import torch
import numpy as np


class SegmentationMetrics:
    """
    Accumulates confusion matrix across batches and computes IoU metrics.
    
    Args:
        num_classes: Number of segmentation classes.
        class_names: Optional list of class names for reporting.
    """

    def __init__(self, num_classes: int, class_names: list = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def reset(self):
        """Reset accumulated confusion matrix."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with a batch of predictions and targets.
        
        Args:
            preds: Predicted class labels (B, H, W) — int/long tensor
            targets: Ground truth labels (B, H, W) — int/long tensor
        """
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Filter out any ignore labels (e.g., 255)
        valid = targets < self.num_classes
        preds = preds[valid]
        targets = targets[valid]

        # Update confusion matrix (vectorized)
        indices = targets * self.num_classes + preds
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """
        Compute mIoU and per-class IoU from accumulated confusion matrix.
        
        IoU for class i = TP_i / (TP_i + FP_i + FN_i)
            TP_i = confusion_matrix[i, i]
            FP_i = sum(confusion_matrix[:, i]) - TP_i
            FN_i = sum(confusion_matrix[i, :]) - TP_i
        
        Returns:
            dict with 'miou' and 'per_class_iou'
        """
        per_class_iou = {}
        ious = []

        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = tp + fp + fn
            if denominator == 0:
                # Class not present in this evaluation — skip for mIoU
                iou = float("nan")
            else:
                iou = tp / denominator

            per_class_iou[self.class_names[i]] = iou
            if not np.isnan(iou):
                ious.append(iou)

        miou = np.mean(ious) if ious else 0.0

        return {
            "miou": float(miou),
            "per_class_iou": per_class_iou,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
