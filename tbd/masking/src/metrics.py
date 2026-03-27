from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConfusionTotals:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.bool()
        targets = targets.bool()
        self.tp += torch.logical_and(preds, targets).sum().item()
        self.fp += torch.logical_and(preds, torch.logical_not(targets)).sum().item()
        self.fn += torch.logical_and(torch.logical_not(preds), targets).sum().item()
        self.tn += torch.logical_and(torch.logical_not(preds), torch.logical_not(targets)).sum().item()

    def compute(self) -> dict[str, float]:
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + 1e-8)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + 1e-8)
        specificity = self.tn / (self.tn + self.fp + 1e-8)
        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "specificity": specificity,
        }


def threshold_predictions(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    return (probabilities >= threshold).float()
