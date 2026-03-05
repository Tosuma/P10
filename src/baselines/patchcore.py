"""
Baseline 4: PatchCore — memory-bank nearest-neighbour anomaly detection.

Uses a pretrained ImageNet ResNet-50 as feature extractor.  Features from
intermediate layers (layer2 + layer3) are averaged and stored in a coreset
memory bank.  Anomaly score = L2 distance to nearest neighbour in the bank.

Input constraint: ResNet-50 expects 3-channel RGB images normalised with
ImageNet statistics.  We use only the first 3 channels of the 10-channel
tensor (R, G, B) and apply standard ImageNet normalisation.

Known limitation — domain gap:
  The RGB-only ResNet-50 features are derived from natural images, not
  aerial multispectral imagery.  This is an intentional design choice for
  this thesis: PatchCore represents the "ImageNet pretrained" baseline,
  and its likely underperformance compared to the MAE + flow approach
  demonstrates the value of (a) domain-specific pretraining and (b)
  multispectral input.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly
Detection", CVPR 2022.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


# ImageNet normalisation constants
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class PatchCore:
    """
    PatchCore anomaly detector using a frozen pretrained ResNet-50.

    Args:
        backbone:               torchvision model name (default "resnet50").
        layer_names:            Which intermediate layers to hook (default layer2+layer3).
        coreset_sampling_ratio: Fraction of features kept in memory bank (default 0.1).
        num_neighbours:         k for k-NN scoring (default 1).
        device:                 Compute device string.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        layer_names: list[str] | None = None,
        coreset_sampling_ratio: float = 0.1,
        num_neighbours: int = 1,
        device: str = "cuda",
    ) -> None:
        self.layer_names            = layer_names or ["layer2", "layer3"]
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbours         = num_neighbours
        self.device                 = torch.device(device if torch.cuda.is_available() else "cpu")

        # Build frozen backbone with intermediate hooks
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        net.eval().to(self.device)
        for p in net.parameters():
            p.requires_grad_(False)

        self._hooks:    list[torch.utils.hooks.RemovableHook] = []
        self._features: dict[str, torch.Tensor] = {}

        for name, module in net.named_children():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

        self.backbone = net
        self.memory_bank: torch.Tensor | None = None

    def _make_hook(self, name: str):
        def _hook(module, inp, out):
            # Adaptive avg pool to a fixed spatial size to handle any input resolution
            self._features[name] = nn.functional.adaptive_avg_pool2d(out, (1, 1)).squeeze(-1).squeeze(-1)
        return _hook

    @staticmethod
    def _imagenet_normalize(rgb_01: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float32 in [0,1] → ImageNet normalised."""
        mean = torch.tensor(_IMAGENET_MEAN, device=rgb_01.device).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  device=rgb_01.device).view(1, 3, 1, 1)
        return (rgb_01 - mean) / std

    def _extract_features(self, loader: DataLoader) -> tuple[torch.Tensor, list[str]]:
        """
        Run RGB channels through ResNet-50, concatenate layer2 + layer3 features.

        Returns:
            (N, D) feature tensor on CPU, list of patch stems.
        """
        all_feats, all_stems = [], []
        with torch.no_grad():
            for batch in loader:
                imgs  = batch["image"].to(self.device)   # (B, 10, H, W)
                rgb   = imgs[:, :3]                       # (B, 3, H, W) — R, G, B channels
                rgb_n = self._imagenet_normalize(rgb)
                self._features.clear()
                self.backbone(rgb_n)
                # Concatenate multi-scale features
                feat = torch.cat([self._features[k] for k in self.layer_names], dim=-1)
                all_feats.append(feat.cpu())
                all_stems.extend(batch["stem"])
        return torch.cat(all_feats, dim=0), all_stems

    def _greedy_coreset(self, features: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Greedy coreset subsampling: iteratively select the feature that is
        furthest from the already-selected set.  Reduces memory bank size
        while retaining coverage of the training distribution.

        Args:
            features: (N, D) float32
            ratio:    Fraction to keep
        Returns:
            (int(N*ratio), D) coreset tensor
        """
        n_keep = max(1, int(features.shape[0] * ratio))
        if n_keep >= features.shape[0]:
            return features

        # Initialise with a random point
        selected = [torch.randint(features.shape[0], (1,)).item()]
        dists    = torch.full((features.shape[0],), float("inf"))

        for _ in range(n_keep - 1):
            last  = features[selected[-1]].unsqueeze(0)          # (1, D)
            d     = torch.norm(features - last, dim=-1)           # (N,)
            dists = torch.minimum(dists, d)
            selected.append(int(dists.argmax().item()))

        return features[torch.tensor(selected)]

    def fit(self, loader: DataLoader) -> "PatchCore":
        """
        Build memory bank from training set features.

        Returns:
            self
        """
        print("[PatchCore] Extracting ResNet-50 training features …")
        feats, _ = self._extract_features(loader)
        print(f"  {feats.shape[0]:,} features, coreset sampling {self.coreset_sampling_ratio}")
        self.memory_bank = self._greedy_coreset(feats, self.coreset_sampling_ratio)
        print(f"  Memory bank: {self.memory_bank.shape[0]:,} vectors")
        return self

    def score(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute patch-level anomaly scores via k-NN distance to memory bank.

        Returns:
            scores: (N,) float32 — distance to nearest neighbour
            stems:  list[str]
        """
        assert self.memory_bank is not None, "Call fit() first"
        feats, stems = self._extract_features(loader)

        # Chunked nearest-neighbour search to avoid OOM
        bank = self.memory_bank.to(self.device)
        scores = []
        chunk_size = 512
        for start in range(0, feats.shape[0], chunk_size):
            chunk = feats[start: start + chunk_size].to(self.device)
            dists = torch.cdist(chunk, bank)              # (chunk, bank_size)
            knn   = dists.topk(self.num_neighbours, dim=-1, largest=False).values
            scores.append(knn.mean(dim=-1).cpu())

        return torch.cat(scores).numpy().astype(np.float32), stems

    def save(self, path: Path) -> None:
        # Remove hooks before pickling (not serialisable)
        for h in self._hooks:
            h.remove()
        self._hooks = []
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "PatchCore":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
