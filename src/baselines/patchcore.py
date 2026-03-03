"""
Baseline 4: PatchCore anomaly detection.

Reference: Roth et al. 2022 "Towards Total Recall in Industrial Anomaly Detection"

Approach
--------
1. Extract multi-scale intermediate features from a pretrained ResNet50 encoder
   (ImageNet weights) for each training patch.
2. Sub-sample the feature memory bank using coreset greedy selection to reduce
   memory and inference time.
3. Anomaly score = nearest-neighbour distance from the query feature to the
   memory bank.

Adaptation notes for agricultural drone imagery
  - The original PatchCore uses ImageNet-pretrained features.  These are
    trained on natural images, not aerial/agricultural scenes.  This creates a
    domain gap that we expect to be visible in the results.
  - We use ResNet50 layers 2 + 3 (as in the original paper) because they
    capture mid-level textures (appropriate for crop canopy).
  - PatchCore operates on patch-level features from a full image, not on
    pre-extracted dataset patches — we adapt it to work on our patch dataset
    by treating each patch as an "image".

Role in thesis
--------------
PatchCore represents the category of "memory-bank + NN retrieval" AD methods.
Its use of ImageNet features (no training on our data) makes it an important
ablation: how much does domain-specific pretraining (MAE) help?
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


class PatchCoreDetector:
    """
    PatchCore anomaly detector with ResNet50 backbone.

    Parameters
    ----------
    backbone : str
        'resnet50' (only option currently, as per original paper).
    layers : list of str
        ResNet layer names for feature extraction.
    input_size : int
        Spatial size of input patches (must be >= 224 for ImageNet pretrained).
    coreset_ratio : float
        Fraction of features to keep in the memory bank (greedy coreset).
    device : torch.device
    """

    def __init__(
        self,
        layers: list[str] | None = None,
        input_size: int = 224,
        coreset_ratio: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        self.layers = layers or ["layer2", "layer3"]
        self.input_size = input_size
        self.coreset_ratio = coreset_ratio
        self.device = device or torch.device("cpu")

        # Load pretrained ResNet50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = create_feature_extractor(
            backbone, return_nodes={l: l for l in self.layers}
        ).to(self.device)
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # Adaptive average pool to align multi-scale features to common spatial size
        # We pool layer2 (28×28 for 224 input) and layer3 (14×14) both to 14×14
        self.avg_pool = nn.AdaptiveAvgPool2d((14, 14))

        self.memory_bank: Optional[np.ndarray] = None  # (M, D)

    # ── Feature extraction ────────────────────────────────────────────────────

    def _preprocess_for_resnet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapt our multi-channel patches to 3-channel ResNet input.

        Strategy: use the first 3 channels (RGB) if available, otherwise take
        a mean of all channels into 3 (rough spectral compression).
        This is a known limitation of ImageNet-pretrained backbones on
        multispectral data — hence why our MAE approach is expected to be better.
        """
        if x.shape[1] >= 3:
            x = x[:, :3, :, :]  # Use RGB only
        else:
            # Repeat to get 3 channels
            x = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)

        # Resize to expected input size
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        # ImageNet normalisation (on [0,1] input)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract and aggregate multi-scale features.

        Returns
        -------
        feats : (B, D)  — one feature vector per image/patch
        """
        images = self._preprocess_for_resnet(images.to(self.device))
        out = self.feature_extractor(images)

        # Pool all feature maps to the same spatial size and concatenate
        pooled = [self.avg_pool(out[l]) for l in self.layers]  # each (B, C_l, 14, 14)
        # Global average pool over spatial dims → (B, sum(C_l))
        combined = torch.cat(
            [p.mean(dim=(-2, -1)) for p in pooled], dim=1
        )
        return combined.cpu().float()

    # ── Coreset selection ─────────────────────────────────────────────────────

    @staticmethod
    def _greedy_coreset(
        features: np.ndarray,
        target_size: int,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Greedy coreset sub-sampling using iterative farthest-point selection.

        Selects `target_size` points from `features` such that the selected
        points maximally cover the full feature space.

        Complexity: O(N × target_size) — tractable for N ≤ 500k, D ≤ 2048.
        """
        rng = np.random.default_rng(seed)
        N = len(features)
        if target_size >= N:
            return features

        # Start with a random point
        selected_idx = [int(rng.integers(N))]
        # Distance of each point to the nearest selected point
        dists = np.full(N, np.inf)

        for _ in tqdm(range(1, target_size), desc="Coreset selection"):
            last = features[selected_idx[-1]]
            new_dists = np.linalg.norm(features - last, axis=1)
            dists = np.minimum(dists, new_dists)
            selected_idx.append(int(np.argmax(dists)))

        return features[selected_idx]

    # ── Fit / Score interface ─────────────────────────────────────────────────

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        seed: int = 42,
    ) -> "PatchCoreDetector":
        """
        Build memory bank from training data.

        Parameters
        ----------
        train_loader : DataLoader yielding batches with key 'image'
        """
        all_feats: list[np.ndarray] = []
        for batch in tqdm(train_loader, desc="PatchCore: extracting training features"):
            images = batch["image"]
            feats = self._extract_features(images).numpy()
            all_feats.append(feats)

        all_feats_np = np.concatenate(all_feats, axis=0)  # (N_total, D)

        target_size = max(1, int(len(all_feats_np) * self.coreset_ratio))
        print(f"PatchCore: memory bank {len(all_feats_np)} → {target_size} (coreset ratio={self.coreset_ratio})")
        self.memory_bank = self._greedy_coreset(all_feats_np, target_size, seed)
        return self

    def score_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Compute per-image anomaly scores.

        Parameters
        ----------
        images : (B, C, H, W)

        Returns
        -------
        scores : np.ndarray  (B,), float32
            Distance to nearest memory bank vector.
        """
        assert self.memory_bank is not None, "Call .fit() first"
        query_feats = self._extract_features(images).numpy()  # (B, D)

        # Nearest-neighbour search (brute force; use faiss for large memory banks)
        diffs = query_feats[:, None, :] - self.memory_bank[None, :, :]  # (B, M, D)
        dists = np.linalg.norm(diffs, axis=-1)  # (B, M)
        return dists.min(axis=1).astype(np.float32)  # (B,)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"memory_bank": self.memory_bank}, f)

    def load(self, path: str) -> "PatchCoreDetector":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.memory_bank = state["memory_bank"]
        return self
