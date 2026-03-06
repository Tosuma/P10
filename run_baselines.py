"""
Run all baseline anomaly detection methods.

Baselines:
  1. NDVI Thresholding        — classical spectral threshold
  2. PCA + k-means            — unsupervised spectral clustering
  3. Convolutional Autoencoder— reconstruction-based anomaly detection
  4. PatchCore                — memory-bank nearest-neighbour (RGB only)

Results are saved to output_dir/. Each method writes:
  - val_scores.npy — anomaly scores for the validation split
  - stats.json     — summary statistics

Usage:
    python run_baselines.py
    python run_baselines.py output_dir=results/baselines data.patch_dir=/path/to/patches
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.data.patch_dataset import WeedyRicePatchDataset
from src.data.dataloader import build_dataloaders
from src.baselines.ndvi_threshold import NDVIThreshold
from src.baselines.pca_kmeans import PCAKMeans
from src.baselines.conv_autoencoder import ConvAutoencoder, ConvAETrainer
from src.baselines.patchcore import PatchCore
from src.evaluation.metrics import compute_score_statistics, save_metrics_json


@hydra.main(config_path="configs", config_name="baselines", version_base="1.3")
def main(cfg: DictConfig) -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        cfg.data.patch_dir = data_root

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir  = Path(cfg.data.patch_dir)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset split info
    train_ds = WeedyRicePatchDataset(patch_dir, split="train",
                                     val_fraction=cfg.data.val_fraction,
                                     seed=cfg.data.seed, augment=False)
    val_ds   = WeedyRicePatchDataset(patch_dir, split="val",
                                     val_fraction=cfg.data.val_fraction,
                                     seed=cfg.data.seed, augment=False)
    train_stems = train_ds.stems
    val_stems   = val_ds.stems

    train_loader, val_loader = build_dataloaders(cfg)
    all_results: dict[str, dict] = {}

    # ── 1. NDVI Threshold ─────────────────────────────────────────────
    print("\n=== Baseline 1: NDVI Threshold ===")
    thresh_cfg = cfg.ndvi_threshold
    if str(thresh_cfg.threshold) == "auto":
        detector = NDVIThreshold.from_data(patch_dir, train_stems)
    else:
        detector = NDVIThreshold(threshold=float(thresh_cfg.threshold))

    stats = detector.run_on_patch_dir(patch_dir, val_stems, output_dir)
    all_results["ndvi_threshold"] = stats
    print(f"  {stats}")

    # ── 2. PCA + k-means ──────────────────────────────────────────────
    print("\n=== Baseline 2: PCA + k-means ===")
    pca_cfg = cfg.pca_kmeans
    pca_km  = PCAKMeans(n_components=pca_cfg.n_components,
                        n_clusters=pca_cfg.n_clusters, seed=pca_cfg.seed)
    stats = pca_km.run(output_dir, train_loader, val_loader)
    all_results["pca_kmeans"] = stats
    print(f"  {stats}")

    # ── 3. Conv Autoencoder ───────────────────────────────────────────
    print("\n=== Baseline 3: Conv Autoencoder ===")

    ae_cfg   = cfg.conv_ae
    ae_model = ConvAutoencoder(in_chans=cfg.data.in_chans)
    ae_trainer = ConvAETrainer(
        ae_model, train_loader, val_loader, device,
        lr=ae_cfg.lr, epochs=ae_cfg.epochs, patience=ae_cfg.patience,
        output_dir=Path(ae_cfg.output_dir),
    )
    ae_trainer.train()

    ae_scores, ae_stems = ae_trainer.score_dataset(val_loader)
    ae_out = output_dir / "conv_ae"
    ae_out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(ae_out / "val_scores.npy", ae_scores)
    stats = compute_score_statistics(ae_scores)
    save_metrics_json(stats, ae_out / "stats.json")
    all_results["conv_ae"] = stats
    print(f"  {stats}")

    # ── 4. PatchCore ──────────────────────────────────────────────────
    print("\n=== Baseline 4: PatchCore ===")
    pc_cfg    = cfg.patchcore
    patchcore = PatchCore(
        backbone=pc_cfg.backbone,
        layer_names=list(pc_cfg.layer_names),
        coreset_sampling_ratio=pc_cfg.coreset_sampling_ratio,
        num_neighbours=pc_cfg.num_neighbours,
        device=str(device),
    )
    patchcore.fit(train_loader)
    pc_scores, pc_stems = patchcore.score(val_loader)

    pc_out = output_dir / "patchcore"
    pc_out.mkdir(parents=True, exist_ok=True)
    np.save(pc_out / "val_scores.npy", pc_scores)
    patchcore.save(pc_out / "model.pkl")
    stats = compute_score_statistics(pc_scores)
    save_metrics_json(stats, pc_out / "stats.json")
    all_results["patchcore"] = stats
    print(f"  {stats}")

    # ── Summary ───────────────────────────────────────────────────────
    save_metrics_json(all_results, output_dir / "all_baselines_summary.json")
    print(f"\nAll baseline results saved to {output_dir}")


if __name__ == "__main__":
    main()
