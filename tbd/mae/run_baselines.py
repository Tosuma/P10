"""
Entry point: Train and evaluate all baselines.

Runs all 4 baselines sequentially, saves anomaly score arrays and heatmaps
for each, and prints a summary comparison table.

Usage:
    python run_baselines.py
    python run_baselines.py baselines.output_dir=outputs/baselines
"""

import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.dataloader import build_dataloaders, build_inference_loader
from src.baselines import (
    NDVIThreshold,
    PCAKMeansAnomalyDetector,
    ConvAutoencoder,
    CAETrainer,
    PatchCoreDetector,
)
from src.evaluation.heatmap import assemble_heatmap, smooth_heatmap, normalise_heatmap, export_geotiff
from src.evaluation.metrics import score_statistics, estimate_anomaly_fraction
from src.evaluation.visualize import plot_score_distribution, save_figure

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="baselines", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.baselines.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        rgb_dir=cfg.data.rgb_dir if cfg.data.include_rgb else None,
        ms_dir=cfg.data.ms_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_fraction=cfg.data.val_fraction,
        patch_size=cfg.data.patch_size,
        patches_per_image=cfg.data.patches_per_image,
        include_rgb=cfg.data.include_rgb,
        include_indices=cfg.data.include_indices,
        seed=cfg.data.seed,
        ms_suffixes=list(cfg.data.ms_suffixes),
    )

    infer_loader, infer_dataset = build_inference_loader(
        rgb_dir=cfg.data.rgb_dir if cfg.data.include_rgb else None,
        ms_dir=cfg.data.ms_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        patch_size=cfg.data.patch_size,
        stride=cfg.data.stride,
        include_rgb=cfg.data.include_rgb,
        include_indices=cfg.data.include_indices,
        ms_suffixes=list(cfg.data.ms_suffixes),
    )

    all_scores: dict[str, np.ndarray] = {}

    # ── Baseline 1: NDVI Threshold ────────────────────────────────────────────
    log.info("=== Baseline 1: NDVI Threshold ===")
    ndvi_model = NDVIThreshold(
        threshold=cfg.baselines.ndvi.threshold,
        nir_channel=cfg.baselines.ndvi.nir_channel,
        red_channel=cfg.baselines.ndvi.red_channel,
    )
    ndvi_scores = []
    for batch in infer_loader:
        imgs = batch["image"].numpy().transpose(0, 2, 3, 1)  # (B, H, W, C)
        for img in imgs:
            s = ndvi_model.score(img)
            ndvi_scores.append(s.ravel())
    all_scores["NDVI Threshold"] = np.concatenate(ndvi_scores)
    stats = score_statistics(all_scores["NDVI Threshold"])
    log.info(f"NDVI stats: {stats}")

    # ── Baseline 2: PCA + k-means ─────────────────────────────────────────────
    log.info("=== Baseline 2: PCA + k-means ===")
    train_images = []
    for batch in train_loader:
        imgs = batch["image"].numpy().transpose(0, 2, 3, 1)
        train_images.extend(list(imgs))

    pca_km = PCAKMeansAnomalyDetector(
        n_clusters=cfg.baselines.pca_kmeans.n_clusters,
        pca_variance=cfg.baselines.pca_kmeans.pca_variance,
        scale=cfg.baselines.pca_kmeans.scale,
        random_state=cfg.baselines.pca_kmeans.random_state,
    )
    pca_km.fit(train_images)

    pca_scores = []
    for batch in infer_loader:
        imgs = batch["image"].numpy().transpose(0, 2, 3, 1)
        for img in imgs:
            s = pca_km.score(img)
            pca_scores.append(s.ravel())
    all_scores["PCA + k-means"] = np.concatenate(pca_scores)
    log.info(f"PCA+kmeans stats: {score_statistics(all_scores['PCA + k-means'])}")

    # ── Baseline 3: Convolutional Autoencoder ─────────────────────────────────
    log.info("=== Baseline 3: Convolutional Autoencoder ===")
    cae = ConvAutoencoder(
        in_channels=cfg.data.in_channels,
        base_channels=cfg.baselines.cae.base_channels,
        latent_dim=cfg.baselines.cae.latent_dim,
    )
    cae_trainer = CAETrainer(
        model=cae,
        device=device,
        lr=cfg.baselines.cae.lr,
        epochs=cfg.baselines.cae.epochs,
        output_dir=cfg.baselines.cae.output_dir,
    )
    cae_trainer.train(train_loader, val_loader)

    cae_scores = []
    for batch in infer_loader:
        imgs = batch["image"].to(device)
        score_maps = cae.anomaly_score(imgs)  # (B, H, W)
        cae_scores.append(score_maps.cpu().numpy().ravel())
    all_scores["Conv Autoencoder"] = np.concatenate(cae_scores)
    log.info(f"CAE stats: {score_statistics(all_scores['Conv Autoencoder'])}")

    # ── Baseline 4: PatchCore ─────────────────────────────────────────────────
    log.info("=== Baseline 4: PatchCore ===")
    pc = PatchCoreDetector(
        layers=list(cfg.baselines.patchcore.layers),
        input_size=cfg.baselines.patchcore.input_size,
        coreset_ratio=cfg.baselines.patchcore.coreset_ratio,
        device=device,
    )
    pc.fit(train_loader)
    pc.save(cfg.baselines.patchcore.memory_bank_path)

    pc_scores = []
    for batch in infer_loader:
        imgs = batch["image"]
        scores = pc.score_batch(imgs)
        pc_scores.append(scores)
    all_scores["PatchCore"] = np.concatenate(pc_scores)
    log.info(f"PatchCore stats: {score_statistics(all_scores['PatchCore'])}")

    # ── Score distribution comparison plot ───────────────────────────────────
    fig = plot_score_distribution(all_scores, title="Baseline Anomaly Score Distributions")
    save_figure(fig, out_dir / "baseline_score_distributions.png")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY — Anomaly fraction (z > 2.5σ)")
    print("=" * 60)
    for name, scores in all_scores.items():
        frac = estimate_anomaly_fraction(scores)
        print(f"  {name:<30s}  {frac * 100:.2f}%")
    print("=" * 60)

    # Save score arrays for later AUROC computation if labels become available
    np.savez(str(out_dir / "baseline_scores.npz"), **{
        k.replace(" ", "_"): v for k, v in all_scores.items()
    })
    log.info(f"Scores saved to {out_dir / 'baseline_scores.npz'}")


if __name__ == "__main__":
    main()
