"""
Evaluation and visualisation entry point.

Generates:
  - Anomaly score distribution histograms (MAE+Flow vs baselines)
  - UMAP projection of encoder features coloured by anomaly score
  - Per-image anomaly heatmaps (PNG + optionally GeoTIFF)
  - MAE reconstruction visualisations

Usage:
    python evaluate.py
    python evaluate.py mae_checkpoint=checkpoints/mae/best.pth \\
                       flow_checkpoint=checkpoints/flow/best.pth
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.dataloader import build_dataloaders
from src.data.patch_dataset import WeedyRicePatchDataset, _image_stem_from_patch
from src.models.mae import build_mae
from src.models.flow_model import FlowModel
from src.training.flow_trainer import FlowTrainer
from src.evaluation.metrics import compute_score_statistics, save_metrics_json
from src.evaluation.heatmap import AnomalyHeatmap
from src.evaluation.visualize import (
    plot_score_histogram, plot_score_comparison, plot_reconstructions,
)
from src.evaluation.umap_analysis import UMAPAnalysis


@hydra.main(config_path="configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        cfg.data.patch_dir = data_root

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load MAE ──────────────────────────────────────────────────────
    mae = build_mae(in_chans=cfg.data.in_chans, img_size=cfg.data.patch_size,
                    use_checkpoint=False).to(device)
    ckpt = torch.load(cfg.mae_checkpoint, map_location=device)
    mae.load_state_dict(ckpt.get("model", ckpt), strict=False)
    mae.eval()

    # ── Load Flow ─────────────────────────────────────────────────────
    flow_ckpt = torch.load(cfg.flow_checkpoint, map_location=device)
    flow = FlowModel(feature_dim=cfg.get("flow", {}).get("feature_dim", 384)).to(device)
    flow.load_state_dict(flow_ckpt.get("flow", flow_ckpt))
    flow.eval()

    # ── DataLoaders ───────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Compute Anomaly Scores ────────────────────────────────────────
    print("Computing anomaly scores …")
    trainer = FlowTrainer.__new__(FlowTrainer)
    trainer.flow   = flow
    trainer.mae    = mae
    trainer.device = device

    all_patch_scores, img_scores, stems = trainer.compute_anomaly_scores(val_loader)
    np.save(out / "val_patch_scores.npy", all_patch_scores.numpy())
    np.save(out / "val_img_scores.npy",   img_scores.numpy())

    stats = compute_score_statistics(img_scores.numpy())
    save_metrics_json(stats, out / "score_stats.json")
    print(f"Score statistics: {stats}")

    # ── Score Histogram ───────────────────────────────────────────────
    plot_score_histogram(
        img_scores.numpy(), "MAE + Flow",
        out / "figures" / "score_histogram.png",
    )

    # ── UMAP Analysis ─────────────────────────────────────────────────
    print(f"Running UMAP on up to {cfg.eval.n_umap_samples} patches …")
    all_feats = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            feats = mae.encode(imgs)              # (B, N, D)
            all_feats.append(feats.mean(dim=1).cpu().numpy())  # mean-pool tokens
    all_feats_arr = np.concatenate(all_feats, axis=0)           # (N_patches, D)

    umap_an = UMAPAnalysis(n_neighbors=15, metric="cosine")
    embeddings = umap_an.fit_transform(all_feats_arr, max_samples=cfg.eval.n_umap_samples)
    umap_an.plot(
        embeddings,
        scores=img_scores.numpy()[: len(embeddings)],
        output_path=out / "figures" / "umap_features.png",
        title="UMAP of MAE Encoder Features — coloured by Anomaly Score",
    )
    umap_an.save(out / "umap_reducer.pkl")

    # ── Heatmaps ─────────────────────────────────────────────────────
    print("Generating anomaly heatmaps …")
    heatmap_gen = AnomalyHeatmap()

    # Build {stem: token_scores} dict
    patch_score_dict = {
        stem: all_patch_scores[i].numpy()
        for i, stem in enumerate(stems)
    }

    # Group stems by parent image
    image_stems = sorted({_image_stem_from_patch(s) for s in stems})
    heatmap_dir = out / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    patch_dir = Path(cfg.data.patch_dir)

    for img_stem in image_stems[:20]:   # visualise first 20 images
        score_map = heatmap_gen.stitch(patch_score_dict, img_stem)

        # Load one RGB patch to approximate the original image colour
        matching = [s for s in stems if s.startswith(img_stem)]
        if not matching:
            continue
        first = matching[0]
        rgb_bgr = cv2.imread(str(patch_dir / "RGB" / f"{first}.jpg"))
        if rgb_bgr is None:
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        heatmap_gen.save_png(score_map, rgb, heatmap_dir / f"{img_stem}_heatmap.png",
                             alpha=cfg.eval.heatmap_alpha)
        if cfg.eval.save_geotiff:
            heatmap_gen.save_geotiff(score_map, heatmap_dir / f"{img_stem}_heatmap.tif")

    # ── MAE Reconstructions ───────────────────────────────────────────
    print("Generating MAE reconstruction visualisations …")
    batch = next(iter(val_loader))
    imgs = batch["image"].to(device)[:cfg.eval.n_recon_samples]
    with torch.no_grad():
        _, _, _ = mae(imgs)
        recon = mae.reconstruct(imgs)
    masks = torch.zeros(imgs.shape[0], 64, dtype=torch.bool)  # placeholder
    plot_reconstructions(
        imgs.cpu(), recon.cpu(), masks,
        out / "figures" / "reconstructions.png",
        n_samples=cfg.eval.n_recon_samples,
    )

    print(f"\nEvaluation complete. Results saved to {out}")


if __name__ == "__main__":
    main()
