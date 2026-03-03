"""
Entry point: Run Stage 2 inference, generate heatmaps, UMAP, and score plots.

Usage:
    python evaluate.py
    python evaluate.py flow.output_dir=outputs/stage2_flow flow.export_geotiff=true
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data.dataloader import build_inference_loader
from src.models.mae import MaskedAutoencoder
from src.models.flow_model import build_flow_model
from src.evaluation.heatmap import assemble_heatmap, smooth_heatmap, normalise_heatmap, export_geotiff
from src.evaluation.metrics import score_statistics, estimate_anomaly_fraction
from src.evaluation.visualize import overlay_heatmap_on_rgb, plot_score_distribution, save_figure
from src.evaluation.umap_analysis import compute_umap_embedding, plot_umap_embedding

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="flow", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heatmap_dir = Path(cfg.flow.heatmap_output_dir)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inference data ───────────────────────────────────────────────────
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

    # ── Load models ───────────────────────────────────────────────────────────
    mae_model = MaskedAutoencoder(
        in_channels=cfg.data.in_channels,
        img_size=cfg.data.patch_size,
        patch_size=cfg.mae.patch_size,
        arch=cfg.mae.arch,
        masking_ratio=cfg.mae.masking_ratio,
        decoder_embed_dim=cfg.mae.decoder_embed_dim,
        decoder_depth=cfg.mae.decoder_depth,
        use_checkpoint=False,
    )
    mae_ckpt = torch.load(cfg.flow.mae_checkpoint, map_location=device)
    mae_model.load_state_dict(mae_ckpt["model"])
    mae_model = mae_model.to(device).eval()
    for p in mae_model.parameters():
        p.requires_grad = False

    flow_model = build_flow_model(
        feature_dim=cfg.flow.feature_dim,
        num_coupling_blocks=cfg.flow.num_coupling_blocks,
        subnet_hidden_dim=cfg.flow.subnet_hidden_dim,
    ).to(device)
    flow_ckpt = torch.load(
        str(Path(cfg.flow.output_dir) / "flow_best.pth"), map_location=device
    )
    flow_model.load_state_dict(flow_ckpt["flow"])
    flow_model.eval()
    score_mean = flow_ckpt["score_mean"]
    score_std  = flow_ckpt["score_std"]

    # ── Run inference ─────────────────────────────────────────────────────────
    # Per-image: collect (score, y, x) tuples
    per_image_patches: dict[int, list[tuple[float, int, int]]] = {}
    all_raw_scores: list[np.ndarray] = []
    all_features:  list[np.ndarray] = []

    log.info("Running inference...")
    with torch.no_grad():
        for batch in infer_loader:
            imgs  = batch["image"].to(device, non_blocking=True)
            r_idx = batch["record_idx"].numpy()
            ys    = batch["patch_y"].numpy()
            xs    = batch["patch_x"].numpy()

            # Feature extraction
            tokens = mae_model.encode(imgs)          # (B, 1+N, D)
            feats  = tokens[:, 1:, :].mean(dim=1)   # (B, D) — pool over patches
            all_features.append(feats.cpu().numpy())

            # NLL scores (z-scored)
            raw_nll = flow_model.nll_score(feats.reshape(-1, cfg.flow.feature_dim))
            z_scores = (raw_nll - score_mean) / score_std  # (B,)
            all_raw_scores.append(z_scores.cpu().numpy())

            for b in range(len(imgs)):
                rid  = int(r_idx[b])
                y, x = int(ys[b]), int(xs[b])
                per_image_patches.setdefault(rid, []).append(
                    (float(z_scores[b].item()), y, x)
                )

    all_scores_np = np.concatenate(all_raw_scores)
    all_feats_np  = np.concatenate(all_features, axis=0)

    # ── Score statistics ──────────────────────────────────────────────────────
    stats = score_statistics(all_scores_np)
    anom_frac = estimate_anomaly_fraction(all_scores_np)
    log.info(f"Score stats: {stats}")
    log.info(f"Estimated anomaly fraction (z>2.5): {anom_frac * 100:.2f}%")

    # ── Score distribution plot ───────────────────────────────────────────────
    fig = plot_score_distribution(
        {"MAE + FastFlow": all_scores_np},
        title="MAE+Flow Anomaly Score Distribution",
    )
    save_figure(fig, fig_dir / "mae_flow_score_distribution.png")

    # ── Heatmap generation ────────────────────────────────────────────────────
    log.info("Generating heatmaps...")
    for rid, patch_list in per_image_patches.items():
        record = infer_dataset.records[rid]
        # Get image size by loading the (cached) image
        img_arr = infer_dataset._load_full_image(record)
        H, W = img_arr.shape[:2]

        heatmap = assemble_heatmap(patch_list, H, W, cfg.data.patch_size)
        heatmap = smooth_heatmap(heatmap, sigma=5.0)
        heatmap_norm = normalise_heatmap(heatmap, method="percentile")

        # Overlay on RGB
        if cfg.data.include_rgb:
            rgb = img_arr[..., :3]
        else:
            # Construct pseudo-RGB from G, R, NIR channels
            rgb = np.stack([img_arr[..., 1], img_arr[..., 0], img_arr[..., 0]], axis=-1)

        fig = overlay_heatmap_on_rgb(
            rgb, heatmap_norm, title=f"Anomaly Heatmap: {record.stem}"
        )
        save_figure(fig, heatmap_dir / f"{record.stem}_heatmap.png")

        if cfg.flow.export_geotiff:
            export_geotiff(
                heatmap,
                output_path=heatmap_dir / f"{record.stem}_anomaly.tif",
            )

    # ── UMAP ─────────────────────────────────────────────────────────────────
    log.info("Computing UMAP embedding...")
    embedding = compute_umap_embedding(all_feats_np, seed=cfg.data.seed)
    # Colour by anomaly score
    fig = plot_umap_embedding(
        embedding,
        scores=all_scores_np[:len(embedding)],  # Align if sub-sampled
        title="UMAP of MAE Patch Features (coloured by anomaly score)",
    )
    save_figure(fig, fig_dir / "umap_mae_features.png")

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
