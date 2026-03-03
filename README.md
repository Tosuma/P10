# Plant Stress Anomaly Detection

Unsupervised anomaly detection for plant stress in drone multispectral imagery.
Two-stage pipeline: MAE pretraining (ViT + Spectral Attention) → FastFlow anomaly scoring.

## Project Structure

```
src/
├── data/
│   ├── band_loader.py          — rasterio-based band loading + spatial alignment
│   ├── vegetation_indices.py   — NDVI, NDRE, SAVI, EVI computation
│   ├── patch_dataset.py        — patch extraction, augmentation, train/val split
│   └── dataloader.py           — DataLoader factory (DDP-aware)
├── models/
│   ├── spectral_attention.py   — SE-Net style per-band gating module
│   ├── vit_encoder.py          — Multi-channel ViT (timm blocks, sinusoidal PE)
│   ├── mae.py                  — Full MAE with masking, decoder, MSE loss
│   └── flow_model.py           — FastFlow (FrEIA) + SimpleRealNVP fallback
├── training/
│   ├── mae_trainer.py          — DDP + AMP + WandB + TensorBoard + warmup cosine LR
│   └── flow_trainer.py         — Frozen encoder extraction + flow training + z-score calibration
├── evaluation/
│   ├── metrics.py              — AUROC, AUPRC, score statistics, anomaly fraction
│   ├── heatmap.py              — Patch score assembly, GeoTIFF export
│   ├── visualize.py            — Heatmap overlay, ROC curves, score distributions
│   └── umap_analysis.py        — UMAP embedding + scatter plots
└── baselines/
    ├── ndvi_threshold.py       — Classical NDVI thresholding
    ├── pca_kmeans.py           — PCA + MiniBatchKMeans with coreset sampling
    ├── conv_autoencoder.py     — U-Net style CAE + trainer
    └── patchcore.py            — ResNet50 memory bank + greedy coreset
configs/                        — Hydra YAML (data, mae, flow, baselines)
scripts/slurm/                  — SLURM job scripts (DDP for Stage 1)
train_mae.py                    — Stage 1 entry point
train_flow.py                   — Stage 2 entry point
run_baselines.py                — Baseline entry point
evaluate.py                     — Inference + heatmap + UMAP entry point
```

## Usage

```bash
pip install -r requirements.txt

# Stage 1 — MAE pretraining (single GPU)
python train_mae.py data.rgb_dir=data/RGB data.ms_dir=data/Multispectral

# Stage 1 — MAE pretraining (4 GPUs)
torchrun --standalone --nproc_per_node=4 train_mae.py

# Stage 2 — FastFlow training
python train_flow.py flow.mae_checkpoint=outputs/stage1_mae/mae_best.pth

# Baselines
python run_baselines.py

# Inference + heatmaps + UMAP
python evaluate.py
```

## Key Architectural Decisions

| Decision | Rationale |
| --- | --- |
| Spectral Attention before patch embed | Per-band gating, interpretable per-band importance weights |
| Fixed sinusoidal PE (not learned) | Generalises across arbitrary patch positions from 4000 images |
| No per-patch normalisation in MAE loss | Preserves inter-channel ratios encoding plant physiology |
| Normalizing flow over reconstruction error | Proper NLL scores; more sensitive to subtle distribution shifts |
| Image-level train/val split | Prevents patch-level data leakage |
| Horizontal flip excluded from augmentation | Field row orientation is meaningful in aerial view |
| PatchCore uses RGB-only | Known limitation (ImageNet domain gap); important ablation point |
