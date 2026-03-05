# Unsupervised Plant Stress Detection in Drone Multispectral Imagery

Master's Thesis — Computer Science, Aalborg University

A two-stage unsupervised anomaly detection pipeline for detecting plant stress in unlabelled drone imagery of agricultural fields.

---

## Overview

**Stage 1 — MAE Pretraining:**
A Vision Transformer (ViT-Small) with a spectral attention module is pretrained via Masked Autoencoder (MAE) on 10-channel multispectral+VI drone patches. No labels are used.

**Stage 2 — Anomaly Scoring:**
A normalizing flow (FastFlow / SimpleRealNVP fallback) is trained on frozen MAE encoder features. Anomaly score = negative log-likelihood (NLL). High NLL → stressed / anomalous vegetation.

**Baselines:** NDVI thresholding, PCA+k-means, Convolutional Autoencoder, PatchCore.

---

## Dataset

**WeedyRice-RGBMS-DB** — 734 paired RGB + multispectral drone images
Sensor: DJI Mavic 3 Multispectral | Location: Mekong Delta, Vietnam (2024–2025)

Pre-extracted patches (128×128, ~220k per modality) in `data/WeedyRice-patches/`:

```
RGB/           .jpg   H×W×3 uint8
Multispectral/ .npy   float32 H×W×4  (G, R, RE, NIR)
Masks/         .png   uint8 binary
NDVI/          .npy   float32 H×W
NDRE/          .npy   float32 H×W
SAVI/          .npy   float32 H×W
```

---

## Setup

```bash
python -m venv my_venv && source my_venv/bin/activate
pip install -r requirements.txt
pip install FrEIA            # optional but recommended for best flow architecture
```

---

## Data Preparation

```bash
# Compute vegetation index maps
python utils/compute_vegetation_indices.py --data-root data/WeedyRice-RGBMS-DB --workers 8

# Extract 128×128 patches
python utils/patch_weedyrice.py \
    --data-root data/WeedyRice-RGBMS-DB --output-root data/WeedyRice-patches \
    --patch-size 128 --workers 8
```

SLURM: `sbatch scripts/slurm/compute_vegetation_indices.sh` then `sbatch scripts/slurm/patch_weedyrice.sh`

---

## Training

### Stage 1 — MAE Pretraining
```bash
torchrun --standalone --nproc_per_node=4 train_mae.py   # multi-GPU
python train_mae.py data.batch_size=64 logging.use_wandb=false  # single-GPU debug
```
SLURM: `sbatch scripts/slurm/train_mae.sh`

### Stage 2 — Flow Model
```bash
python train_flow.py mae_checkpoint=checkpoints/mae/best.pth
```
SLURM: `sbatch scripts/slurm/train_flow.sh`

---

## Baselines
```bash
python run_baselines.py         # or: sbatch scripts/slurm/run_baselines.sh
```

---

## Evaluation
```bash
python evaluate.py mae_checkpoint=checkpoints/mae/best.pth flow_checkpoint=checkpoints/flow/best.pth
# or: sbatch scripts/slurm/evaluate.sh
```

Outputs: `results/evaluation/` — score histogram, UMAP, per-image heatmaps (PNG + GeoTIFF).

---

## Project Structure

```
src/
  data/       patch_dataset.py, dataloader.py
  models/     spectral_attention.py, vit_encoder.py, mae.py, flow_model.py
  training/   mae_trainer.py, flow_trainer.py
  evaluation/ metrics.py, heatmap.py, visualize.py, umap_analysis.py
  baselines/  ndvi_threshold.py, pca_kmeans.py, conv_autoencoder.py, patchcore.py
configs/      mae.yaml, flow.yaml, baselines.yaml, eval.yaml
scripts/slurm/ train_mae.sh, train_flow.sh, run_baselines.sh, evaluate.sh
utils/        compute_vegetation_indices.py, patch_weedyrice.py
train_mae.py, train_flow.py, run_baselines.py, evaluate.py
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SpectralAttention before patch embed | Learns per-band importance before spatial aggregation; interpretable |
| Fixed 2D sinusoidal PE | Better spatial generalisation than learned PE for finite aerial datasets |
| Mask ratio 0.75 | He et al. 2022: forces semantic representation learning |
| No per-patch normalisation | Preserves inter-channel spectral ratios for VI computation |
| Image-level train/val split | Prevents data leakage between spatially adjacent patches |
| No horizontal flip | Field row orientation is meaningful in aerial imagery |
| Feature caching in Stage 2 | Frozen encoder → pre-compute once; 5–10× faster flow training |
| Per-token flow scoring | 64 NLL scores per patch → 8×8 heatmap → stitchable full-image map |

---

## Baselines Comparison

| Method | Input | Pretraining | Spatial Resolution |
|--------|-------|-------------|-------------------|
| NDVI Threshold | NDVI (1ch) | None | Pixel-level |
| PCA + k-means | 10-ch spectral mean | None | Patch-mean |
| Conv Autoencoder | 10 channels | None | Full patch |
| PatchCore | RGB only (3ch) | ImageNet | Pooled spatial |
| **MAE + Flow (ours)** | 10 channels | Domain-specific SSL | Token-level (8×8) |

---

## HPC (AAU SLURM Cluster)

```bash
export DATA_ROOT=/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-patches

# Submit in order (each depends on the previous completing):
sbatch scripts/slurm/train_mae.sh
sbatch scripts/slurm/train_flow.sh
sbatch scripts/slurm/run_baselines.sh   # can overlap with flow training
sbatch scripts/slurm/evaluate.sh
```

WandB monitoring: project `P10-MAE` (Stage 1) and `P10-Flow` (Stage 2).
