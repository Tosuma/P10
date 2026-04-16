# Config Reference

This document explains the fields used by the YAML config files in `tbd\masking\configs`.

Numeric fields may use normal decimal notation or scientific notation such as `1e-4` and `1e-6`.

## How config inheritance works

- A config can declare `base_config: base.yaml`.
- When present, the child config overrides only the fields it redefines.
- Effective runtime config is written into each run directory as `config.yaml`.

## Top-level fields

### `experiment_name`

- Human-readable run name prefix.
- Used when building the output run directory name.

### `modality`

- Selects the model input type.
- Supported values:
  - `rgb`
  - `synthetic_msi`
  - `real_msi`
  - `rgb_synthetic_msi`
  - `rgb_real_msi`

### `in_channels`

- Number of image channels the model expects.
- Typical values:
  - `3` for RGB
  - `4` for MSI-only inputs in this project
- `7` for RGB + MSI combined inputs

### `model`

- Defines which segmentation model family and encoder backbone to build.

#### `model.architecture`

- Segmentation model family.
- Supported values in this project:
  - `Unet`
  - `UnetPlusPlus`
  - `DeepLabV3Plus`
  - `Segformer`

#### `model.encoder_name`

- Encoder or backbone name passed to the model implementation.
- Typical values used in this project:
  - `resnet34`
  - `resnet50`
  - `mit_b0`
  - `mit_b1`

#### `model.encoder_weights`

- Pretrained encoder weights identifier.
- Typical value is `imagenet`.

#### `model.classes`

- Number of output segmentation classes.
- This binary segmentation pipeline uses `1`.

### `seed`

- Global random seed used for Python, NumPy, and PyTorch.
- If set to an integer such as `42`, that exact seed is used.
- If set to `null`, a fresh random seed is generated at runtime.
- If the field is omitted and no inherited config provides a value, a fresh random seed is generated at runtime.
- If your config inherits from `base.yaml`, use `seed: null` to override the inherited seed and force random-seed generation.
- Training writes the resolved seed into the run metadata and saved run config.

### `device`

- Optional PyTorch device string.
- Examples: `cpu`, `cuda`, `cuda:0`
- If `null`, the code auto-selects CUDA when available, otherwise CPU.

## `paths`

### `run_root`

- Directory where training runs are written.

### `train_manifest`, `val_manifest`, `test_manifest`

- CSV files describing the samples in each split.
- `real_msi_path` entries in these manifests should point to a single packed `.npy` file for each sample.

### `patch_manifest_dir`

- Directory where `src.patchify` writes generated patch CSVs.

### `train_patch_manifest`, `val_patch_manifest`, `test_patch_manifest`

- Patch-level CSV manifests consumed by training and evaluation.

### `normalization_stats`

- JSON file storing channel mean and standard deviation used for normalization.
- RGB runs use ImageNet stats.
- Other modalities compute stats from train patches and cache them here.

## `patch`

### `mode`

- Patch sampling strategy.
- Supported values:
  - `fixed_grid`
  - `random_crop`

### `size`

- Patch size as `[width, height]`.

### `patches_per_image`

- Only used by random cropping.
- Number of random patches to sample per image.

### `keep_empty_probability`

- Probability of keeping a patch with no weed pixels.
- Helps reduce class imbalance without dropping all-negative context entirely.

## `augmentations`

### `brightness`

- Random brightness jitter strength.
- Applied to RGB channels only.

### `contrast`

- Random contrast jitter strength.
- Applied to RGB channels only.

### `noise_std`

- Standard deviation of additive Gaussian noise.
- Applied to all channels.

## `loss`

### `bce_weight`

- Weight of the binary cross-entropy term in the segmentation loss.

### `dice_weight`

- Weight of the Dice loss term in the segmentation loss.

## `target`

### `mode`

- Training/evaluation target style.
- Supported values:
  - `binary`: current hard weed mask
  - `fuzzy_halo`: keep the labeled weed pixels at `1.0` and add a soft halo outside the weed boundary

### `halo_radius_px`

- Only used for `fuzzy_halo`.
- Number of pixels outside the hard weed mask where the relaxed target remains positive.

### `halo_min_value`

- Only used for `fuzzy_halo`.
- Lowest soft-target value used near the outer edge of the halo band.

## `training`

### `batch_size`

- Number of patches per optimizer step.

### `num_workers`

- Number of PyTorch dataloader worker processes.

### `lr`

- Initial learning rate for `AdamW`.

### `weight_decay`

- Weight decay used by `AdamW`.

### `epochs`

- Maximum number of training epochs.

### `grad_clip`

- Gradient norm clipping threshold.

### `early_stopping_patience`

- Number of consecutive epochs without validation IoU improvement before training stops early.

### `freeze_encoder_epochs`

- Number of initial epochs where the encoder is frozen.
- Use `0` to train the full network from the start.

### `scheduler`

- Scheduler settings for `torch.optim.lr_scheduler.CosineAnnealingLR`.

#### `t_max`

- Number of scheduler steps in one cosine decay cycle.
- In this project, the scheduler steps once per epoch.
- A common choice is to match `training.epochs`.

#### `eta_min`

- Minimum learning rate reached at the end of the cosine schedule.

## `evaluation`

### `threshold`

- Probability threshold used to convert predicted probabilities into binary masks.

### `num_visualizations`

- Number of sample visualization panels to save during evaluation.
- For `fuzzy_halo` runs, evaluation reports both the primary relaxed-target metrics and additional `original_*` metrics against the hard weed mask.
