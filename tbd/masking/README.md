# Binary Weed Segmentation POC

This repo contains a proof-of-concept binary weed segmentation pipeline built around a U-Net with a ResNet34 encoder from `segmentation_models_pytorch`. The same training and evaluation pipeline supports RGB, synthetic multispectral, real multispectral, RGB plus synthetic multispectral, and RGB plus real multispectral input.

Config field documentation is available in `tbd\masking\CONFIG_REFERENCE.md`.

Supported model families in this repo:

- `Unet` with `resnet34`
- `UnetPlusPlus` with `resnet34` and `resnet50`
- `DeepLabV3Plus` with `resnet34` and `resnet50`
- `Segformer` with `mit_b0` and `mit_b1`

## Workflow

Run the commands below from the repository root. Activate the project virtual environment first, then set `PYTHONPATH` so the `tbd\masking\src` package resolves correctly:

```bash
source ./.venv/bin/activate
```

Convenience scripts are available if you want one command for setup or training:

- `bash ./tbd/masking/setup_data.sh`
- `bash ./tbd/masking/run_smoke.sh`
- `bash ./tbd/masking/train_configs.sh --config ...`
- `bash ./tbd/masking/train_rgb_architectures.sh`
- `bash ./tbd/masking/run_multi_seed.sh --config ...`
- `bash ./tbd/masking/evaluate_base_configs.sh --config ...`

1. Create aligned original-image splits:

```bash
python -m src.pack_real_msi --input-dir ./tbd/masking/data/weedy-rice/Multispectral --output-dir ./tbd/masking/data/weedy-rice/MultispectralNPY
python -m src.create_splits --dataset-root ./tbd/masking/data/weedy-rice --output-dir ./tbd/masking/data/splits --group-strategy datetime
```

Or run the full setup in one command:

```bash
bash ./tbd/masking/setup_data.sh
```

2. Create deterministic patch manifests for each modality config:

```bash
python -m src.patchify --config ./tbd/masking/configs/binary/rgb.yaml
python -m src.patchify --config ./tbd/masking/configs/binary/real_msi.yaml
python -m src.patchify --config ./tbd/masking/configs/binary/synth_msi.yaml
python -m src.patchify --config ./tbd/masking/configs/binary/rgb_real_msi.yaml
python -m src.patchify --config ./tbd/masking/configs/binary/rgb_synth_msi.yaml
```

3. Train RGB only:

```bash
python -m src.train --config ./tbd/masking/configs/binary/rgb.yaml
```

Optional fuzzy-halo RGB variant:

```bash
python -m src.train --config ./tbd/masking/configs/fuzzy/rgb.yaml
```

4. Train RGB + synthetic multispectral:

```bash
python -m src.train --config ./tbd/masking/configs/binary/rgb_synth_msi.yaml
```

5. Train RGB + real multispectral:

```bash
python -m src.train --config ./tbd/masking/configs/binary/rgb_real_msi.yaml
```

Architecture-specific examples:

```bash
python -m src.train --config ./tbd/masking/configs/binary/rgb_unetpp_resnet50.yaml
python -m src.train --config ./tbd/masking/configs/binary/rgb_deeplabv3plus_resnet34.yaml
python -m src.train --config ./tbd/masking/configs/binary/rgb_segformer_b1.yaml
```

Or train a preset architecture suite with one command:

```bash
bash ./tbd/masking/train_rgb_architectures.sh
```

Or train any list of configs with one command:

```bash
bash ./tbd/masking/train_configs.sh \
  --config ./tbd/masking/configs/binary/rgb_unetpp_resnet34.yaml \
  --config ./tbd/masking/configs/binary/rgb_deeplabv3plus_resnet50.yaml \
  --summary-output ./tbd/masking/outputs/metrics/custom_train_summary.json
```

Or run the full smoke pipeline with one command:

```bash
bash ./tbd/masking/run_smoke.sh
```

Fuzzy-halo smoke variant:

```bash
bash ./tbd/masking/run_smoke.sh --config configs/smoke/smoke_rgb_fuzzy_halo.yaml
```

Config layout:

- `./tbd/masking/configs/binary/`: standard hard-mask training configs
- `./tbd/masking/configs/fuzzy/`: fuzzy-halo variants that mirror the binary config names
- `./tbd/masking/configs/smoke/`: smoke-test configs
- `./tbd/masking/configs/base.yaml`: shared root defaults

Training writes run artifacts under `outputs/runs/<run_name>/`, including `logs/execution.log` with entries formatted as `%(asctime)s [Trainer] :: %(message)s`. At the end of training, the best validation checkpoint is evaluated once on the configured test split. The resulting test summary is written to `metrics/test_summary.json`, and reconstructed test prediction masks are written to `metrics/test_masks/` as binary PNG files. Per-epoch training and validation do not save masks. Fuzzy-halo runs use the relaxed target as the primary training/evaluation target and also record additional `original_*` test metrics against the hard weed masks.

Additional supported multimodal runs:

- `./tbd/masking/configs/binary/rgb_synth_msi.yaml`: RGB concatenated with synthetic multispectral input, for a 7-channel model.
- `./tbd/masking/configs/binary/rgb_real_msi.yaml`: RGB concatenated with real multispectral input, for a 7-channel model.

Architecture config naming:

- `<modality>_unetpp_resnet34.yaml`
- `<modality>_unetpp_resnet50.yaml`
- `<modality>_deeplabv3plus_resnet34.yaml`
- `<modality>_deeplabv3plus_resnet50.yaml`
- `<modality>_segformer_b0.yaml`
- `<modality>_segformer_b1.yaml`

6. Evaluate the best checkpoint:

```bash
python -m src.evaluate --checkpoint ./tbd/masking/outputs/runs/<run>/checkpoints/best.pt --split test
```

Evaluate an unfine-tuned base model from config:

```bash
python -m src.evaluate_base --config ./tbd/masking/configs/binary/rgb.yaml --split test
```

Evaluation writes artifacts under `outputs/runs/<run>/evaluation/<split>/`, including `overall_metrics.json`, per-patch and per-image CSV metrics, reconstructed predicted masks in `masks/`, limited preview panels in `visuals/`, and `execution.log` with entries formatted as `%(asctime)s [Evaluator] :: %(message)s`. Fuzzy-halo runs keep the primary CSV/JSON metrics on the relaxed target and add `original_*` metrics and companion CSV files for scoring against the hard masks.

Baseline evaluation writes the same evaluation artifacts into a baseline-named run directory under `outputs/runs/`, plus `config.yaml` and `run_metadata.json` with `run_kind: baseline`. It does not create checkpoints or training history.

7. Summarize multiple runs:

```bash
python -m src.summarize --runs ./tbd/masking/outputs/runs/<rgb_run> ./tbd/masking/outputs/runs/<synth_run> ./tbd/masking/outputs/runs/<real_run> --output ./tbd/masking/outputs/metrics/final_comparison.json
```

Summarization writes a JSON file to the requested `--output` path with two top-level sections: `runs` for per-run metrics and `aggregates` for per-model aggregate metrics. Each aggregate row now includes median, mean, and standard deviation fields for every tracked metric. Fuzzy-halo runs also include `original_test_*` metrics so the hard-mask scoring view is preserved alongside the relaxed target view. It also writes a companion log file next to it, for example `outputs/metrics/final_comparison.log`, with entries formatted as `%(asctime)s [Summarizer] :: %(message)s`.

Confidence-style metrics are also included in the summary. They are derived from the sigmoid output probabilities and indicate how certain the model was about its predicted mask values:

- `test_patch_confidence` and `test_image_confidence`: mean certainty across all predicted pixels, computed from `max(p, 1 - p)`
- `test_patch_positive_confidence` and `test_image_positive_confidence`: mean predicted probability on pixels predicted as weed
- `test_patch_negative_confidence` and `test_image_negative_confidence`: mean `1 - p` on pixels predicted as background

These values are useful as model-certainty indicators, but they are not calibrated probabilities of correctness.

## Multi-Seed Runs

To estimate average performance, train the same config multiple times with different seeds and summarize the resulting runs.

Example for one model:

```bash
bash ./tbd/masking/run_multi_seed.sh --config ./tbd/masking/configs/binary/rgb.yaml --repeats 5 --base-seed 1000 --summary-output ./tbd/masking/outputs/metrics/rgb_multi_seed.json
```

Example for multiple models in one shell:

```bash
bash ./tbd/masking/run_multi_seed.sh \
  --config ./tbd/masking/configs/binary/rgb_unetpp_resnet34.yaml \
  --config ./tbd/masking/configs/binary/rgb_deeplabv3plus_resnet34.yaml \
  --config ./tbd/masking/configs/binary/rgb_segformer_b0.yaml \
  --repeats 5 \
  --base-seed 1000 \
  --summary-output ./tbd/masking/outputs/metrics/rgb_architectures_multi_seed.json
```

To spread models across different shells, launch the same script in each shell with a different config list.

The script:

- uses the same deterministic seed schedule for every model
- trains the model, including the training-time test metrics and test mask export
- evaluates the best checkpoint on the test split to produce the full evaluation artifacts used by summarization
- writes a JSON summary with per-run metrics
- includes aggregate per-model mean and standard deviation across runs

Single-run convenience scripts:

- `setup_data.sh`: packs real MSI `.TIF` bands, creates split manifests, and patchifies all five modality baselines
- `run_smoke.sh`: patchifies the checked-in smoke split, trains the smoke config, runs test evaluation, and writes a one-run summary
- `train_configs.sh`: trains and evaluates any explicit list of config files once each
- `train_rgb_architectures.sh`: trains the six requested RGB architecture variants once each and summarizes them
- `evaluate_base_configs.sh`: evaluates one or more unfine-tuned base models from config and can summarize the baseline run set

If you pass `--skip-evaluate`, the script trains only. Training still writes `metrics/test_summary.json` and `metrics/test_masks/`, but the script does not run `src.evaluate` and does not generate the final summary JSON, because summarization reads `evaluation/test/overall_metrics.json`.

Seed scheduling:

- `--base-seed N` controls the first seed used
- repetition `1` uses seed `N`
- repetition `2` uses seed `N + 1`
- repetition `3` uses seed `N + 2`
- because the schedule depends only on repetition index, all models are trained on the same set of seeds
- the summary JSON includes `architecture` and `encoder_name` so runs from different model families stay separated

## Command Reference

Train RGB:

```bash
export PYTHONPATH=./tbd/masking
python -m src.train --config ./tbd/masking/configs/binary/rgb.yaml
```

Train RGB + synthetic multispectral:

```bash
export PYTHONPATH=./tbd/masking
python -m src.train --config ./tbd/masking/configs/binary/rgb_synth_msi.yaml
```

Train RGB + real multispectral:

```bash
export PYTHONPATH=./tbd/masking
python -m src.train --config ./tbd/masking/configs/binary/rgb_real_msi.yaml
```

Evaluate a run:

```bash
export PYTHONPATH=./tbd/masking
python -m src.evaluate --checkpoint ./tbd/masking/outputs/runs/<run>/checkpoints/best.pt --split test
```

Evaluate an unfine-tuned base run:

```bash
export PYTHONPATH=./tbd/masking
python -m src.evaluate_base --config ./tbd/masking/configs/binary/rgb.yaml --split test
```

Create a summary table:

```bash
export PYTHONPATH=./tbd/masking
python -m src.summarize --runs ./tbd/masking/outputs/runs/<rgb_run> ./tbd/masking/outputs/runs/<synth_run> ./tbd/masking/outputs/runs/<real_run> --output ./tbd/masking/outputs/metrics/final_comparison.json
```

Run repeated multi-seed experiments:

```bash
bash ./tbd/masking/run_multi_seed.sh \
  --config ./tbd/masking/configs/binary/rgb_unetpp_resnet34.yaml \
  --config ./tbd/masking/configs/binary/rgb_deeplabv3plus_resnet34.yaml \
  --config ./tbd/masking/configs/binary/rgb_segformer_b0.yaml \
  --repeats 5 \
  --base-seed 1000 \
  --summary-output ./tbd/masking/outputs/metrics/rgb_architectures_multi_seed.json
```

Run the full smoke pipeline:

```bash
bash ./tbd/masking/run_smoke.sh
```

## Notes

- Splits are created at the original-image level before patching.
- `run_smoke.sh` uses the checked-in `smoke_train.csv`, `smoke_val.csv`, and `smoke_test.csv` manifests rather than regenerating smoke splits.
- Patch manifests are deterministic given the config seed.
- RGB normalization uses ImageNet statistics.
- All other modalities, including the two RGB+MSI combined variants, compute normalization from train patches only and cache it as JSON.
- The optional `fuzzy_halo` target variant keeps the labeled weed mask as the core region and adds a soft halo only outside the labeled boundary.
- Evaluation reconstructs full-image predictions by averaging patch probabilities into the original image canvas.
- Training-time test evaluation saves reconstructed binary prediction masks under `metrics/test_masks/`.
- Standalone evaluation saves reconstructed binary prediction masks under `evaluation/<split>/masks/`.
- `synthetic_msi_path` is expected to point to a single `.npy` or `.npz` file per sample.
- `real_msi_path` is expected to point to a single packed `.npy` file per sample. Use `python -m src.pack_real_msi` to convert legacy per-band `.TIF` files into this format.
