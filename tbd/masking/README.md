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
source ./.venv/Scripts/activate
export PYTHONPATH=./tbd/masking
```

1. Create aligned original-image splits:

```bash
python -m src.pack_real_msi --input-dir ./tbd/masking/data/weedy-rice/Multispectral --output-dir ./tbd/masking/data/weedy-rice/MultispectralNPY
python -m src.create_splits --dataset-root ./tbd/masking/data/weedy-rice --output-dir ./tbd/masking/data/splits --group-strategy datetime
```

2. Create deterministic patch manifests for each modality config:

```bash
python -m src.patchify --config ./tbd/masking/configs/rgb.yaml
python -m src.patchify --config ./tbd/masking/configs/real_msi.yaml
python -m src.patchify --config ./tbd/masking/configs/synth_msi.yaml
python -m src.patchify --config ./tbd/masking/configs/rgb_real_msi.yaml
python -m src.patchify --config ./tbd/masking/configs/rgb_synth_msi.yaml
```

3. Train RGB only:

```bash
python -m src.train --config ./tbd/masking/configs/rgb.yaml
```

4. Train RGB + synthetic multispectral:

```bash
python -m src.train --config ./tbd/masking/configs/rgb_synth_msi.yaml
```

5. Train RGB + real multispectral:

```bash
python -m src.train --config ./tbd/masking/configs/rgb_real_msi.yaml
```

Architecture-specific examples:

```bash
python -m src.train --config ./tbd/masking/configs/rgb_unetpp_resnet50.yaml
python -m src.train --config ./tbd/masking/configs/rgb_deeplabv3plus_resnet34.yaml
python -m src.train --config ./tbd/masking/configs/rgb_segformer_b1.yaml
```

Training writes run artifacts under `outputs/runs/<run_name>/`, including `logs/execution.log` with entries formatted as `%(asctime)s [Trainer] :: %(message)s`.

Additional supported multimodal runs:

- `./tbd/masking/configs/rgb_synth_msi.yaml`: RGB concatenated with synthetic multispectral input, for a 7-channel model.
- `./tbd/masking/configs/rgb_real_msi.yaml`: RGB concatenated with real multispectral input, for a 7-channel model.

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

Evaluation writes artifacts under `outputs/runs/<run>/evaluation/<split>/`, including `execution.log` with entries formatted as `%(asctime)s [Evaluator] :: %(message)s`.

7. Summarize multiple runs:

```bash
python -m src.summarize --runs ./tbd/masking/outputs/runs/<rgb_run> ./tbd/masking/outputs/runs/<synth_run> ./tbd/masking/outputs/runs/<real_run> --output ./tbd/masking/outputs/metrics/final_comparison.json
```

Summarization writes a JSON file to the requested `--output` path with two top-level sections: `runs` for per-run metrics and `aggregates` for per-model mean/std values. It also writes a companion log file next to it, for example `outputs/metrics/final_comparison.log`, with entries formatted as `%(asctime)s [Summarizer] :: %(message)s`.

Confidence-style metrics are also included in the summary. They are derived from the sigmoid output probabilities and indicate how certain the model was about its predicted mask values:

- `test_patch_confidence` and `test_image_confidence`: mean certainty across all predicted pixels, computed from `max(p, 1 - p)`
- `test_patch_positive_confidence` and `test_image_positive_confidence`: mean predicted probability on pixels predicted as weed
- `test_patch_negative_confidence` and `test_image_negative_confidence`: mean `1 - p` on pixels predicted as background

These values are useful as model-certainty indicators, but they are not calibrated probabilities of correctness.

## Multi-Seed Runs

To estimate average performance, train the same config multiple times with different seeds and summarize the resulting runs.

Example for one model:

```bash
bash ./tbd/masking/run_multi_seed.sh --config ./tbd/masking/configs/rgb.yaml --repeats 5 --base-seed 1000 --summary-output ./tbd/masking/outputs/metrics/rgb_multi_seed.json
```

Example for multiple models in one shell:

```bash
bash ./tbd/masking/run_multi_seed.sh \
  --config ./tbd/masking/configs/rgb_unetpp_resnet34.yaml \
  --config ./tbd/masking/configs/rgb_deeplabv3plus_resnet34.yaml \
  --config ./tbd/masking/configs/rgb_segformer_b0.yaml \
  --repeats 5 \
  --base-seed 1000 \
  --summary-output ./tbd/masking/outputs/metrics/rgb_architectures_multi_seed.json
```

To spread models across different shells, launch the same script in each shell with a different config list.

The script:

- uses the same deterministic seed schedule for every model
- trains the model
- evaluates the best checkpoint on the test split
- writes a JSON summary with per-run metrics
- includes aggregate per-model mean and standard deviation across runs

If you pass `--skip-evaluate`, the script trains only and does not generate the final summary JSON, because evaluation metrics are required for summarization.

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
python -m src.train --config ./tbd/masking/configs/rgb.yaml
```

Train RGB + synthetic multispectral:

```bash
export PYTHONPATH=./tbd/masking
python -m src.train --config ./tbd/masking/configs/rgb_synth_msi.yaml
```

Train RGB + real multispectral:

```bash
export PYTHONPATH=./tbd/masking
python -m src.train --config ./tbd/masking/configs/rgb_real_msi.yaml
```

Evaluate a run:

```bash
export PYTHONPATH=./tbd/masking
python -m src.evaluate --checkpoint ./tbd/masking/outputs/runs/<run>/checkpoints/best.pt --split test
```

Create a summary table:

```bash
export PYTHONPATH=./tbd/masking
python -m src.summarize --runs ./tbd/masking/outputs/runs/<rgb_run> ./tbd/masking/outputs/runs/<synth_run> ./tbd/masking/outputs/runs/<real_run> --output ./tbd/masking/outputs/metrics/final_comparison.json
```

Run repeated multi-seed experiments:

```bash
bash ./tbd/masking/run_multi_seed.sh \
  --config ./tbd/masking/configs/rgb_unetpp_resnet34.yaml \
  --config ./tbd/masking/configs/rgb_deeplabv3plus_resnet34.yaml \
  --config ./tbd/masking/configs/rgb_segformer_b0.yaml \
  --repeats 5 \
  --base-seed 1000 \
  --summary-output ./tbd/masking/outputs/metrics/rgb_architectures_multi_seed.json
```

## Notes

- Splits are created at the original-image level before patching.
- Patch manifests are deterministic given the config seed.
- RGB normalization uses ImageNet statistics.
- All other modalities, including the two RGB+MSI combined variants, compute normalization from train patches only and cache it as JSON.
- Evaluation reconstructs full-image predictions by averaging patch probabilities into the original image canvas.
- `synthetic_msi_path` is expected to point to a single `.npy` or `.npz` file per sample.
- `real_msi_path` is expected to point to a single packed `.npy` file per sample. Use `python -m src.pack_real_msi` to convert legacy per-band `.TIF` files into this format.
