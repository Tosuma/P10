# Binary Weed Segmentation POC

This repo contains a proof-of-concept binary weed segmentation pipeline built around a U-Net with a ResNet34 encoder from `segmentation_models_pytorch`. The same training and evaluation pipeline supports RGB, synthetic multispectral, and ground-truth multispectral input.

## Workflow

1. Create aligned original-image splits:

```powershell
.venv\Scripts\python.exe -m src.create_splits --dataset-root data/weedy-rice --output-dir data/splits --group-strategy datetime
```

2. Create deterministic patch manifests for each modality config:

```powershell
.venv\Scripts\python.exe -m src.patchify --config configs/rgb.yaml
.venv\Scripts\python.exe -m src.patchify --config configs/synth_msi.yaml
.venv\Scripts\python.exe -m src.patchify --config configs/real_msi.yaml
```

3. Train:

```powershell
.venv\Scripts\python.exe -m src.train --config configs/rgb.yaml
```

Training writes run artifacts under `outputs/runs/<run_name>/`, including `logs/execution.log` with entries formatted as `%(asctime)s [Trainer] :: %(message)s`.

4. Evaluate the best checkpoint:

```powershell
.venv\Scripts\python.exe -m src.evaluate --checkpoint outputs/runs/<run>/checkpoints/best.pt --split test
```

Evaluation writes artifacts under `outputs/runs/<run>/evaluation/<split>/`, including `execution.log` with entries formatted as `%(asctime)s [Evaluator] :: %(message)s`.

5. Summarize multiple runs:

```powershell
.venv\Scripts\python.exe -m src.summarize --runs outputs/runs/<rgb_run> outputs/runs/<synth_run> outputs/runs/<real_run> --output outputs/metrics/final_comparison.csv
```

Summarization writes the comparison CSV to the requested `--output` path and writes a companion log file next to it, for example `outputs/metrics/final_comparison.log`, with entries formatted as `%(asctime)s [Summarizer] :: %(message)s`.

## Command Reference

Train a run:

```powershell
cd tbd\masking
..\..\.venv\Scripts\python.exe -m src.train --config configs\rgb.yaml
```

Evaluate a run:

```powershell
cd tbd\masking
..\..\.venv\Scripts\python.exe -m src.evaluate --checkpoint outputs\runs\<run>\checkpoints\best.pt --split test
```

Create a summary table:

```powershell
cd tbd\masking
..\..\.venv\Scripts\python.exe -m src.summarize --runs outputs\runs\<rgb_run> outputs\runs\<synth_run> outputs\runs\<real_run> --output outputs\metrics\final_comparison.csv
```

## Notes

- Splits are created at the original-image level before patching.
- Patch manifests are deterministic given the config seed.
- RGB normalization uses ImageNet statistics.
- Non-RGB normalization is computed from train patches only and cached as JSON.
- Evaluation reconstructs full-image predictions by averaging patch probabilities into the original image canvas.
- `synthetic_msi_path` is expected to point to a single `.npy` or `.npz` file per sample.
