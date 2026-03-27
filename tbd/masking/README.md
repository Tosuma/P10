# Binary Weed Segmentation POC

This repo contains a proof-of-concept binary weed segmentation pipeline built around a U-Net with a ResNet34 encoder from `segmentation_models_pytorch`. The same training and evaluation pipeline supports RGB, synthetic multispectral, real multispectral, RGB plus synthetic multispectral, and RGB plus real multispectral input.

## Workflow

Run the commands below from the repository root. They use the root virtual environment at `.\.venv` and set `PYTHONPATH` so the `tbd\masking\src` package resolves correctly:

```powershell
$env:PYTHONPATH=".\tbd\masking"
```

1. Create aligned original-image splits:

```powershell
.\.venv\Scripts\python.exe -m src.create_splits --dataset-root .\tbd\masking\data\weedy-rice --output-dir .\tbd\masking\data\splits --group-strategy datetime
```

2. Create deterministic patch manifests for each modality config:

```powershell
.\.venv\Scripts\python.exe -m src.patchify --config .\tbd\masking\configs\rgb.yaml
.\.venv\Scripts\python.exe -m src.patchify --config .\tbd\masking\configs\real_msi.yaml
.\.venv\Scripts\python.exe -m src.patchify --config .\tbd\masking\configs\synth_msi.yaml
.\.venv\Scripts\python.exe -m src.patchify --config .\tbd\masking\configs\rgb_real_msi.yaml
.\.venv\Scripts\python.exe -m src.patchify --config .\tbd\masking\configs\rgb_synth_msi.yaml
```

3. Train RGB only:

```powershell
.\.venv\Scripts\python.exe -m src.train --config .\tbd\masking\configs\rgb.yaml
```

4. Train RGB + synthetic multispectral:

```powershell
.\.venv\Scripts\python.exe -m src.train --config .\tbd\masking\configs\rgb_synth_msi.yaml
```

5. Train RGB + real multispectral:

```powershell
.\.venv\Scripts\python.exe -m src.train --config .\tbd\masking\configs\rgb_real_msi.yaml
```

Training writes run artifacts under `outputs/runs/<run_name>/`, including `logs/execution.log` with entries formatted as `%(asctime)s [Trainer] :: %(message)s`.

Additional supported multimodal runs:

- `.\tbd\masking\configs\rgb_synth_msi.yaml`: RGB concatenated with synthetic multispectral input, for a 7-channel model.
- `.\tbd\masking\configs\rgb_real_msi.yaml`: RGB concatenated with real multispectral input, for a 7-channel model.

6. Evaluate the best checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint .\tbd\masking\outputs\runs\<run>\checkpoints\best.pt --split test
```

Evaluation writes artifacts under `outputs/runs/<run>/evaluation/<split>/`, including `execution.log` with entries formatted as `%(asctime)s [Evaluator] :: %(message)s`.

7. Summarize multiple runs:

```powershell
.\.venv\Scripts\python.exe -m src.summarize --runs .\tbd\masking\outputs\runs\<rgb_run> .\tbd\masking\outputs\runs\<synth_run> .\tbd\masking\outputs\runs\<real_run> --output .\tbd\masking\outputs\metrics\final_comparison.csv
```

Summarization writes the comparison CSV to the requested `--output` path and writes a companion log file next to it, for example `outputs/metrics/final_comparison.log`, with entries formatted as `%(asctime)s [Summarizer] :: %(message)s`.

## Command Reference

Train RGB:

```powershell
$env:PYTHONPATH=".\tbd\masking"
.\.venv\Scripts\python.exe -m src.train --config .\tbd\masking\configs\rgb.yaml
```

Train RGB + synthetic multispectral:

```powershell
$env:PYTHONPATH=".\tbd\masking"
.\.venv\Scripts\python.exe -m src.train --config .\tbd\masking\configs\rgb_synth_msi.yaml
```

Train RGB + real multispectral:

```powershell
$env:PYTHONPATH=".\tbd\masking"
.\.venv\Scripts\python.exe -m src.train --config .\tbd\masking\configs\rgb_real_msi.yaml
```

Evaluate a run:

```powershell
$env:PYTHONPATH=".\tbd\masking"
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint .\tbd\masking\outputs\runs\<run>\checkpoints\best.pt --split test
```

Create a summary table:

```powershell
$env:PYTHONPATH=".\tbd\masking"
.\.venv\Scripts\python.exe -m src.summarize --runs .\tbd\masking\outputs\runs\<rgb_run> .\tbd\masking\outputs\runs\<synth_run> .\tbd\masking\outputs\runs\<real_run> --output .\tbd\masking\outputs\metrics\final_comparison.csv
```

## Notes

- Splits are created at the original-image level before patching.
- Patch manifests are deterministic given the config seed.
- RGB normalization uses ImageNet statistics.
- All other modalities, including the two RGB+MSI combined variants, compute normalization from train patches only and cache it as JSON.
- Evaluation reconstructs full-image predictions by averaging patch probabilities into the original image canvas.
- `synthetic_msi_path` is expected to point to a single `.npy` or `.npz` file per sample.
