# Binary Weed Segmentation POC

This repo contains a proof-of-concept binary weed segmentation pipeline built around a U-Net with a ResNet34 encoder from `segmentation_models_pytorch`. The same training and evaluation pipeline supports RGB, synthetic multispectral, real multispectral, RGB plus synthetic multispectral, and RGB plus real multispectral input.

Config field documentation is available in `tbd\masking\CONFIG_REFERENCE.md`.

Supported model families in this repo:

- `Unet` with `resnet34`
- `UnetPlusPlus` with `resnet34` and `resnet50`
- `DeepLabV3Plus` with `resnet34` and `resnet50`
- `Segformer` with `mit_b0` and `mit_b1`

## Workflow

Run the commands below from `tbd/masking`. Activate the project virtual environment first, then set `PYTHONPATH` so the `src` package resolves correctly:

```bash
source ./.venv/bin/activate
```

Convenience scripts are available if you want one command for setup or training:

- `bash ./scripts/setup/setup_data.sh`
- `bash ./scripts/smoke/run_smoke.sh`
- `bash ./scripts/shared/train_configs.sh --config ...`
- `bash ./scripts/binary/train_rgb_architectures.sh`
- `bash ./scripts/shared/run_multi_seed.sh --config ...`
- `bash ./scripts/shared/run_multi_seed_baselines.sh --config ...`
- `bash ./scripts/shared/evaluate_base_configs.sh --config ...`
- `bash ./scripts/slurm/run_masking_batch.sh --manifest ...`
- `bash ./scripts/slurm/run_multi_seed_batch.sh --config ...`

1. Create aligned original-image splits:

```bash
python -m src.pack_real_msi --input-dir ./data/weedy-rice/Multispectral --output-dir ./data/weedy-rice/MultispectralNPY
python -m src.create_splits --dataset-root ./data/weedy-rice --output-dir ./data/splits --group-strategy datetime
```

Or run the full setup in one command:

```bash
bash ./scripts/setup/setup_data.sh
```

2. Create deterministic patch manifests for each modality config:

```bash
python -m src.patchify --config ./configs/binary/rgb.yaml
python -m src.patchify --config ./configs/binary/real_msi.yaml
python -m src.patchify --config ./configs/binary/synth_msi.yaml
python -m src.patchify --config ./configs/binary/rgb_real_msi.yaml
python -m src.patchify --config ./configs/binary/rgb_synth_msi.yaml
```

3. Train RGB only:

```bash
python -m src.train --config ./configs/binary/rgb.yaml
```

Optional fuzzy-halo RGB variant:

```bash
python -m src.train --config ./configs/fuzzy/rgb.yaml
```

4. Train RGB + synthetic multispectral:

```bash
python -m src.train --config ./configs/binary/rgb_synth_msi.yaml
```

5. Train RGB + real multispectral:

```bash
python -m src.train --config ./configs/binary/rgb_real_msi.yaml
```

Architecture-specific examples:

```bash
python -m src.train --config ./configs/binary/rgb_unetpp_resnet50.yaml
python -m src.train --config ./configs/binary/rgb_deeplabv3plus_resnet34.yaml
python -m src.train --config ./configs/binary/rgb_segformer_b1.yaml
```

Or train a preset architecture suite with one command:

```bash
bash ./scripts/binary/train_rgb_architectures.sh
```

Or train any list of configs with one command:

```bash
bash ./scripts/shared/train_configs.sh \
  --config configs/binary/rgb_unetpp_resnet34.yaml \
  --config configs/binary/rgb_deeplabv3plus_resnet50.yaml \
  --summary-output outputs/metrics/custom_train_summary.json
```

Train the full binary config family:

```bash
bash ./scripts/binary/train_all_binary.sh
```

Train the full fuzzy config family:

```bash
bash ./scripts/fuzzy/train_all_fuzzy.sh
```

Or run the full smoke pipeline with one command:

```bash
bash ./scripts/smoke/run_smoke.sh
```

Fuzzy-halo smoke variant:

```bash
bash ./scripts/smoke/run_smoke.sh --config configs/smoke/smoke_rgb_fuzzy_halo.yaml
```

Config layout:

- `./configs/binary/`: standard hard-mask training configs
- `./configs/fuzzy/`: fuzzy-halo variants that mirror the binary config names
- `./configs/smoke/`: smoke-test configs
- `./configs/base.yaml`: shared root defaults

Training writes run artifacts under `outputs/runs/<run_name>/`, including `logs/execution.log` with entries formatted as `%(asctime)s [Trainer] :: %(message)s`. At the end of training, the best validation checkpoint is evaluated once on the configured test split. The resulting test summary is written to `metrics/test_summary.json`, and reconstructed test prediction masks are written to `metrics/test_masks/` as binary PNG files. Per-epoch training and validation do not save masks. Every model is now test-scored against both the original hard mask and the matching fuzzy-halo mask, so `test_summary.json` contains explicit `original_*` and `fuzzy_*` metric sections alongside the compatibility aliases `patch_level`, `patch_summary`, and `image_level`.

Additional supported multimodal runs:

- `./configs/binary/rgb_synth_msi.yaml`: RGB concatenated with synthetic multispectral input, for a 7-channel model.
- `./configs/binary/rgb_real_msi.yaml`: RGB concatenated with real multispectral input, for a 7-channel model.

Architecture config naming:

- `<modality>_unetpp_resnet34.yaml`
- `<modality>_unetpp_resnet50.yaml`
- `<modality>_deeplabv3plus_resnet34.yaml`
- `<modality>_deeplabv3plus_resnet50.yaml`
- `<modality>_segformer_b0.yaml`
- `<modality>_segformer_b1.yaml`

6. Evaluate the best checkpoint:

```bash
python -m src.evaluate --checkpoint ./outputs/runs/<run>/checkpoints/best.pt --split test
```

Evaluate an unfine-tuned base model from config:

```bash
python -m src.evaluate_base --config ./configs/binary/rgb.yaml --split test
```

Evaluate all unfine-tuned binary baselines:

```bash
bash ./scripts/baseline/evaluate_all_binary_baselines.sh
```

Evaluate all unfine-tuned fuzzy baselines:

```bash
bash ./scripts/baseline/evaluate_all_fuzzy_baselines.sh
```

Evaluate all unfine-tuned baselines across multiple seeds and summarize them:

```bash
bash ./scripts/baseline/evaluate_all_binary_baselines_multi_seed.sh \
  --repeats 10 \
  --base-seed 1000 \
  --summary-output outputs/metrics/binary_baseline_multi_seed.json
```

The fuzzy and combined variants work the same way:

```bash
bash ./scripts/baseline/evaluate_all_fuzzy_baselines_multi_seed.sh --repeats 10 --base-seed 1000
bash ./scripts/baseline/evaluate_all_baselines_multi_seed.sh --repeats 10 --base-seed 1000
```

Run the full binary, fuzzy, and unfine-tuned baseline workload through Slurm with up to six one-GPU jobs active at a time:

```bash
bash ./scripts/slurm/run_masking_batch.sh --manifest scripts/slurm/workloads/all.json
```

The Slurm batching scripts also provide narrower manifests:

```bash
bash ./scripts/slurm/run_masking_batch.sh --manifest scripts/slurm/workloads/binary_train.json
bash ./scripts/slurm/run_masking_batch.sh --manifest scripts/slurm/workloads/fuzzy_train.json
bash ./scripts/slurm/run_masking_batch.sh --manifest scripts/slurm/workloads/binary_baseline.json
bash ./scripts/slurm/run_masking_batch.sh --manifest scripts/slurm/workloads/fuzzy_baseline.json
```

To run repeated training on Slurm without putting multiple seeds inside one long job, use the multi-seed Slurm wrapper. It expands each config into one training job per seed, then writes a combined summary after the batch drains:

```bash
bash ./scripts/slurm/run_multi_seed_batch.sh \
  --config configs/binary/rgb.yaml \
  --repeats 5 \
  --base-seed 1000
```

Or expand an existing train manifest:

```bash
bash ./scripts/slurm/run_multi_seed_batch.sh \
  --manifest scripts/slurm/workloads/binary_train.json \
  --repeats 5 \
  --base-seed 1000
```

To run repeated unfine-tuned baselines on Slurm, use the baseline multi-seed wrapper. It expands the selected baseline workload into one task per seed, submits the expanded manifest through the same controller used for training, keeps up to `--max-parallel` Slurm jobs active, and starts new jobs as earlier ones complete:

```bash
bash ./scripts/slurm/run_multi_seed_baseline_batch.sh \
  --manifest scripts/slurm/workloads/binary_baseline.json \
  --repeats 10 \
  --base-seed 1000 \
  --max-parallel 16
```

Use `scripts/slurm/workloads/fuzzy_baseline.json` instead for fuzzy baselines.

To split binary baseline multi-seed runs into non-synthetic and synthetic-only batches, run them as two separate controller invocations:

```bash
bash ./scripts/slurm/run_multi_seed_baseline_batch.sh \
  --manifest scripts/slurm/workloads/binary_baseline_no_synth.json \
  --repeats 10 \
  --base-seed 1000 \
  --max-parallel 16

bash ./scripts/slurm/run_multi_seed_baseline_batch.sh \
  --manifest scripts/slurm/workloads/binary_baseline_only_synth.json \
  --repeats 10 \
  --base-seed 1000 \
  --max-parallel 16
```

Evaluation writes artifacts under `outputs/runs/<run>/evaluation/<split>/`, including `overall_metrics.json`, per-patch and per-image CSV metrics, reconstructed predicted masks in `masks/`, limited preview panels in `visuals/`, and `execution.log` with entries formatted as `%(asctime)s [Evaluator] :: %(message)s`. Every evaluated model now produces both `original_*` and `fuzzy_*` metrics. Binary and baseline configs use the matching file in `configs/fuzzy/` as the fuzzy scoring reference, while fuzzy configs use their own halo settings.

`overall_metrics.json` fields are interpreted as follows:

- `original_patch_level`, `original_patch_summary`, `original_image_level`: pooled patch score, per-patch summary, and reconstructed-image summary against the original hard weed mask
- `fuzzy_patch_level`, `fuzzy_patch_summary`, `fuzzy_image_level`: the same three views scored against the fuzzy-halo mask from the matching config in `configs/fuzzy/`
- `patch_level`, `patch_summary`, `image_level`: compatibility aliases that point to the model's own target view
  - binary and baseline runs alias to `original_*`
  - fuzzy-halo runs alias to `fuzzy_*`

In practice, `original_image_level` and `fuzzy_image_level` are the best headline full-scene comparisons, `patch_level` remains a convenient legacy alias for the model's own target view, and `patch_summary` is useful when you want the distribution across individual patches.

Baseline evaluation writes the same evaluation artifacts into a baseline-named run directory under `outputs/runs/`, plus `config.yaml` and `run_metadata.json` with `run_kind: baseline`. It does not create checkpoints or training history.

7. Summarize multiple runs:

```bash
python -m src.summarize --runs ./outputs/runs/<rgb_run> ./outputs/runs/<synth_run> ./outputs/runs/<real_run> --output ./outputs/metrics/final_comparison.json
```

Summarization writes a JSON file to the requested `--output` path with two top-level sections: `runs` for per-run metrics and `aggregates` for per-model aggregate metrics. Each aggregate row now includes median, mean, and standard deviation fields for every tracked metric. All model families include both `original_test_*` and `fuzzy_test_*` metrics so hard-mask and fuzzy-mask scoring can be compared directly. It also writes a companion log file next to it, for example `outputs/metrics/final_comparison.log`, with entries formatted as `%(asctime)s [Summarizer] :: %(message)s`.

Confidence-style metrics are also included in the summary. They are derived from the sigmoid output probabilities and indicate how certain the model was about its predicted mask values:

- `test_patch_confidence` and `test_image_confidence`: mean certainty across all predicted pixels, computed from `max(p, 1 - p)`
- `test_patch_positive_confidence` and `test_image_positive_confidence`: mean predicted probability on pixels predicted as weed
- `test_patch_negative_confidence` and `test_image_negative_confidence`: mean `1 - p` on pixels predicted as background

These values are useful as model-certainty indicators, but they are not calibrated probabilities of correctness.

## Multi-Seed Runs

To estimate average performance, train the same config multiple times with different seeds and summarize the resulting runs.

Example for one model:

```bash
bash ./scripts/shared/run_multi_seed.sh --config configs/binary/rgb.yaml --repeats 5 --base-seed 1000 --summary-output outputs/metrics/rgb_multi_seed.json
```

Example for multiple models in one shell:

```bash
bash ./scripts/shared/run_multi_seed.sh \
  --config configs/binary/rgb_unetpp_resnet34.yaml \
  --config configs/binary/rgb_deeplabv3plus_resnet34.yaml \
  --config configs/binary/rgb_segformer_b0.yaml \
  --repeats 5 \
  --base-seed 1000 \
  --summary-output outputs/metrics/rgb_architectures_multi_seed.json
```

Baseline-only multi-seed evaluation follows the same seed schedule, but uses `src.evaluate_base` instead of training:

```bash
bash ./scripts/shared/run_multi_seed_baselines.sh \
  --config configs/binary/rgb.yaml \
  --config configs/binary/real_msi.yaml \
  --repeats 10 \
  --base-seed 1000 \
  --summary-output outputs/metrics/baseline_multi_seed.json
```

To spread models across different shells, launch the same script in each shell with a different config list.

The script:

- uses the same deterministic seed schedule for every model
- trains the model, including the training-time test metrics and test mask export
- evaluates the best checkpoint on the test split to produce the full evaluation artifacts used by summarization
- writes a JSON summary with per-run metrics
- includes aggregate per-model mean and standard deviation across runs

The baseline multi-seed script:

- uses the same deterministic seed schedule for every model
- evaluates unfine-tuned base models with different random initialization seeds
- writes the same style of summary JSON when run on the `test` split
- skips summary on non-`test` splits because `src.summarize` reads `evaluation/test/overall_metrics.json`

Single-run convenience scripts:

- `scripts/setup/setup_data.sh`: packs real MSI `.TIF` bands, creates split manifests, and patchifies all five modality baselines
- `scripts/smoke/run_smoke.sh`: patchifies the checked-in smoke split, trains the smoke config, runs test evaluation, and writes a one-run summary
- `scripts/shared/train_configs.sh`: trains and evaluates any explicit list of config files once each
- `scripts/shared/run_multi_seed.sh`: runs repeated training and evaluation across explicit config lists
- `scripts/shared/run_multi_seed_baselines.sh`: runs repeated unfine-tuned baseline evaluation across explicit config lists
- `scripts/shared/evaluate_base_configs.sh`: evaluates one or more unfine-tuned base models from config and can summarize the baseline run set
- `scripts/binary/train_rgb_architectures.sh`: trains the six requested RGB architecture variants once each and summarizes them
- `scripts/binary/train_all_binary.sh`: trains the full binary config family
- `scripts/fuzzy/train_all_fuzzy.sh`: trains the full fuzzy config family
- `scripts/baseline/evaluate_all_binary_baselines.sh`: evaluates the full unfine-tuned binary baseline family
- `scripts/baseline/evaluate_all_fuzzy_baselines.sh`: evaluates the full unfine-tuned fuzzy baseline family
- `scripts/baseline/evaluate_all_binary_baselines_multi_seed.sh`: evaluates the full binary baseline family across repeated seeds and summarizes it
- `scripts/baseline/evaluate_all_fuzzy_baselines_multi_seed.sh`: evaluates the full fuzzy baseline family across repeated seeds and summarizes it
- `scripts/baseline/evaluate_all_baselines_multi_seed.sh`: evaluates the combined binary and fuzzy baseline architecture families across repeated seeds and summarizes them
- `scripts/slurm/run_masking_batch.sh`: runs a manifest through Slurm, keeping up to six one-GPU jobs active and validating each completed job before starting more work
- `scripts/slurm/run_multi_seed_batch.sh`: expands repeated training into one Slurm job per seed, reuses the batch controller, and writes a combined multi-seed summary
- `scripts/slurm/run_multi_seed_baseline_batch.sh`: expands a baseline manifest into one Slurm job per seed and reuses the batch controller
- `scripts/slurm/masking_job.sh`: the Slurm job submitted by the batch controller for one train or baseline task
- `scripts/slurm/read_manifest.py`: validates JSON workload manifests and emits the normalized task rows consumed by the batch controller
- `scripts/slurm/write_multi_seed_manifest.py`: expands configs, train manifests, or baseline manifests into one task per seed
- `scripts/slurm/collect_successful_runs.py`: reads Slurm status files and emits successful run directories for final summary generation

Slurm workload manifests:

- `scripts/slurm/workloads/all.json`: binary training, fuzzy training, binary baselines, and fuzzy baselines
- `scripts/slurm/workloads/binary_train.json`: the same configs as `scripts/binary/train_all_binary.sh`
- `scripts/slurm/workloads/fuzzy_train.json`: the same configs as `scripts/fuzzy/train_all_fuzzy.sh`
- `scripts/slurm/workloads/binary_baseline.json`: the same configs as `scripts/baseline/evaluate_all_binary_baselines.sh`
- `scripts/slurm/workloads/binary_baseline_no_synth.json`: binary baseline configs without `synth` in the config path
- `scripts/slurm/workloads/binary_baseline_only_synth.json`: binary baseline configs with synthetic data
- `scripts/slurm/workloads/fuzzy_baseline.json`: the same configs as `scripts/baseline/evaluate_all_fuzzy_baselines.sh`

The manifest format is JSON with a top-level `tasks` list. Each task has `group`, `kind`, `config`, and `split` fields, and may also include an optional `seed` for explicit per-run training or baseline tasks:

```json
{
  "tasks": [
    {
      "group": "binary_train",
      "kind": "train",
      "config": "configs/binary/rgb_unetpp_resnet34.yaml",
      "split": "test",
      "seed": 12345
    }
  ]
}
```

The Slurm controller is intended to run from a login shell in `tbd/masking`. It submits individual `sbatch` jobs, keeps `--max-parallel 6` jobs active by default, retries failed or suspicious tasks according to the `MAX_RETRIES` variable near the top of `scripts/slurm/run_masking_batch.sh`, and then writes per-group summaries to `outputs/metrics/slurm_<manifest>_<group>_summary.json`. Each batch status directory contains `controller.log`, per-task status files, per-attempt job logs referenced by `job_log=...`, and `failed_tasks.tsv` when failures occur. Slurm stdout/stderr still go under `logs/masking/slurm/`. It also creates an atomic controller lock under `outputs/slurm/status/.masking_batch_controller.lock` so two controllers are not accidentally started against the same GPU pool. You may optionally pass `--status-dir` when another wrapper, such as the multi-seed runner, needs a deterministic status directory.

The multi-seed Slurm runner writes its expanded manifest under `outputs/slurm/manifests/`, reuses the same task validation and retry logic as `run_masking_batch.sh`, and writes a final combined summary JSON after collecting successful run directories from the batch status files. It still runs one training seed per Slurm job, so `--repeats 10` means ten separate jobs per config rather than one long job. For baseline multi-seed runs, use `run_multi_seed_baseline_batch.sh`; it wraps the same manifest expansion and controller submission path.

To rerun a retry manifest, such as one produced by `scripts/slurm/find_failed_tasks.py`, use:

```bash
bash ./scripts/slurm/run_retry_manifest.sh \
  --manifest retry-manifest.json \
  --runs-root outputs/runs \
  --summary-output outputs/metrics/all_no_synth_combined_after_retry.json
```

The retry wrapper submits the manifest through the normal Slurm controller without expanding seeds. After the controller finishes, it scans `outputs/runs` for completed runs with `config.yaml` and `evaluation/test/overall_metrics.json`, writes a combined summary, and writes a companion `*_runs.txt` file listing exactly which run directories were included. If some retry tasks still fail, the combined summary is still written from completed runs and the wrapper exits with the controller's nonzero status.

Cluster defaults are editable near the top of `scripts/slurm/run_masking_batch.sh` and in the SBATCH header of `scripts/slurm/masking_job.sh`. Set `MAX_RETRIES=10` or `MAX_RETRIES=50` in the controller if you want more retries after bad logs or missing outputs; set `SBATCH_SUBMIT_RETRIES` separately for cases where `sbatch` does not return a job id. The default job settings request one GPU, 15 CPUs, 24 GB memory, and 12 hours, and run through `/ceph/container/pytorch/pytorch_26.02.sif`. The job script tries `p10_venv/bin/activate`, `../../p10_venv/bin/activate`, and `../../.venv/bin/activate` from `tbd/masking`; override `VENV_ACTIVATE` if the cluster venv lives elsewhere. Use `--dry-run` to verify manifest parsing without submitting jobs:

```bash
bash ./scripts/slurm/run_masking_batch.sh --manifest scripts/slurm/workloads/all.json --dry-run
```

If you pass `--skip-evaluate`, the shared training scripts train only. Training still writes `metrics/test_summary.json` and `metrics/test_masks/`, but the script does not run `src.evaluate` and does not generate the final summary JSON, because summarization reads `evaluation/test/overall_metrics.json`.

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
export PYTHONPATH=.
python -m src.train --config ./configs/binary/rgb.yaml
```

Train RGB + synthetic multispectral:

```bash
export PYTHONPATH=.
python -m src.train --config ./configs/binary/rgb_synth_msi.yaml
```

Train RGB + real multispectral:

```bash
export PYTHONPATH=.
python -m src.train --config ./configs/binary/rgb_real_msi.yaml
```

Evaluate a run:

```bash
export PYTHONPATH=.
python -m src.evaluate --checkpoint ./outputs/runs/<run>/checkpoints/best.pt --split test
```

Evaluate an unfine-tuned base run:

```bash
export PYTHONPATH=.
python -m src.evaluate_base --config ./configs/binary/rgb.yaml --split test
```

Create a summary table:

```bash
export PYTHONPATH=.
python -m src.summarize --runs ./outputs/runs/<rgb_run> ./outputs/runs/<synth_run> ./outputs/runs/<real_run> --output ./outputs/metrics/final_comparison.json
```

Run repeated multi-seed experiments:

```bash
bash ./scripts/shared/run_multi_seed.sh \
  --config configs/binary/rgb_unetpp_resnet34.yaml \
  --config configs/binary/rgb_deeplabv3plus_resnet34.yaml \
  --config configs/binary/rgb_segformer_b0.yaml \
  --repeats 5 \
  --base-seed 1000 \
  --summary-output outputs/metrics/rgb_architectures_multi_seed.json
```

Run the full smoke pipeline:

```bash
bash ./scripts/smoke/run_smoke.sh
```

## Notes

- Splits are created at the original-image level before patching.
- `scripts/smoke/run_smoke.sh` uses the checked-in `smoke_train.csv`, `smoke_val.csv`, and `smoke_test.csv` manifests rather than regenerating smoke splits.
- Patch manifests are deterministic given the config seed.
- RGB normalization uses ImageNet statistics.
- All other modalities, including the two RGB+MSI combined variants, compute normalization from train patches only and cache it as JSON.
- The optional `fuzzy_halo` target variant keeps the labeled weed mask as the core region and adds a soft halo only outside the labeled boundary.
- Evaluation reconstructs full-image predictions by averaging patch probabilities into the original image canvas.
- Training-time test evaluation saves reconstructed binary prediction masks under `metrics/test_masks/`.
- Standalone evaluation saves reconstructed binary prediction masks under `evaluation/<split>/masks/`.
- `synthetic_msi_path` is expected to point to a single `.npy` or `.npz` file per sample.
- `real_msi_path` is expected to point to a single packed `.npy` file per sample. Use `python -m src.pack_real_msi` to convert legacy per-band `.TIF` files into this format.
