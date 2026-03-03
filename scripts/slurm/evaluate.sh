#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Inference + evaluation (generate heatmaps, UMAP, score histograms)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=evaluate
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

module load cuda/12.1
module load python/3.11
source "$HOME/venvs/p10/bin/activate"
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs outputs/heatmaps outputs/figures

python evaluate.py \
    data.rgb_dir="$DATA_ROOT/RGB" \
    data.ms_dir="$DATA_ROOT/Multispectral" \
    data.batch_size=64 \
    flow.mae_checkpoint="outputs/stage1_mae/mae_best.pth" \
    flow.output_dir="outputs/stage2_flow" \
    flow.heatmap_output_dir="outputs/heatmaps" \
    flow.export_geotiff=true
