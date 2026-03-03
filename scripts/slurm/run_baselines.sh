#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Run all baselines (CPU-heavy; PatchCore benefits from GPU)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                  # For PatchCore ResNet50 feature extraction
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu

module load cuda/12.1
module load python/3.11
source "$HOME/venvs/p10/bin/activate"
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs outputs/baselines

python run_baselines.py \
    data.rgb_dir="$DATA_ROOT/RGB" \
    data.ms_dir="$DATA_ROOT/Multispectral" \
    data.batch_size=64 \
    data.num_workers=8 \
    baselines.output_dir="outputs/baselines"
