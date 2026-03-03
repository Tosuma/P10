#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Stage 2 FastFlow Training (single GPU)
#
# Stage 2 is much lighter than Stage 1 (frozen encoder, small flow model).
# A single GPU is sufficient.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=flow_train
#SBATCH --output=logs/flow_train_%j.out
#SBATCH --error=logs/flow_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

module load cuda/12.1
module load python/3.11
source "$HOME/venvs/p10/bin/activate"
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs outputs/stage2_flow

python train_flow.py \
    data.rgb_dir="$DATA_ROOT/RGB" \
    data.ms_dir="$DATA_ROOT/Multispectral" \
    data.batch_size=128 \
    data.num_workers=8 \
    flow.mae_checkpoint="outputs/stage1_mae/mae_best.pth" \
    flow.epochs=100 \
    flow.use_wandb=true
