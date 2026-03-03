#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --output=logs/mae_pretrain_%j.out
#SBATCH --error=logs/mae_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 4 GPUs per node
#SBATCH --mem=24G
#SBATCH --time=12:00:00

mkdir -p logs outputs/stage1_mae

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/12.1
module load python/3.11

# Activate your virtual environment
source "$HOME/venvs/p10/bin/activate"

# Navigate to project root (scripts must be submitted from project root via sbatch)
cd "$SLURM_SUBMIT_DIR"

# Add project root to PYTHONPATH so `from src.*` imports resolve
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# Create log directories

# ── DDP via torchrun ─────────────────────────────────────────────────────────
# torchrun handles process spawning and MASTER_ADDR/PORT setup automatically.
# --nproc_per_node must match --ntasks-per-node above.
#
# --config-path uses an absolute path so Hydra finds configs/ at the project
# root regardless of where the script file lives (tbd/mae/train_mae.py).

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    tbd/mae/train_mae.py \
        --config-path "$SLURM_SUBMIT_DIR/configs" \
        data.rgb_dir="$DATA_ROOT/RGB" \
        data.ms_dir="$DATA_ROOT/Multispectral" \
        data.batch_size=64 \
        data.num_workers=8 \
        mae.epochs=200 \
        mae.use_checkpoint=true \
        mae.use_wandb=true

# Note: DATA_ROOT should be set in your environment or substituted here.
# Example: export DATA_ROOT=/scratch/$USER/drone_data
