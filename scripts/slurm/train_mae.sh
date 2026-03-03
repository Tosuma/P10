#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --output=logs/mae_pretrain_%j.out
#SBATCH --error=logs/mae_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

mkdir -p logs outputs/stage1_mae

# ── Environment ───────────────────────────────────────────────────────────────
# Load CUDA drivers from host (needed even with containers for GPU access)
module load cuda/12.1

# Navigate to project root
cd "$SLURM_SUBMIT_DIR"

# Add project root to PYTHONPATH — passed through into the container automatically
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CONTAINER="/ceph/container/pytorch/pytorch_26_02.sif"

# ── DDP via torchrun inside Singularity container ────────────────────────────
# --nv   : passes through NVIDIA GPU drivers from the host
# --bind : makes the project directory visible inside the container
#
# torchrun is called inside the container so it uses the container's Python/torch.

singularity exec --nv \
    --bind "$SLURM_SUBMIT_DIR" \
    "$CONTAINER" \
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

# Note: set DATA_ROOT before submitting, e.g.:
#   export DATA_ROOT=/ceph/data/drone && sbatch scripts/slurm/train_mae.sh