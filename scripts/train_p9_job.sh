#!/bin/bash
#SBATCH --job-name=train_p9
#SBATCH --output=logs/train_p9_job.out
#SBATCH --error=logs/train_p9_job.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p logs
hostname
date

# Keep thread-heavy libs from oversubscribing CPU cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Torchrun needs to know how many processes to launch (one per GPU)
GPUS=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.01.sif \
    /bin/bash -lc "source my_venv/bin/activate && \
        python -u -m torch.distributed.run \
            --standalone \
            --nproc_per_node=${GPUS} \
            src/tl-pipeline_ddp.py \
                --stage1_data_path data/East-Kaza \
                --stage1_data_type Kazakhstan \
                --stage1_epochs 300 \
                --stage1_lr 4e-4 \
                --stage2_epochs 0 \
                --stage3_epochs 0 \
                --cluster"

date
