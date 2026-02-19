#!/bin/bash

#SBATCH --job-name=train_p9
#SBATCH --output=logs/train_p9_job.out
#SBATCH --error=logs/train_p9_job.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

mkdir -p logs
hostname

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.01.sif \
    /bin/bash -c "echo 'venving' && source my_venv/bin/activate && echo 'pythoning' && \
        python src/tl-pipeline.py \
        --stage1_data_path data/East-Kaza \
        --stage1_data_type Kazakhstan \
        --stage1_epochs 300 \
        --stage1_lr 4e-4 \
        --stage2_epochs 0 \
        --stage3_epochs 0 \
        --cluster"

date