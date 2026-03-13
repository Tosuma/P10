#!/bin/bash
#SBATCH --job-name=train_mami_weedy
#SBATCH --output=logs/train_mami_weedy_job.out
#SBATCH --error=logs/train_mami_weedy_job.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

mkdir -p logs
hostname
date

# Keep thread-heavy libs from oversubscribing CPU cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Torchrun needs to know how many processes to launch (one per GPU)
GPUS=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}

#singularity exec --nv \
#    /ceph/container/pytorch/pytorch_26.01.sif \
#    /bin/bash -lc "source p10_venv/bin/activate && \
#        python -u -m torch.distributed.run \
#            --standalone \
#            --nproc_per_node=${GPUS} \
#            mami/tl-pipeline.py \
#                --stage1_data_path data/East-Kaza \
#                --stage1_data_type Kazakhstan \
#                --stage1_epochs 0 \
#                --stage1_lr 4e-4 \
#                --stage2_data_path data/WeedyRice \
#                --stage2_data_type Weedy-Rice \
#                --stage2_model checkpoints/basemodel-kaz-ndvi-new2/stage1_best_final.pth \
#                --stage2_epochs 300 \
#                --stage2_lr 1e-5 \
#                --stage3_epochs 0 \
#                --loss_w_mrae 1.0 \
#                --loss_w_ndvi 0.1 \
#                --loss_w_ndre 0.1 \
#                --cluster \
#                --dir_name vi/weedy-stage2 \
#                --model_name vi-stage2-weedy"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.01.sif \
    /bin/bash -lc "source p10_venv/bin/activate && \
        python -u -m torch.distributed.run \
            --standalone \
            --nproc_per_node=${GPUS} \
            mami/tl-pipeline.py \
                --stage1_epochs 0 \
                --stage2_epochs 0 \
                --stage3_data_path data/WeedyRice \
                --stage3_data_type Weedy-Rice \
                --stage3_model checkpoints/vi/weedy-stage2/vi-stage2-weedy.pth \
                --stage3_epochs 300 \
                --stage3_lr 1e-7 \
                --loss_w_mrae 1.0 \
                --loss_w_ndvi 0.1 \
                --loss_w_ndre 0.1 \
                --cluster \
                --dir_name vi/weedy-stage3 \
                --model_name vi-stage3-weedy"

date
