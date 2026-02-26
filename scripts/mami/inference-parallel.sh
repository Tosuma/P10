#!/bin/bash
#SBATCH --job-name=infer_mami
#SBATCH --output=logs/infer_mami_job.out
#SBATCH --error=logs/infer_mami_job.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

mkdir -p logs
hostname
date

# Keep thread-heavy libs from oversubscribing CPU cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Torchrun needs to know how many processes to launch (one per GPU)
GPUS=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}

# ---- Edit these arguments for your run ----
sri_path="./data/sri-lanka-aligned/"
sri_type="Sri-Lanka"
weedy_path="./data/WeedyRice/"
weedy_type="Weedy-Rice"
kaz_path="data/East-Kaza"
kaz_type="Kazakhstan"

MODEL_PATH="checkpoints/basemodel-kaz-ndvi-new/stage1_best_final.pth"
SAVE_DIR="results/300/basemodel-new-ndvi---sri-lanka/"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.01.sif \
    /bin/bash -lc "source p10_venv/bin/activate && \
        python -u -m torch.distributed.run \
            --standalone \
            --nproc_per_node=${GPUS} \
            mami/inference.py \
                --data_path ${sri_path} \
                --data_type ${sri_type} \
                --model ${MODEL_PATH} \
                --save_path ${SAVE_DIR} \
                --save_images"

date
