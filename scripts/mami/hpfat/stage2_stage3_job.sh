#!/bin/bash
#SBATCH --job-name=train_mami_stage23
#SBATCH --output=logs/hpfat/stage23/train_mami_stage23_%j.out
#SBATCH --error=logs/hpfat/stage23/train_mami_stage23_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

set -euo pipefail

stage2_model=""
dir_name=""
model_name=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage2_model) stage2_model="$2"; shift 2 ;; 
        --dir_name)     dir_name="$2"; shift 2 ;; 
        --model_name)   model_name="$2"; shift 2 ;; 
        -h|--help)
            cat <<EOF
Usage: $0 --stage2_model PATH --dir_name DIR --model_name MODEL

This script launches stage2 and stage3 in one mami.py command.
Stage2 uses the provided stage2 model, and stage3 inherits stage2 best model automatically.
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

: "${stage2_model:?Missing --stage2_model}"
: "${dir_name:?Missing --dir_name}"
: "${model_name:?Missing --model_name}"

mkdir -p logs/hpfat/stage23
hostname
date

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
GPUS=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}

singularity exec --nv \
    -B /ceph/project/tbd/data/:/ceph/project/tbd/data \
    /ceph/container/pytorch/pytorch_26.01.sif \
    /bin/bash -lc "source p10_venv/bin/activate && \
        python -u -m torch.distributed.run \
            --standalone \
            --nproc_per_node=${GPUS} \
            mami/mami.py \
                --stage1_epochs 0 \
                --stage2_epochs 300 \
                --stage2_model ${stage2_model} \
                --stage2_data_path data/data/sri-lanka-aligned \
                --stage2_data_type Sri-Lanka \
                --stage2_lr 1e-5 \
                --stage2_loss_mrae_w 1.0 \
                --stage2_loss_ndvi_w 0.0 \
                --stage2_loss_ndre_w 0.0 \
                --stage3_epochs 300 \
                --stage3_data_path data/data/sri-lanka-aligned \
                --stage3_data_type Sri-Lanka \
                --stage3_lr 1e-7 \
                --stage3_loss_mrae_w 1.0 \
                --stage3_loss_ndvi_w 0.2 \
                --stage3_loss_ndre_w 0.1 \
                --dir_name ${dir_name} \
                --model_name ${model_name} \
                --cluster"

date
