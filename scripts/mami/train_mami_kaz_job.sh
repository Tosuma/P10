#!/bin/bash
#SBATCH --job-name=train_mami
#SBATCH --output=logs/vi/train_mami_%j.out
#SBATCH --error=logs/vi/train_mami_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

# -------------------------
# Parse command-line args
# -------------------------
lr=""
mrae=""
ndvi=""
ndre=""
dir_name=""
model_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lr)             lr="$2"; shift 2 ;;
    --loss_mrae_w)    mrae="$2"; shift 2 ;;
    --loss_ndvi_w)    ndvi="$2"; shift 2 ;;
    --loss_ndre_w)    ndre="$2"; shift 2 ;;
    --dir_name)       dir_name="$2"; shift 2 ;;
    --model_name)     model_name="$2"; shift 2 ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      echo "Unexpected positional arg: $1" >&2
      exit 2
      ;;
  esac
done

# Basic required-arg checks (optional but recommended)
: "${lr:?Missing --lr}"
: "${mrae:?Missing --loss_mrae_w}"
: "${ndvi:?Missing --loss_ndvi_w}"
: "${ndre:?Missing --loss_ndre_w}"
: "${dir_name:?Missing --dir_name}"
: "${model_name:?Missing --model_name}"

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
    /bin/bash -lc "source p10_venv/bin/activate && \
        python -u -m torch.distributed.run \
            --standalone \
            --nproc_per_node=${GPUS} \
            mami/mami.py \
                --stage1_data_path data/East-Kaza \
                --stage1_data_type Kazakhstan \
                --stage1_epochs 300 \
                --stage1_lr ${lr} \
                --stage1_loss_mrae_w ${mrae} \
                --stage1_loss_ndvi_w ${ndvi} \
                --stage1_loss_ndre_w ${ndre} \
                --dir_name ${dir_name} \
                --model_name ${model_name} \
                --cluster"

date
