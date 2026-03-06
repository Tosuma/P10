#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --output=logs/mae_pretrain_%j.out
#SBATCH --error=logs/mae_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60          # 4 GPUs × 15 CPUs/GPU
#SBATCH --mem=24G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

hostname; date
mkdir -p logs

# Override data location on HPC
DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-patches}"

# Copy Packed/ + RGB/ to node-local /tmp for near-zero metadata latency.
#   Packed/ : ~66 GB (single .npz per patch — 1 file open per sample)
#   RGB/    : ~2  GB (JPEG stems — used only to collect patch filenames)
#   Total   : ~68 GB — check free space before copying.
LOCAL_DATA="/tmp/mae_data_${SLURM_JOB_ID}"
FREE_KB=$(df /tmp | tail -1 | awk '{print $4}')
if [ -d "$DATA_ROOT/Packed" ] && [ "$FREE_KB" -gt 70000000 ]; then
    echo "Copying RGB/ and Packed/ to local /tmp (${FREE_KB} KB free) ..."
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA_ROOT/RGB"    "$LOCAL_DATA/RGB"
    cp -r "$DATA_ROOT/Packed" "$LOCAL_DATA/Packed"
    DATA_ROOT="$LOCAL_DATA"
    echo "Local copy done — training will read from $LOCAL_DATA"
else
    echo "Skipping local copy (Packed/ missing or <70 GB free in /tmp: ${FREE_KB} KB)."
fi

# Run identity — each job gets a unique dir by default (SLURM_JOB_ID).
# Override at submit time to use a meaningful name instead:
#   RUN_NAME=mae_v2 OUTPUT_DIR=checkpoints/mae_v2 sbatch scripts/slurm/train_mae.sh
RUN_NAME="${RUN_NAME:-mae_${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/mae_${SLURM_JOB_ID}}"

# Each DDP process gets 1 OMP thread to avoid thread contention
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

GPUS=4
VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE \
        torchrun \
            --standalone \
            --nproc_per_node=${GPUS} \
            $SLURM_SUBMIT_DIR/train_mae.py \
                data.patch_dir=$DATA_ROOT \
                data.num_workers=15 \
                data.batch_size=1536 \
                mae.arch=vit_small_patch16 \
                mae.use_checkpoint=false \
                logging.use_wandb=true \
                logging.run_name=$RUN_NAME \
                logging.output_dir=$OUTPUT_DIR"

date
