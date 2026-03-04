#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --output=logs/mae_pretrain.out
#SBATCH --error=logs/mae_pretrain.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

hostname
date

CEPH_DATA="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-RGBMS-DB/}"

# ── Stage data from Ceph to local /tmp (eliminates network FS bottleneck) ──
# Reading 128×128 patches from Ceph on every __getitem__ starves the GPUs.
# Copying once to local SSD/RAM makes data loading ~10× faster.
LOCAL_DATA="/tmp/mae_data_${SLURM_JOB_ID}"
echo "Staging data from Ceph to ${LOCAL_DATA} ..."
mkdir -p "${LOCAL_DATA}/RGB" "${LOCAL_DATA}/Multispectral"
rsync -a --no-progress "${CEPH_DATA}/RGB/" "${LOCAL_DATA}/RGB/"
rsync -a --no-progress "${CEPH_DATA}/Multispectral/" "${LOCAL_DATA}/Multispectral/"
echo "Data staging complete: $(du -sh ${LOCAL_DATA} | cut -f1)"

# Each DDP process must use 1 OMP thread — with 4 processes, setting this to
# SLURM_CPUS_PER_TASK (32) would spawn 128 OMP threads competing for 32 CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

GPUS=4  # matches #SBATCH --gres=gpu:4
echo "Using $GPUS GPU(s)"

# Use the container's Python/torch (2.6.x) to avoid version conflicts.
# The venv's site-packages are added to PYTHONPATH so that extra dependencies
# (hydra, timm, einops, wandb, etc.) are found without activating the venv
# and pulling in the venv's older torch (2.3.1), which would segfault against
# the container's NCCL/CUDA libraries.
VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE python -u -m torch.distributed.run \
            --standalone \
            --nproc_per_node=${GPUS} \
            tbd/mae/train_mae.py \
                --config-path $SLURM_SUBMIT_DIR/configs \
                data.rgb_dir=${LOCAL_DATA}/RGB \
                data.ms_dir=${LOCAL_DATA}/Multispectral \
                data.batch_size=512 \
                data.num_workers=7 \
                data.cache_images=true \
                mae.epochs=200 \
                mae.arch=vit_small_patch16 \
                mae.base_lr=6e-4 \
                mae.use_checkpoint=true \
                mae.use_wandb=true"

# Clean up local staging area
rm -rf "${LOCAL_DATA}"

date