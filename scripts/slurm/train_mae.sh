#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --output=logs/mae_pretrain%j.out
#SBATCH --error=logs/mae_pretrain%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

hostname
date

DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-RGBMS-DB/}"

# Each DDP process must use 1 OMP thread — with 4 processes, setting this to
# SLURM_CPUS_PER_TASK (32) would spawn 128 OMP threads competing for 32 CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

GPUS=4  # matches #SBATCH --gres=gpu:4
echo "Using $GPUS GPU(s)"

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE python -u -m torch.distributed.run \
            --standalone \
            --nproc_per_node=${GPUS} \
            tbd/mae/train_mae.py \
                --config-path $SLURM_SUBMIT_DIR/configs \
                data.rgb_dir=$DATA_ROOT/RGB \
                data.ms_dir=$DATA_ROOT/Multispectral \
                data.batch_size=512 \
                data.num_workers=7 \
                data.cache_images=true \
                mae.epochs=200 \
                mae.arch=vit_small_patch16 \
                mae.base_lr=6e-4 \
                mae.use_checkpoint=true \
                mae.use_wandb=true"

date