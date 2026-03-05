#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --output=logs/mae_pretrain_%j.out
#SBATCH --error=logs/mae_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15          # 4 GPUs × 15 CPUs/GPU
#SBATCH --mem=24G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

hostname; date
mkdir -p logs

# Override data location on HPC
DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-patches}"

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
                data.batch_size=256 \
                mae.arch=vit_small_patch16 \
                mae.use_checkpoint=true \
                logging.use_wandb=true"

date
