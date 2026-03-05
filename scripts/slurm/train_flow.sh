#!/bin/bash
#SBATCH --job-name=flow_train
#SBATCH --output=logs/flow_train_%j.out
#SBATCH --error=logs/flow_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

hostname; date
mkdir -p logs checkpoints/flow

DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-patches}"
MAE_CKPT="${MAE_CKPT:-$SLURM_SUBMIT_DIR/checkpoints/mae/best.pth}"

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE \
        python -u $SLURM_SUBMIT_DIR/train_flow.py \
            data.patch_dir=$DATA_ROOT \
            mae_checkpoint=$MAE_CKPT \
            flow.cache_features=true \
            logging.use_wandb=true"

date
