#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

hostname; date
mkdir -p logs results/evaluation

DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-patches}"
MAE_CKPT="${MAE_CKPT:-$SLURM_SUBMIT_DIR/checkpoints/mae/best.pth}"
FLOW_CKPT="${FLOW_CKPT:-$SLURM_SUBMIT_DIR/checkpoints/flow/best.pth}"

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE \
        python -u $SLURM_SUBMIT_DIR/evaluate.py \
            data.patch_dir=$DATA_ROOT \
            mae_checkpoint=$MAE_CKPT \
            flow_checkpoint=$FLOW_CKPT \
            output_dir=$SLURM_SUBMIT_DIR/results/evaluation"

date
