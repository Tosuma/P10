#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

hostname; date
mkdir -p logs results/baselines

DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-patches}"

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE \
        python -u $SLURM_SUBMIT_DIR/run_baselines.py \
            data.patch_dir=$DATA_ROOT \
            output_dir=$SLURM_SUBMIT_DIR/results/baselines"

date
