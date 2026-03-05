#!/bin/bash
#SBATCH --job-name=pack_patches
#SBATCH --output=logs/pack_patches_%j.out
#SBATCH --error=logs/pack_patches_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=02:00:00

hostname; date
mkdir -p logs

DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-patches}"

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE \
        python -u $SLURM_SUBMIT_DIR/utils/pack_patches.py \
            --patch-dir $DATA_ROOT \
            --workers 15"

date