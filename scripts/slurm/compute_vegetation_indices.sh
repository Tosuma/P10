#!/bin/bash
#SBATCH --job-name=compute_vi
#SBATCH --output=logs/compute_vi_%j.out
#SBATCH --error=logs/compute_vi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00

hostname
date

mkdir -p logs

# ---- Paths ---------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-RGBMS-DB}"

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

# ---- Run -----------------------------------------------------------------
singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE python -u \
        $SLURM_SUBMIT_DIR/utils/compute_vegetation_indices.py \
            --data-root $DATA_ROOT \
            --workers   $SLURM_CPUS_PER_TASK"

date