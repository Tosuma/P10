#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time job: pre-build the image shape cache so train_mae.sh starts
# instantly instead of spending 13+ minutes opening rasterio files on Ceph.
# Run this ONCE before your first training job. Never needs to run again
# unless you change the dataset.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=build_cache
#SBATCH --output=logs/build_cache_%j.out
#SBATCH --error=logs/build_cache_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

hostname
date

mkdir -p logs

DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-RGBMS-DB/}"

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE python -u \
        scripts/build_shape_cache.py \
            $DATA_ROOT/RGB \
            $DATA_ROOT/Multispectral"

date