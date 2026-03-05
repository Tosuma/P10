#!/bin/bash
#SBATCH --job-name=patch_weedyrice
#SBATCH --output=logs/patch_weedyrice_%j.out
#SBATCH --error=logs/patch_weedyrice_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00

hostname
date

mkdir -p logs

# ---- Paths ---------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-RGBMS-DB}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-patches}"

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

# ---- Run -----------------------------------------------------------------
# NOTE: compute_vegetation_indices.sh must have completed successfully before
#       running this job. The NDVI/, NDRE/ and SAVI/ folders must exist inside
#       DATA_ROOT.  Add --skip-vi if they are not yet available.
singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE python -u \
        $SLURM_SUBMIT_DIR/utils/patch_weedyrice.py \
            --data-root   $DATA_ROOT \
            --output-root $OUTPUT_ROOT \
            --patch-size  128 \
            --workers     $SLURM_CPUS_PER_TASK"

date