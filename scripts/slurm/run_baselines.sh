#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Run all baselines (CPU-heavy; PatchCore benefits from GPU)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                  # For PatchCore ResNet50 feature extraction
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

hostname
date

mkdir -p logs outputs/baselines

DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-RGBMS-DB/}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE python -u \
            tbd/mae/run_baselines.py \
                --config-path $SLURM_SUBMIT_DIR/configs \
                data.rgb_dir=$DATA_ROOT/RGB \
                data.ms_dir=$DATA_ROOT/Multispectral \
                data.batch_size=64 \
                data.num_workers=8 \
                baselines.output_dir=$SLURM_SUBMIT_DIR/outputs/baselines"

date