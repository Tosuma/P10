#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Inference + evaluation (generate heatmaps, UMAP, score histograms)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=evaluate
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

hostname
date

mkdir -p logs outputs/heatmaps outputs/figures

DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-RGBMS-DB/}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE python -u \
            tbd/mae/evaluate.py \
                --config-path $SLURM_SUBMIT_DIR/configs \
                data.rgb_dir=$DATA_ROOT/RGB \
                data.ms_dir=$DATA_ROOT/Multispectral \
                data.batch_size=64 \
                flow.mae_checkpoint=$SLURM_SUBMIT_DIR/outputs/stage1_mae/mae_best.pth \
                flow.output_dir=$SLURM_SUBMIT_DIR/outputs/stage2_flow \
                flow.heatmap_output_dir=$SLURM_SUBMIT_DIR/outputs/heatmaps \
                flow.export_geotiff=true"

date