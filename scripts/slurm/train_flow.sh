#!/bin/bash
#SBATCH --job-name=flow_train
#SBATCH --output=logs/flow_train_%j.out
#SBATCH --error=logs/flow_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

mkdir -p logs outputs/stage2_flow
hostname
date

DATA_ROOT="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-RGBMS-DB/}"

# Stage 2 is single-GPU, so OMP threads can use full CPU budget
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 WANDB_API_KEY=$WANDB_API_KEY WANDB_MODE=offline PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE python -u \
            tbd/mae/train_flow.py \
                --config-path $SLURM_SUBMIT_DIR/configs \
                data.rgb_dir=$DATA_ROOT/RGB \
                data.ms_dir=$DATA_ROOT/Multispectral \
                data.batch_size=256 \
                data.num_workers=8 \
                data.cache_images=false \
                flow.mae_checkpoint=$SLURM_SUBMIT_DIR/outputs/stage1_mae/mae_best.pth \
                flow.epochs=100 \
                flow.use_wandb=true"

date