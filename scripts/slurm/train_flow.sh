#!/bin/bash
#SBATCH --job-name=flow_train
#SBATCH --output=logs/flow_train_%j.out
#SBATCH --error=logs/flow_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

mkdir -p logs outputs/stage2_flow
hostname
date

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "source my_venv/bin/activate && \
        PYTHONPATH=$SLURM_SUBMIT_DIR python -u \
            tbd/mae/train_flow.py \
                --config-path $SLURM_SUBMIT_DIR/configs \
                data.rgb_dir=$DATA_ROOT/RGB \
                data.ms_dir=$DATA_ROOT/Multispectral \
                data.batch_size=128 \
                data.num_workers=8 \
                flow.mae_checkpoint=outputs/stage1_mae/mae_best.pth \
                flow.epochs=100 \
                flow.use_wandb=true"

date