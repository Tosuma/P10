#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --output=logs/mae_pretrain_%j.out
#SBATCH --error=logs/mae_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

mkdir -p logs outputs/stage1_mae
hostname
date

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

GPUS=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26_02.sif \
    /bin/bash -lc "source my_venv/bin/activate && \
        PYTHONPATH=$SLURM_SUBMIT_DIR python -u -m torch.distributed.run \
            --standalone \
            --nproc_per_node=${GPUS} \
            tbd/mae/train_mae.py \
                --config-path $SLURM_SUBMIT_DIR/configs \
                data.rgb_dir=$DATA_ROOT/RGB \
                data.ms_dir=$DATA_ROOT/Multispectral \
                data.batch_size=64 \
                data.num_workers=8 \
                mae.epochs=200 \
                mae.use_checkpoint=true \
                mae.use_wandb=true"

date