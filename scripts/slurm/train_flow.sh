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

hostname
date

CEPH_DATA="${DATA_ROOT:-/ceph/home/student.aau.dk/ba35so/P10/data/WeedyRice-RGBMS-DB/}"

# ── Stage data from Ceph to local /tmp ────────────────────────────────────────
LOCAL_DATA="/tmp/mae_flow_data_${SLURM_JOB_ID}"
echo "Staging data from Ceph to ${LOCAL_DATA} ..."
mkdir -p "${LOCAL_DATA}/RGB" "${LOCAL_DATA}/Multispectral"
rsync -a --no-progress "${CEPH_DATA}/RGB/" "${LOCAL_DATA}/RGB/"
rsync -a --no-progress "${CEPH_DATA}/Multispectral/" "${LOCAL_DATA}/Multispectral/"
echo "Data staging complete: $(du -sh ${LOCAL_DATA} | cut -f1)"

# Stage 2 is single-GPU, so OMP threads can use full CPU budget
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

VENV_SITE="$SLURM_SUBMIT_DIR/my_venv/lib/python3.12/site-packages"

singularity exec --nv \
    /ceph/container/pytorch/pytorch_26.02.sif \
    /bin/bash -lc "HYDRA_FULL_ERROR=1 WANDB_API_KEY=$WANDB_API_KEY PYTHONPATH=$SLURM_SUBMIT_DIR:$VENV_SITE python -u \
            tbd/mae/train_flow.py \
                --config-path $SLURM_SUBMIT_DIR/configs \
                data.rgb_dir=${LOCAL_DATA}/RGB \
                data.ms_dir=${LOCAL_DATA}/Multispectral \
                data.batch_size=256 \
                data.num_workers=8 \
                data.cache_images=true \
                flow.mae_checkpoint=$SLURM_SUBMIT_DIR/outputs/stage1_mae/mae_best.pth \
                flow.epochs=100 \
                flow.use_wandb=true"

# Clean up local staging area
rm -rf "${LOCAL_DATA}"

date