srun singularity exec --nv \
     -B ~/third/p10/p10_venv:/scratch/p10_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/container/pytorch/pytorch_26.01.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
     source /scratch/p10_venv/bin/activate && \
     pip install --no-cache-dir opencv-python-headless einops matplotlib numpy torchvision scipy h5py hdf5storage tqdm spectral pandas openpyxl segmentation-models-pytorch"
