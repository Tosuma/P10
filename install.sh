srun singularity exec --nv \
     -B ~<PATH TIL DIN VENV>p10_venv:/scratch/p10_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/container/pytorch/pytorch_26.02.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
     source /scratch/p10_venv/bin/activate && \
     pip install --no-cache-dir torch torchmetrics einops opencv-python-headless matplotlib torchvision scipy h5py hdf5storage tqdm spectral pandas openpyxl torchvision segmentation-models-pytorch numpy Pillow PyYAML"