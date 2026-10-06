#!/bin/bash
# =============================================================================
# Linux HPC Setup Script
# Installs Native OpenBabel, compiles AutoDock-GPU, and sets up Conda.
# =============================================================================

echo "1. Installing System Dependencies (Requires sudo/admin on HPC or use module load)"
# If you don't have sudo, use: module load openbabel cuda autodock
sudo apt-get update
sudo apt-get install -y openbabel libopenbabel-dev build-essential make autodock

echo "2. Compiling AutoDock-GPU from source..."
cd bin
git clone https://github.com/ccsb-scripps/AutoDock-GPU
cd AutoDock-GPU
# Assuming CUDA is installed and nvcc is in PATH:
export GPU_INCLUDE_PATH=/usr/local/cuda/include
export GPU_LIBRARY_PATH=/usr/local/cuda/lib64
make DEVICE=CUDA NUMWI=128
cp bin/autodock_gpu_128wi ../
cd ../../
echo "AutoDock-GPU compiled and placed in bin/"

echo "3. Setting up Conda Environment..."
conda create -n mol_linux python=3.10 -y
source activate mol_linux
conda install -c conda-forge rdkit -y
pip install -r requirements_linux_hpc.txt

echo "4. Generating Receptor Grid Maps for AutoDock-GPU..."
python scripts/prepare_adgpu_maps.py

echo "Setup Complete! You are ready to submit jobs via SLURM."
