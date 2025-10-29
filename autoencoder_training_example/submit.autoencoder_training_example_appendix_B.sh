#!/bin/bash -l

#SBATCH --job-name=cylinder_test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=h200
#SBATCH --output=out_err_files/cylinder_with_cool_loss_%j.out
#SBATCH --error=out_err_files/cylinder_with_cool_loss_%j.err

# Load the Conda environment
conda activate torch_cuda_pyro

# Run the Python script directly
python -u autoencoder_training_example_appendix_B.py
