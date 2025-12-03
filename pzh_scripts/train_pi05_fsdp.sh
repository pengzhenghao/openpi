#!/bin/bash
#SBATCH --job-name="pi05_libero_fsdp"
#SBATCH --output="pi05_libero_fsdp.out"
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --partition="gpu09"

# Note: no need to activate conda env

# Navigate to the project root
cd ~/vla/openpi

echo "Current directory: $(pwd)"

# Run the training command with FSDP enabled (4 devices)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=1202_debug_fsdp --overwrite --fsdp-devices=4

