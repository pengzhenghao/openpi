#!/bin/bash
#SBATCH --job-name="pi05_libero_fsdp_%j_%t"
#SBATCH --output="pi05_libero_fsdp_%j_%t.out"
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --time=7-00:00:00
#SBATCH --partition="gpu08"

# Note: no need to activate conda env

# Navigate to the project root
cd ~/vla/openpi

echo "Current directory: $(pwd)"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Run the training command with FSDP enabled (8 devices)
uv run scripts/train.py pi05_libero --exp-name="1202_debug_fsdp_$(date +%Y%m%d_%H%M%S)" --overwrite --fsdp-devices=8 --log-interval=5

