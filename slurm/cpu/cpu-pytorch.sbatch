#!/bin/bash
#SBATCH --job-name=cpu-pytorch
#SBATCH --output=cpu-pytorch-%j.out
#SBATCH --error=cpu-pytorch-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

# Workload: CPU PyTorch Linear Model Training
# Description: Simple linear regression with synthetic data using PyTorch (CPU)
# Type: CPU

echo "Starting CPU PyTorch workload on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

if ! python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Installing..."
    pip install torch --quiet
    echo "PyTorch installed successfully"
fi

python3 cpu_smr/cpu_pytorch.py
