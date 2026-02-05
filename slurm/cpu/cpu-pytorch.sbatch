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

WORKSPACE_ROOT="$SLURM_SUBMIT_DIR"
while [[ "$WORKSPACE_ROOT" != "/" ]]; do
    if [[ -d "$WORKSPACE_ROOT/cpu_smr" ]]; then
        break
    fi
    WORKSPACE_ROOT="$(dirname "$WORKSPACE_ROOT")"
done

if [[ ! -d "$WORKSPACE_ROOT/cpu_smr" ]]; then
    echo "ERROR: Could not find workspace root (looking for cpu_smr directory)"
    exit 1
fi

cd "$WORKSPACE_ROOT" || { echo "ERROR: Failed to cd to $WORKSPACE_ROOT"; exit 1; }

if ! python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Installing..."
    pip install torch --quiet
    echo "PyTorch installed successfully"
fi

python3 cpu_smr/cpu_pytorch.py
