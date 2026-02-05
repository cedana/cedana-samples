#!/bin/bash
#SBATCH --job-name=cuda-gpu-props
#SBATCH --output=cuda-gpu-props-%j.out
#SBATCH --error=cuda-gpu-props-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1

# Workload: CUDA GPU Properties
# Description: Displays detailed information about available CUDA devices
# Type: GPU

echo "Starting CUDA GPU properties detection on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

WORKSPACE_ROOT="$SLURM_SUBMIT_DIR"
while [[ "$WORKSPACE_ROOT" != "/" ]]; do
    if [[ -d "$WORKSPACE_ROOT/gpu_smr" ]]; then
        break
    fi
    WORKSPACE_ROOT="$(dirname "$WORKSPACE_ROOT")"
done

if [[ ! -d "$WORKSPACE_ROOT/gpu_smr" ]]; then
    echo "ERROR: Could not find workspace root (looking for gpu_smr directory)"
    exit 1
fi

cd "$WORKSPACE_ROOT" || { echo "ERROR: Failed to cd to $WORKSPACE_ROOT"; exit 1; }

cd gpu_smr || exit 1

if [ ! -f gpu_props ]; then
    echo "ERROR: gpu_props binary not found. Please compile first: cd gpu_smr && bash build.sh"
    exit 1
fi

echo "=== CUDA Device Properties ==="
./gpu_props
echo ""

echo ""
echo "Job completed at: $(date)"
