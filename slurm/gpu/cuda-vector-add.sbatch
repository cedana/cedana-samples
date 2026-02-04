#!/bin/bash
#SBATCH --job-name=cuda-vector-add
#SBATCH --output=cuda-vector-add-%j.out
#SBATCH --error=cuda-vector-add-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

# Workload: CUDA Vector Addition
# Description: Simple CUDA vector addition running in infinite loop
# Type: GPU

echo "Starting CUDA vector addition on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

cd gpu_smr || exit 1

if [ ! -f vector_add ]; then
    echo "ERROR: vector_add binary not found. Please compile first: cd gpu_smr && bash build.sh"
    exit 1
fi

echo "Running vector addition (infinite loop, Ctrl+C to stop)..."
./vector_add

echo "Job completed at: $(date)"
