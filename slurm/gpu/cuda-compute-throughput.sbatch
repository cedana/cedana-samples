#!/bin/bash
#SBATCH --job-name=cuda-compute-throughput
#SBATCH --output=cuda-compute-throughput-%j.out
#SBATCH --error=cuda-compute-throughput-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

# Workload: CUDA Compute Throughput (SAXPY)
# Description: Measures GPU compute throughput in GFLOP/s using SAXPY operation
# Type: GPU

echo "Starting CUDA compute throughput benchmark on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

cd gpu_smr || exit 1

if [ ! -f compute-throughput-saxpy ]; then
    echo "ERROR: compute-throughput-saxpy binary not found. Please compile first: cd gpu_smr && bash build.sh"
    exit 1
fi

echo ""
echo "=== Running compute throughput benchmarks ==="
echo ""

echo "Test 1: Default configuration (64M elements)"
./compute-throughput-saxpy
echo ""

echo "Job completed at: $(date)"
