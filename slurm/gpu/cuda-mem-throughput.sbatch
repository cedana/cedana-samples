#!/bin/bash
#SBATCH --job-name=cuda-mem-throughput
#SBATCH --output=cuda-mem-throughput-%j.out
#SBATCH --error=cuda-mem-throughput-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

# Workload: CUDA Memory Throughput (SAXPY)
# Description: Measures GPU memory bandwidth using SAXPY operation
# Type: GPU

echo "Starting CUDA memory throughput benchmark on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

cd gpu_smr || exit 1

if [ ! -f mem-throughput-saxpy ]; then
    echo "ERROR: mem-throughput-saxpy binary not found. Please compile first: cd gpu_smr && bash build.sh"
    exit 1
fi

echo ""
echo "=== Running memory throughput benchmarks ==="
echo ""

echo "Test 1: Default configuration (64M elements)"
./mem-throughput-saxpy
echo ""

echo "Job completed at: $(date)"
