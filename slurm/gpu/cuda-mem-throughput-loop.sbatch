#!/bin/bash
#SBATCH --job-name=cuda-mem-throughput-loop
#SBATCH --output=cuda-mem-throughput-loop-%j.out
#SBATCH --error=cuda-mem-throughput-loop-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

# Workload: CUDA Memory Throughput Loop (SAXPY)
# Description: Measures GPU memory bandwidth in infinite loop using SAXPY operation
# Type: GPU

echo "Starting CUDA memory throughput loop benchmark on $(hostname)"
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

if [ ! -f mem-throughput-saxpy-loop ]; then
    echo "ERROR: mem-throughput-saxpy-loop binary not found. Please compile first: cd gpu_smr && bash build.sh"
    exit 1
fi

echo ""
echo "=== Running memory throughput in infinite loop (GB/s) ==="
echo ""
./mem-throughput-saxpy-loop

echo "Job completed at: $(date)"
