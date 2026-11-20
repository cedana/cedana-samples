#!/bin/bash
#SBATCH --job-name=cuda-graph-events
#SBATCH --output=cuda-graph-events-%j.out
#SBATCH --error=cuda-graph-events-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --export=CEDANA_ENABLE=1

# Workload: CUDA Graph Events
# Description: CUDA graph loop with graph-node events, periodic graph teardown/rebuild and event-pool churn
# Type: GPU

echo "Starting CUDA Graph Events on $(hostname)"
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

if [ ! -f cuda_graph_events ]; then
    echo "ERROR: cuda_graph_events binary not found. Please compile first: cd gpu_smr && bash build.sh"
    exit 1
fi

echo "Running CUDA graph loop with event churn (infinite loop, Ctrl+C to stop)..."
./cuda_graph_events || exit $?

echo "Job completed at: $(date)"
