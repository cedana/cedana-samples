#!/bin/bash
#SBATCH --job-name=cuda-graph-siblings
#SBATCH --output=cuda-graph-siblings-%j.out
#SBATCH --error=cuda-graph-siblings-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --export=CEDANA_ENABLE=1

# Workload: CUDA Graph Siblings
# Description: Multiple live CUDA graphs of one topology, the cold ones deferred until $GATE appears
# Type: GPU

echo "Starting CUDA Graph Siblings on $(hostname)"
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

if [ ! -f cuda_graph_siblings ]; then
    echo "ERROR: cuda_graph_siblings binary not found. Please compile first: cd gpu_smr && bash build.sh"
    exit 1
fi

GATE=/tmp/cedana-graph-siblings-gate

echo "Running sibling CUDA graphs, deferring the cold ones until $GATE appears..."
./cuda_graph_siblings "$GATE" || exit $?

echo "Job completed at: $(date)"
