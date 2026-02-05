#!/bin/bash
#SBATCH --job-name=counting
#SBATCH --output=counting-%j.out
#SBATCH --error=counting-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M

# Workload: Timestamp Logger
# Description: Continuously logs timestamps for checkpoint/restore testing
# Type: CPU

echo "Starting counting workload on $(hostname)"
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
echo "Working directory: $(pwd)"

# Run the counting script
bash cpu_smr/counting.sh
