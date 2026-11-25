#!/bin/bash
#SBATCH --job-name=go-counting
#SBATCH --output=go-counting-%j.out
#SBATCH --error=go-counting-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M
#SBATCH --cedana
#SBATCH --signal=B:USR1@0

# Workload: Go Timestamp Logger
# Description: Logs a counter every second from a program built with Go 1.25+
# Type: CPU

echo "Starting go-counting workload on $(hostname)"
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

# Run the Go counter
exec ./cpu_smr/go-counting/go-counting
