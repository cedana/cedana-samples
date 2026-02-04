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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

bash cpu_smr/counting.sh
