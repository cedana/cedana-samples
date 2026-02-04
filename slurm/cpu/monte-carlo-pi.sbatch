#!/bin/bash
#SBATCH --job-name=monte-carlo-pi
#SBATCH --output=monte-carlo-pi-%j.out
#SBATCH --error=monte-carlo-pi-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

# Workload: Monte Carlo Pi Estimation
# Description: Stateful, long-running computation estimating Pi using Monte Carlo method
# Type: CPU

echo "Starting Monte Carlo Pi estimation on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

python3 << 'EOF'
"""
Monte Carlo Pi Estimation - A stateful, long-running computation.

This workload maintains accumulated state (points inside/outside circle)
across many iterations, making it ideal for checkpoint/restore testing.
"""
import random
import time
import sys
import signal

# Configuration
BATCH_SIZE = 1_000_000        # Points per batch
TOTAL_BATCHES = 1000          # Total batches to run
REPORT_INTERVAL = 10          # Report every N batches

# State variables
total_inside = 0
total_points = 0
start_time = time.time()

def signal_handler(sig, frame):
    print(f"\nInterrupted! Final state:")
    print(f"  Total points: {total_points:,}")
    print(f"  Points inside: {total_inside:,}")
    if total_points > 0:
        pi_estimate = 4.0 * total_inside / total_points
        print(f"  Pi estimate: {pi_estimate:.10f}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

print("=" * 60)
print("Monte Carlo Pi Estimation")
print("=" * 60)
print(f"Configuration:")
print(f"  Batch size: {BATCH_SIZE:,} points")
print(f"  Total batches: {TOTAL_BATCHES:,}")
print(f"  Total points: {BATCH_SIZE * TOTAL_BATCHES:,}")
print(f"  True Pi: 3.14159265358979323846...")
print("=" * 60)
print()

for batch in range(1, TOTAL_BATCHES + 1):
    # Monte Carlo sampling: count points inside unit circle
    inside = 0
    for _ in range(BATCH_SIZE):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside += 1

    # Update accumulated state
    total_inside += inside
    total_points += BATCH_SIZE

    # Estimate pi: ratio of circle area to square area = pi/4
    pi_estimate = 4.0 * total_inside / total_points
    error = abs(pi_estimate - 3.14159265358979323846)

    # Report progress
    if batch % REPORT_INTERVAL == 0 or batch == 1:
        elapsed = time.time() - start_time
        rate = total_points / elapsed / 1_000_000
        print(f"Batch {batch:4d}/{TOTAL_BATCHES} | "
              f"Points: {total_points:>12,} | "
              f"Pi: {pi_estimate:.10f} | "
              f"Error: {error:.2e} | "
              f"Rate: {rate:.2f}M pts/s")
        sys.stdout.flush()

    # Small delay to make it easier to checkpoint mid-computation
    time.sleep(0.1)

# Final results
elapsed = time.time() - start_time
pi_estimate = 4.0 * total_inside / total_points
error = abs(pi_estimate - 3.14159265358979323846)

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Total points sampled: {total_points:,}")
print(f"Points inside circle: {total_inside:,}")
print(f"Pi estimate:          {pi_estimate:.15f}")
print(f"True Pi:              3.141592653589793")
print(f"Absolute error:       {error:.2e}")
print(f"Total time:           {elapsed:.2f}s")
print(f"Throughput:           {total_points/elapsed/1e6:.2f} million points/s")
print("=" * 60)
EOF

echo "Job completed at: $(date)"
