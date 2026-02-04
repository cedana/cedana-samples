#!/bin/bash
#SBATCH --job-name=numpy-matrix
#SBATCH --output=numpy-matrix-%j.out
#SBATCH --error=numpy-matrix-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Workload: NumPy/SciPy Matrix Operations
# Description: Performs computationally intensive matrix operations
# Type: CPU

echo "Starting NumPy matrix operations workload on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

echo "Installing dependencies..."
pip install numpy scipy --quiet

python3 << 'EOF'
"""
NumPy/SciPy Matrix Operations Benchmark

This workload performs a series of computationally intensive matrix
operations that are representative of scientific computing workloads.
Large matrices are kept in memory, making this ideal for testing
checkpoint/restore with significant memory state.
"""
import numpy as np
from scipy import linalg
import time
import sys

# Use all available threads
np.show_config()

print("=" * 70)
print("NumPy/SciPy Matrix Operations Benchmark")
print("=" * 70)
print(f"NumPy version: {np.__version__}")
print()

# Configuration - adjust matrix sizes based on available memory
SIZES = [2000, 3000, 4000, 5000]
ITERATIONS = 3

results = {}

def benchmark(name, func, *args, iterations=ITERATIONS):
    """Run a benchmark and report timing."""
    print(f"\n--- {name} ---")
    times = []
    result = None
    for i in range(iterations):
        start = time.time()
        result = func(*args)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.3f}s")
        sys.stdout.flush()
    avg = np.mean(times)
    std = np.std(times)
    print(f"  Average: {avg:.3f}s (+/- {std:.3f}s)")
    return result, avg

for n in SIZES:
    print()
    print("=" * 70)
    print(f"Matrix Size: {n} x {n}")
    print(f"Memory per matrix: ~{n*n*8/1e9:.2f} GB")
    print("=" * 70)

    # Generate random matrices
    print("\nGenerating random matrices...")
    np.random.seed(42)
    A = np.random.rand(n, n).astype(np.float64)
    B = np.random.rand(n, n).astype(np.float64)
    v = np.random.rand(n).astype(np.float64)
    print(f"  Matrix A checksum: {np.sum(A):.6e}")
    print(f"  Matrix B checksum: {np.sum(B):.6e}")

    # 1. Matrix Multiplication (DGEMM)
    C, t = benchmark(
        f"Matrix Multiplication (DGEMM) {n}x{n}",
        np.dot, A, B
    )
    flops = 2.0 * n**3
    print(f"  Performance: {flops/t/1e9:.2f} GFLOPS")
    print(f"  Result checksum: {np.sum(C):.6e}")
    results[f'matmul_{n}'] = t

    # 2. Matrix-Vector Multiplication
    y, t = benchmark(
        f"Matrix-Vector Multiplication {n}x{n}",
        np.dot, A, v
    )
    print(f"  Result checksum: {np.sum(y):.6e}")
    results[f'matvec_{n}'] = t

    # 3. LU Decomposition
    lu, t = benchmark(
        f"LU Decomposition {n}x{n}",
        linalg.lu, A
    )
    print(f"  P checksum: {np.sum(lu[0]):.6e}")
    print(f"  L checksum: {np.sum(lu[1]):.6e}")
    print(f"  U checksum: {np.sum(lu[2]):.6e}")
    results[f'lu_{n}'] = t

    # 4. QR Decomposition
    qr, t = benchmark(
        f"QR Decomposition {n}x{n}",
        np.linalg.qr, A
    )
    print(f"  Q checksum: {np.sum(qr[0]):.6e}")
    print(f"  R checksum: {np.sum(qr[1]):.6e}")
    results[f'qr_{n}'] = t

    # 5. Singular Value Decomposition (SVD) - expensive!
    if n <= 4000:  # SVD is very expensive for large matrices
        svd, t = benchmark(
            f"SVD {n}x{n}",
            np.linalg.svd, A, iterations=1
        )
        print(f"  Singular values (first 5): {svd[1][:5]}")
        results[f'svd_{n}'] = t

    # 6. Eigenvalue Decomposition (symmetric)
    S = A @ A.T  # Make symmetric positive definite
    eigvals, t = benchmark(
        f"Eigenvalue Decomposition (symmetric) {n}x{n}",
        np.linalg.eigvalsh, S, iterations=1
    )
    print(f"  Eigenvalues (first 5): {eigvals[:5]}")
    results[f'eig_{n}'] = t

    # 7. Cholesky Decomposition (symmetric positive definite)
    S = A @ A.T + n * np.eye(n)  # Ensure positive definite
    chol, t = benchmark(
        f"Cholesky Decomposition {n}x{n}",
        np.linalg.cholesky, S
    )
    print(f"  L checksum: {np.sum(chol):.6e}")
    results[f'chol_{n}'] = t

    # 8. Linear System Solve (Ax = b)
    b = np.random.rand(n).astype(np.float64)
    x, t = benchmark(
        f"Linear Solve (Ax=b) {n}x{n}",
        np.linalg.solve, A, b
    )
    residual = np.linalg.norm(A @ x - b)
    print(f"  Solution checksum: {np.sum(x):.6e}")
    print(f"  Residual norm: {residual:.6e}")
    results[f'solve_{n}'] = t

    # Clean up to free memory before next size
    del A, B, C, S, v, y
    print()

# Final Summary
print()
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
for key, value in sorted(results.items()):
    print(f"  {key:30s}: {value:8.3f}s")
print("=" * 70)
print("Benchmark complete!")
EOF

echo "Job completed at: $(date)"
