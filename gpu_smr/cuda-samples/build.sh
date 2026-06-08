#!/usr/bin/env bash
# Build NVIDIA cuda-samples from the tag matching the host's CUDA version.
# Intended to be invoked from the cedana-samples Dockerfile builder stage.
#
# Usage: build.sh <CUDA_VERSION> [<out_dir>]
#   CUDA_VERSION:  full X.Y or X.Y.Z (e.g. 12.4 or 12.8.0). Only X.Y is used.
#   out_dir:       defaults to /app/gpu_smr/cuda-samples
#
# Output: <out_dir>/bin/. cuda-samples.sh + samples.txt run a curated subset
# under cedana. v12.8+ builds via root CMake; older tags are skipped (exit 0).
set -euxo pipefail

CUDA_VERSION="${1:?CUDA_VERSION required (e.g. 12.4)}"
OUT_DIR="${2:-/app/gpu_smr/cuda-samples}"
SAMPLES_TAG="v$(echo "${CUDA_VERSION}" | cut -d. -f1,2)"

# Resolve OUT_DIR to an absolute path before we cd into the work dir below.
# Otherwise a relative OUT_DIR (e.g. ./out) is created *inside* WORK_DIR and
# wiped by the cleanup trap, leaving an empty build that exits 0.
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

git clone --depth 1 --branch "${SAMPLES_TAG}" \
    https://github.com/NVIDIA/cuda-samples.git "${WORK_DIR}/cuda-samples"

cd "${WORK_DIR}/cuda-samples"

# Pre-CMake tags (<= v12.6) aren't supported; skip rather than ship junk.
if [ ! -f CMakeLists.txt ]; then
    echo "Skipping cuda-samples build for ${SAMPLES_TAG}: no root CMakeLists.txt (pre-CMake tag)."
    exit 0
fi

mkdir -p "${OUT_DIR}/bin"
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "$(nproc)"
find build -type f -executable ! -name '*.so*' -exec cp {} "${OUT_DIR}/bin/" \;

# Fail loudly if nothing built, rather than ship an empty (falsely green) test.
sample_count=$(find "${OUT_DIR}/bin" -maxdepth 1 -type f -executable \
    ! -name '*.so*' ! -name '*.dll' | wc -l)
if [ "${sample_count}" -eq 0 ]; then
    echo "ERROR: no cuda-samples binaries built into ${OUT_DIR}/bin (CUDA ${CUDA_VERSION}, ${SAMPLES_TAG})" >&2
    exit 1
fi
echo "Built ${sample_count} cuda-samples binaries into ${OUT_DIR}/bin"
