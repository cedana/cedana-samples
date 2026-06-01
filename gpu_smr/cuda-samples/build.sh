#!/usr/bin/env bash
# Build NVIDIA cuda-samples from the tag matching the host's CUDA version.
# Intended to be invoked from the cedana-samples Dockerfile builder stage.
#
# Usage: build.sh <CUDA_VERSION> [<out_dir>]
#   CUDA_VERSION:  full X.Y or X.Y.Z (e.g. 12.4 or 12.8.0). Only X.Y is used.
#   out_dir:       defaults to /app/gpu_smr/cuda-samples
#
# Output layout:
#   <out_dir>/bin/               prebuilt sample executables (flattened)
#   <out_dir>/run_tests.py       vendored test harness (cedana fork — emits results.json)
#   <out_dir>/compare-results.py differential gate (native vs intercepted)
#   <out_dir>/test_args.json     upstream test config (pulled from master)
#
# Notes:
# - v12.8+ uses CMake at the repo root; older tags use per-sample Makefiles.
# - run_tests.py is a cedana fork checked in next to this script (it emits a
#   machine-readable results.json the differential smoke test diffs). It and
#   compare-results.py are copied into <out_dir> from here.
# - test_args.json only exists on master and v12.9+ tags, so we always fetch it
#   from master regardless of the built tag. Samples not present in the built
#   bin/ are silently absent from the test run.
set -euxo pipefail

CUDA_VERSION="${1:?CUDA_VERSION required (e.g. 12.4)}"
OUT_DIR="${2:-/app/gpu_smr/cuda-samples}"
SAMPLES_TAG="v$(echo "${CUDA_VERSION}" | cut -d. -f1,2)"

# Directory of this script — source of the vendored harness/gate scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
mkdir -p "${OUT_DIR}/bin"

if [ -f CMakeLists.txt ]; then
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel "$(nproc)"
    find build -type f -executable ! -name '*.so*' -exec cp {} "${OUT_DIR}/bin/" \;
else
    make -C Samples -j"$(nproc)" -k || true
    if [ -d bin ]; then
        find bin -type f -executable -exec cp {} "${OUT_DIR}/bin/" \;
    fi
    find Samples -type f -executable \
        ! -name '*.sh' ! -name 'Makefile' \
        -path '*/release/*' \
        -exec cp {} "${OUT_DIR}/bin/" \; || true
fi

# Vendored harness + gate live next to this script. Copy them into OUT_DIR
# unless we're already building in place (default OUT_DIR == SCRIPT_DIR).
for f in run_tests.py compare-results.py; do
    if ! [ "${SCRIPT_DIR}/${f}" -ef "${OUT_DIR}/${f}" ]; then
        cp "${SCRIPT_DIR}/${f}" "${OUT_DIR}/${f}"
    fi
done

curl -fsSL https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/test_args.json \
    -o "${OUT_DIR}/test_args.json"
chmod +x "${OUT_DIR}/run_tests.py" "${OUT_DIR}/compare-results.py"
