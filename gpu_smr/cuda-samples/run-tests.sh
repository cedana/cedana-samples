#!/usr/bin/env bash
# Smoke-test entrypoint: runs every prebuilt NVIDIA cuda-samples binary via the
# upstream run_tests.py harness. Intended for the GPU runner / interception
# smoke test. Exits non-zero on the first sample that fails so it can gate CI.
set -euo pipefail

SAMPLES_DIR="${SAMPLES_DIR:-/app/gpu_smr/cuda-samples}"
BIN_DIR="${BIN_DIR:-${SAMPLES_DIR}/bin}"
CONFIG="${CONFIG:-${SAMPLES_DIR}/test_args.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/cuda-samples-results}"
PARALLEL="${PARALLEL:-1}"

if [[ ! -f "${SAMPLES_DIR}/run_tests.py" ]]; then
    echo "run_tests.py missing at ${SAMPLES_DIR}/run_tests.py" >&2
    exit 2
fi
if [[ ! -d "${BIN_DIR}" ]]; then
    echo "No sample binaries at ${BIN_DIR}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo "Running cuda-samples smoke test"
echo "  bin dir:   ${BIN_DIR}"
echo "  config:    ${CONFIG}"
echo "  output:    ${OUTPUT_DIR}"
echo "  parallel:  ${PARALLEL}"

exec python3 "${SAMPLES_DIR}/run_tests.py" \
    --dir "${BIN_DIR}" \
    --config "${CONFIG}" \
    --output "${OUTPUT_DIR}" \
    --parallel "${PARALLEL}"
