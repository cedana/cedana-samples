#!/usr/bin/env bash
# Run each sample in samples.txt natively and under `cedana run process -g`;
# fail only on samples that pass natively but regress under interception.
# Starts its own daemon unless CEDANA_ADDRESS already points at a live one.
set -uo pipefail

SAMPLES_DIR="${SAMPLES_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BIN_DIR="${BIN_DIR:-${SAMPLES_DIR}/bin}"
SAMPLES_LIST="${SAMPLES_LIST:-${SAMPLES_DIR}/samples.txt}"
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-120}" # per-run wall-clock bound (seconds)

[ -d "${BIN_DIR}" ]      || { echo "no sample bin dir at ${BIN_DIR} (rebuild cedana-samples image)" >&2; exit 2; }
[ -f "${SAMPLES_LIST}" ] || { echo "no allowlist at ${SAMPLES_LIST}" >&2; exit 2; }

STARTED_DAEMON=0
DAEMON_TAG="cudasamples-$$"

start_daemon() {
    if [ -n "${CEDANA_ADDRESS:-}" ] && [ -S "${CEDANA_ADDRESS}" ]; then
        echo "using existing daemon at ${CEDANA_ADDRESS}"
        return 0
    fi
    export CEDANA_ADDRESS="/tmp/cedana-${DAEMON_TAG}.sock"
    export CEDANA_CONFIG_DIR="/tmp/cedana-${DAEMON_TAG}"
    export CEDANA_GPU_LOG_DIR="${CEDANA_CONFIG_DIR}"
    export CEDANA_GPU_SOCK_DIR="${CEDANA_CONFIG_DIR}"
    echo "starting daemon at ${CEDANA_ADDRESS}"
    cedana daemon start --init-config --db "/tmp/cedana-${DAEMON_TAG}.db" \
        >"/tmp/cedana-${DAEMON_TAG}.log" 2>&1 &
    STARTED_DAEMON=1
    local i=0
    while [ ! -S "${CEDANA_ADDRESS}" ]; do
        sleep 1
        i=$((i + 1))
        if [ "${i}" -gt 60 ]; then
            echo "daemon failed to start after 60s; log:" >&2
            cat "/tmp/cedana-${DAEMON_TAG}.log" >&2 || true
            exit 1
        fi
    done
}

stop_daemon() {
    [ "${STARTED_DAEMON}" = "1" ] || return 0
    cedana daemon stop >/dev/null 2>&1 \
        || pkill -TERM -f "cedana daemon start .*${DAEMON_TAG}" 2>/dev/null \
        || true
}
trap stop_daemon EXIT

start_daemon

regressions=()
compared=0
skipped=0

while IFS= read -r line || [ -n "${line}" ]; do
    sample="${line%%#*}"                 # strip trailing comment
    sample="${sample//[[:space:]]/}"     # strip whitespace
    [ -z "${sample}" ] && continue

    bin="${BIN_DIR}/${sample}"
    if [ ! -x "${bin}" ]; then
        echo "skip ${sample}: not present in image"
        skipped=$((skipped + 1))
        continue
    fi

    # Skip samples that fail/waive natively — environmental, not cedana's fault.
    if ! timeout "${SAMPLE_TIMEOUT}" "${bin}" >/dev/null 2>&1; then
        echo "skip ${sample}: native rc=$? (environmental, not cedana)"
        skipped=$((skipped + 1))
        continue
    fi

    if timeout "${SAMPLE_TIMEOUT}" cedana run process --attach -g \
        --jid "${DAEMON_TAG}-${sample}" -- "${bin}" >/dev/null 2>&1; then
        echo "ok   ${sample}: native=0 intercepted=0"
    else
        rc=$?
        echo "FAIL ${sample}: native=0 intercepted=${rc} (cedana regression)"
        regressions+=("${sample}(rc=${rc})")
    fi
    compared=$((compared + 1))
done <"${SAMPLES_LIST}"

echo "----"
echo "cuda-samples interception: compared=${compared} skipped=${skipped} regressions=${#regressions[@]}"

if [ "${compared}" -eq 0 ]; then
    echo "FAIL: no samples ran under interception (check ${BIN_DIR} and native failures above)" >&2
    exit 1
fi
if [ "${#regressions[@]}" -ne 0 ]; then
    echo "FAIL: cedana-induced regression(s): ${regressions[*]}" >&2
    exit 1
fi
echo "PASS: no cedana-induced regressions across ${compared} sample(s)"
