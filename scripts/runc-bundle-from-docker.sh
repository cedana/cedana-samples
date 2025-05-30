#!/bin/bash

# Creates a runc bundle from a Docker image.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <docker-image>" >&2
    exit 1
fi

IMG="$1"
BUNDLE="bundle"

mkdir -p "$BUNDLE/rootfs"

# Create and export the docker container's filesystem
cid=$(docker create "$IMG")
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
docker export "$cid" | tar -C "$BUNDLE/rootfs" -xf -

# Generate default config.json via runc
cd "$BUNDLE"
runc spec

# Modify config.json for NVIDIA GPU support
if command -v nvidia-container-toolkit >/dev/null 2>&1; then
    if command -v nvidia-container-cli >/dev/null 2>&1; then
        nvidia-container-cli configure --runtime=oci --config=./config.json .
    else
        echo "Warning: nvidia-container-cli not found, manual config of GPUs required." >&2
    fi
else
    echo "Warning: nvidia-container-toolkit not found, GPU support may not be configured." >&2
fi

echo "Bundle created in $BUNDLE"
