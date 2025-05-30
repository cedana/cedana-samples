#!/bin/bash

# Creates a runc bundle from a Docker image, ensuring /tmp exists and copying envs/args.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <docker-image>" >&2
    exit 1
fi

IMG="$1"
BUNDLE="bundle"

mkdir -p "$BUNDLE"

sudo mkdir "$BUNDLE/rootfs"

# Create and export the docker container's filesystem
cid=$(docker create "$IMG")
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
docker export "$cid" | sudo tar -C "$BUNDLE/rootfs" -xf -

# Ensure /tmp exists and is writable
sudo mkdir -p "$BUNDLE/rootfs/tmp"
sudo chmod 1777 "$BUNDLE/rootfs/tmp"

# Get entrypoint, cmd, env from image
inspect=$(docker inspect "$cid")
entrypoint=$(echo "$inspect" | jq -r '.[0].Path | @sh' | sed "s/^'//;s/'$//")
args=$(echo "$inspect" | jq -r '.[0].Args | @sh' | sed "s/^'//;s/'$//")
envs=$(echo "$inspect" | jq -c '.[0].Config.Env')

# Generate default config.json via runc
cd "$BUNDLE"
runc spec

CONFIG="config.json"
if [ -f "$CONFIG" ]; then
    jq --argjson env "$envs" \
        --arg entrypoint "$entrypoint" \
        --arg args "$args" \
        '
        .process.args = ($entrypoint | split(" ")) + ($args | split(" "))
       | .process.env = $env
       | .process.env +=
         ["NVIDIA_VISIBLE_DEVICES=all", "NVIDIA_DRIVER_CAPABILITIES=all", "NVIDIA_REQUIRE_CUDA=cuda>=11.0"]
       | .hooks["prestart"] += [{
          "path": "/usr/bin/nvidia-container-runtime-hook",
          "args": ["nvidia-container-runtime-hook", "prestart"]
         }]
       | .mounts += [{
           "destination": "/tmp",
           "type": "tmpfs",
           "source": "tmpfs",
           "options": ["nosuid","strictatime","mode=1777","size=65536k"]
         }]
    ' "$CONFIG" > config_nvidia.json && mv config_nvidia.json "$CONFIG" || {
        echo "Failed to update $CONFIG for args/env/NVIDIA. Check if jq is installed and /dev/nvidia* exist." >&2
        exit 1
    }
    echo "Modified $CONFIG for NVIDIA GPU support, original container args/env, and /tmp tmpfs mount."
else
    echo "Warning: $CONFIG not found, cannot update for NVIDIA or env/args." >&2
fi

echo "Bundle created in $BUNDLE"
