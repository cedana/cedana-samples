#!/bin/bash

# Creates a runc bundle from a Docker image.

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <docker-image>"
    exit 1
fi

IMG="$1"
BUNDLE="bundle"

mkdir -p "$BUNDLE/rootfs"

# Create and export the docker container's filesystem
cid=$(docker create "$IMG")
docker export "$cid" | tar -C "$BUNDLE/rootfs" -xf -
docker rm "$cid"

# Generate default config.json via runc
cd "$BUNDLE"
runc spec

echo "Bundle created in $BUNDLE"
