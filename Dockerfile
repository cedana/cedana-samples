# This image is used in CI to run cedana-sample tests.
# This container itself is checkpoint/restored, which gives us some flexibility
# when deciding what to run.
#
# TODO:
#   - PVC for weights?

# Install dependencies
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="Niranjan Ravichandra <nravic@cedana.ai>"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-venv \
    python3-pip \
    jq \
    libnet1-dev \
    libprotobuf-dev \
    libprotobuf-c-dev \
    protobuf-c-compiler \
    protobuf-compiler \
    python3-protobuf \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Set up Python virtual environment and install dependencies
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for workloads
RUN mkdir -p /app/workloads

# Copy workload directories
COPY cpu_smr/ /app/cpu_smr/
COPY gpu_smr/ /app/gpu_smr/
COPY requirements.txt /app/

# Define entrypoint script
RUN echo '#!/bin/bash\n\
python3 /app/workloads/$1\n' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
