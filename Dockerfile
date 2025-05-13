ARG CUDA_VERSION=12.4

# Install dependencies
FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu22.04

ARG TORCH_VERSION=2.4

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
    cmake

# Install MPI dependencies
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    openmpi-doc \
    libopenmpi-dev

# Create app directory
WORKDIR /app

# Copy requirements file
COPY requirements-torch${TORCH_VERSION}.txt /app/requirements.txt

# Set up Python virtual environment and install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for workloads
RUN mkdir -p /app/workloads

# Copy workload src
COPY cpu_smr/ /app/cpu_smr/
COPY gpu_smr/ /app/gpu_smr/

# Build workloads
WORKDIR /app/gpu_smr
RUN <<EOT
set -eux
cmake $@ -B build -S .
cmake --build build
find /app/gpu_smr/build -type f -executable -exec mv {} /app/gpu_smr \;
rm -rf /app/gpu_smr/build
EOT

# Define entrypoint script
WORKDIR /app

ENTRYPOINT ["/bin/bash"]
