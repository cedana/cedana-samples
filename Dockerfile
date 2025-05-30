ARG CUDA_VERSION=12.4

# Install dependencies
FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="Niranjan Ravichandra <nravic@cedana.ai>"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    wget \
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

# Copy workload src
COPY cpu_smr/ /app/cpu_smr/
COPY gpu_smr/ /app/gpu_smr/

# Build GPU workloads
WORKDIR /app/gpu_smr
RUN <<EOT
set -eux
cmake $@ -B build -S .
cmake --build build
find /app/gpu_smr/build -type f -executable -exec mv {} /app/gpu_smr \;
rm -rf /app/gpu_smr/build
EOT

# build CPU workloads
WORKDIR /app/cpu_smr/mpi
RUN <<EOT
set -eux
mpicc mpi_pi_loop.c -o mpi_pi_loop
EOT

# Download and setup llama.cpp
# FIXME: Download llama cpp for for specific arch
# RUN <<EOT
# set -eux
# wget https://github.com/ggml-org/llama.cpp/releases/download/b5497/llama-b5497-bin-ubuntu-x64.zip -o llama.zip
# unzip llama.zip && cp -r build/bin/* /usr/local/bin/
# EOT

# Use smaller image for actually running the app
FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu22.04 AS runtime 

ARG TORCH_VERSION=2.4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip

# Install MPI dependencies
RUN apt-get update && apt-get install -y openmpi-bin 

# Copy the app from the builder stage
COPY --from=builder /app /app

WORKDIR /app

# Copy requirements file
COPY requirements-torch${TORCH_VERSION}.txt /app/requirements.txt

# Set up Python virtual environment and install dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/bash"]
