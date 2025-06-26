ARG CUDA_VERSION=12.4

# Install dependencies
FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="Niranjan Ravichandra <nravic@cedana.ai>"

# Install all system dependencies in a single layer with minimal packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    wget \
    git \
    python3 \
    python3-venv \
    python3-pip \
    cmake \
    openmpi-bin \
    openmpi-doc \
    libopenmpi-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Create app directory
WORKDIR /app

# Copy only necessary source files
COPY cpu_smr/ /app/cpu_smr/
COPY gpu_smr/ /app/gpu_smr/

# Build GPU workloads and CPU workloads in a single layer
RUN <<EOT
set -eux
# Build GPU workloads
cd /app/gpu_smr
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
find /app/gpu_smr/build -type f -executable -exec mv {} /app/gpu_smr \;
rm -rf /app/gpu_smr/build

# Build CPU workloads
cd /app/cpu_smr/mpi
mpicc -O3 mpi_pi_loop.c -o mpi_pi_loop

# Clean up build artifacts
find /app -name "*.o" -delete
find /app -name "*.a" -delete
EOT

# Use smaller runtime image
FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu22.04 AS runtime 

ARG TORCH_VERSION=2.4

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    openmpi-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Copy the app from the builder stage
COPY --from=builder /app /app

WORKDIR /app

# Copy requirements file
COPY requirements-torch${TORCH_VERSION}.txt /app/requirements.txt

# Set up Python virtual environment and install dependencies efficiently
RUN python3 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/* /var/tmp/*

# Set the virtual environment as the default Python
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for better security
RUN useradd -m -u 1000 cedana && chown -R cedana:cedana /app
USER cedana

ENTRYPOINT ["/bin/bash"] 