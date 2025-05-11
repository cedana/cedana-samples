# Install dependencies
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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
RUN cmake $@ -B build -S .
RUN cmake --build build
RUN find /app/gpu_smr/build -type f -executable -exec mv {} /app/gpu_smr \;
RUN rm -rf /app/gpu_smr/build

# Define entrypoint script
RUN echo '#!/bin/bash\n\
python3 /app/workloads/$1\n' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
