ARG CUDA_VERSION=12.4

FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="Niranjan Ravichandra <nravic@cedana.ai>"

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

WORKDIR /app

COPY cpu_smr/ /app/cpu_smr/
COPY gpu_smr/ /app/gpu_smr/
COPY kubernetes/ /app/kubernetes/

RUN <<EOT
set -eux
cd /app/gpu_smr
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
find /app/gpu_smr/build -type f -executable -exec mv {} /app/gpu_smr \;
rm -rf /app/gpu_smr/build

cd /app/cpu_smr/mpi
mpicc -O3 mpi_pi_loop.c -o mpi_pi_loop

find /app -name "*.o" -delete
find /app -name "*.a" -delete
EOT

FROM nvidia/cuda:${CUDA_VERSION}.0-base-ubuntu22.04 AS runtime
ARG TORCH_VERSION=2.4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    openmpi-bin \
    git \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

COPY --from=builder /app /app
WORKDIR /app

COPY requirements-torch${TORCH_VERSION}.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/* /var/tmp/*


RUN <<EOT
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
EOT

ENTRYPOINT ["/bin/bash"]
