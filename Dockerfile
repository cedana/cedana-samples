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
    libfreeimage-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
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

# Build NVIDIA cuda-samples from the tag matching CUDA_VERSION (e.g. CUDA_VERSION=12.8 -> tag v12.8).
# v12.8+ uses CMake at the repo root; older tags use per-sample Makefiles.
# Note: the nvidia/cuda base image sets CUDA_VERSION env to the full X.Y.Z (e.g. 12.8.0),
# which shadows the build ARG inside the shell, so we strip the patch component before
# constructing the cuda-samples tag.
RUN <<EOT
set -eux
SAMPLES_TAG="v$(echo "${CUDA_VERSION}" | cut -d. -f1,2)"
git clone --depth 1 --branch "${SAMPLES_TAG}" https://github.com/NVIDIA/cuda-samples.git /tmp/cuda-samples
cd /tmp/cuda-samples
if [ -f CMakeLists.txt ]; then
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel $(nproc)
    mkdir -p /app/cuda-samples/bin
    find build -type f -executable ! -name '*.so*' -exec cp {} /app/cuda-samples/bin/ \;
else
    make -C Samples -j$(nproc) -k || true
    mkdir -p /app/cuda-samples/bin
    if [ -d bin ]; then
        find bin -type f -executable -exec cp {} /app/cuda-samples/bin/ \;
    fi
    find Samples -type f -executable ! -name '*.sh' ! -name 'Makefile' -path '*/release/*' -exec cp {} /app/cuda-samples/bin/ \; || true
fi
rm -rf /tmp/cuda-samples
EOT

FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu22.04 AS runtime
ARG TORCH_VERSION=2.4
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    openmpi-bin \
    git \
    build-essential \
    libfreeimage3 \
    libglfw3 \
    libgles2 \
    $(if [ "$TARGETARCH" = "arm64" ]; then echo "gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"; fi) \
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
