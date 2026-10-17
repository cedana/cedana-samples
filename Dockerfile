ARG CUDA_VERSION=12.4

FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu22.04 AS builder

ARG CUDA_VERSION
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
# CUDA 13 dropped Volta (compute_70); Turing (compute_75) is its minimum.
CUDA_ARCH=70
if [ "${CUDA_VERSION%%.*}" -ge 13 ]; then CUDA_ARCH=75; fi
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build build --parallel $(nproc)
find /app/gpu_smr/build -type f -executable -exec mv {} /app/gpu_smr \;
rm -rf /app/gpu_smr/build

cd /app/cpu_smr/mpi
mpicc -O3 mpi_pi_loop.c -o mpi_pi_loop

find /app -name "*.o" -delete
find /app -name "*.a" -delete
EOT


RUN /app/gpu_smr/cuda-samples/build.sh "${CUDA_VERSION}"

# Collect only the native CUDA workloads (no python/pytorch/compbio) for the
# slim `smr` image, laid out as /cedana-samples/gpu_smr/... like cedana's CI image.
FROM builder AS smr-files
RUN <<EOT
set -eux
out=/smr/cedana-samples/gpu_smr
mkdir -p "${out}"
for src in /app/gpu_smr/*.cu; do
    cp "${src%.cu}" "${out}/"
done
if [ -d /app/gpu_smr/cuda-samples/bin ]; then
    mkdir -p "${out}/cuda-samples/bin"
    cp /app/gpu_smr/cuda-samples/samples.txt "${out}/cuda-samples/"
    for s in $(sed 's/#.*//; s/[[:space:]]//g; /^$/d' /app/gpu_smr/cuda-samples/samples.txt); do
        cp "/app/gpu_smr/cuda-samples/bin/${s}" "${out}/cuda-samples/bin/"
    done
fi
EOT

# Slim CUDA workload image (customer-shaped: CUDA base + app binaries only).
# Build with `--target smr`; the default (last) stage remains `runtime`.
FROM nvidia/cuda:${CUDA_VERSION}.0-base-ubuntu22.04 AS smr
COPY --from=smr-files /smr/ /
WORKDIR /cedana-samples/gpu_smr
CMD ["/bin/sh"]

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
