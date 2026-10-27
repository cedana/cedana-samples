ARG CUDA_VERSION=12.4

FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu22.04 AS devel

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

# NVIDIA's cuda-samples (~20 min): only its build script and the CUDA version
# decide it, so a change to the workloads doesn't rebuild it.
FROM devel AS cuda-samples
ARG CUDA_VERSION
COPY gpu_smr/cuda-samples/build.sh /build.sh
RUN /build.sh "${CUDA_VERSION}" /cuda-samples && mkdir -p /cuda-samples/bin

FROM devel AS builder
ARG CUDA_VERSION
WORKDIR /app

COPY cpu_smr/ /app/cpu_smr/
COPY gpu_smr/ /app/gpu_smr/

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

COPY --from=cuda-samples /cuda-samples/bin/ /app/gpu_smr/cuda-samples/bin/
COPY kubernetes/ /app/kubernetes/

# The lite image's /cedana-samples: the built workloads, without the
# cuda-samples that samples.txt doesn't list. Its CUDA workloads must run on
# the CUDA base image (libcudart only), as a customer's would.
FROM builder AS lite-files
RUN <<'EOT'
set -eu
out=/lite/cedana-samples
base_libs='^(libcudart\.so\.[0-9]+|libc\.so\.6|libstdc\+\+\.so\.6|libm\.so\.6|libgcc_s\.so\.1|libdl\.so\.2|libpthread\.so\.0|librt\.so\.1|ld-linux.*)$'
extra_libs() { readelf -d "$1" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | grep -Ev "${base_libs}" | tr '\n' ' ' || true; }

for src in /app/gpu_smr/*.cu; do
    bin="${src%.cu}"
    extra="$(extra_libs "${bin}")"
    if [ -n "${extra}" ]; then echo "ERROR: ${bin} needs ${extra}" >&2; exit 1; fi
done

mkdir -p /lite
cp -a /app "${out}"
bin=/app/gpu_smr/cuda-samples/bin
rm -rf "${out}/gpu_smr/cuda-samples/bin"
mkdir -p "${out}/gpu_smr/cuda-samples/bin"
for s in $(sed 's/#.*//; s/[[:space:]]//g; /^$/d' /app/gpu_smr/cuda-samples/samples.txt); do
    if [ -f "${bin}/${s}" ]; then cp "${bin}/${s}" "${out}/gpu_smr/cuda-samples/bin/"; fi
done
EOT

# The image cedana's tests run (`cuda-lite`): every workload, built, with the
# Kubernetes manifests and the SLURM scripts, at /cedana-samples (and /app, as
# in the other images) on the CUDA base image. Build with `--target lite`; the
# default (last) stage remains `runtime`. python3 (stdlib only, no pip;
# ~32 MiB) runs the host memory and fd samples. requirements.txt is what the
# GPU tests' PyTorch workloads install.
FROM nvidia/cuda:${CUDA_VERSION}.0-base-ubuntu22.04 AS lite
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=lite-files /lite/ /
COPY slurm/ /cedana-samples/slurm/
COPY requirements-torch2.7.txt /cedana-samples/requirements.txt
RUN ln -s /cedana-samples /app
WORKDIR /cedana-samples
ENTRYPOINT ["/bin/bash"]

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
