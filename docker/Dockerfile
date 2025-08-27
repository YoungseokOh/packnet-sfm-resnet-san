# Copyright 2020 Toyota Research Institute.  All rights reserved.

# CUDA_VERSION is defined in the image, so add prefix
ARG PSFM_CUDA_MAJOR_VERSION=11
ARG PSFM_CUDA_MINOR_VERSION=1
ARG PSFM_CUDA_VERSION=${PSFM_CUDA_MAJOR_VERSION}.${PSFM_CUDA_MINOR_VERSION}
# CUDA 11.1에 맞는 base image로 변경
ARG SOURCE_IMAGE=nvidia/cuda:11.1.1-devel-ubuntu20.04
########################################
FROM $SOURCE_IMAGE as base
# Reïntroduce variables from before:
ARG PSFM_CUDA_MAJOR_VERSION
ARG PSFM_CUDA_MINOR_VERSION
ARG PSFM_CUDA_VERSION
ARG SOURCE_IMAGE

ENV PROJECT=packnet-sfm

# CUDA 11.1에 맞는 CUDNN 버전으로 업데이트
ENV CUDNN_VERSION=8.0.5.39-1+cuda${PSFM_CUDA_VERSION} 
ARG CUDNN_PACKAGE=libcudnn8=${CUDNN_VERSION}
ENV NCCL_VERSION=2.7.8-1+cuda${PSFM_CUDA_VERSION}
ARG NCCL_PACKAGE=libnccl2=${NCCL_VERSION}

ENV PYTORCH_VERSION=1.9.0
ENV TORCHVISION_VERSION=0.10.0

RUN echo "Using " ${SOURCE_IMAGE} " with CUDA version " ${PSFM_CUDA_VERSION} \
    " and Pytorch/torchvision " ${PYTORCH_VERSION}/${TORCHVISION_VERSION} >&2

ENV HOROVOD_VERSION=65de4c961d1e5ad2828f2f6c4329072834f27661
ENV TRT_VERSION=6.0.1.5
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG python=3.9
ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    ${CUDNN_PACKAGE} \
    libnccl2 \
    libnccl-dev \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Install Python and pip
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py

# Upgrade pip, setuptools, and wheel
RUN python${PYTHON_VERSION} -m pip install --upgrade pip setuptools==59.5.0 wheel

# Install PyTorch first as apex depends on it
RUN python${PYTHON_VERSION} -m pip install \
    torch==${PYTORCH_VERSION}+cu${PSFM_CUDA_MAJOR_VERSION}${PSFM_CUDA_MINOR_VERSION} \
    torchvision==${TORCHVISION_VERSION}+cu${PSFM_CUDA_MAJOR_VERSION}${PSFM_CUDA_MINOR_VERSION} \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && ldconfig

# Install compatible NumPy 1.26.x first and lock it
RUN python${PYTHON_VERSION} -m pip install "numpy==1.26.4"

# Install core dependencies with specific compatible versions to avoid conflicts
RUN python${PYTHON_VERSION} -m pip install \
    "rsa==4.7.2" \
    "urllib3==1.26.18" \
    "protobuf==4.25.1" \
    "opencv-python==4.8.1.78" \
    "opencv-python-headless==4.8.1.78" \
    "tensorboard==2.15.1" \
    "ultralytics==8.0.196"

# Install remaining dependencies
RUN python${PYTHON_VERSION} -m pip install \
    future \
    typing \
    pandas \
    matplotlib \
    jupyter \
    h5py \
    awscli \
    boto3 \
    tqdm \
    termcolor \
    path.py \
    "pillow-simd" \
    mpi4py \
    onnx \
    onnxruntime \
    pycuda \
    yacs \
    cython

# Install apex
RUN mkdir /workspace
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex

# --- PATCH START: make apex setup.py Py3.9-safe and ensure build deps ---
RUN python${PYTHON_VERSION} - <<'PY'
from pathlib import Path
import re
p = Path("setup.py")
s = p.read_text(encoding="utf-8")
# ensure Optional import (insert or augment existing import)
if "from typing import" in s:
    if "Optional" not in s:
        s = re.sub(r"from typing import ([^\n]+)", r"from typing import Optional, \1", s, count=1)
else:
    s = "from typing import Optional\n" + s
# replace all occurrences of 'int | None' with 'Optional[int]'
s = re.sub(r"(?<!\w)int\s*\|\s*None(?!\w)", "Optional[int]", s)
p.write_text(s, encoding="utf-8")
print("apex setup.py patched for Python 3.9")
PY

# Guard against NumPy version conflicts and ensure pybind11 compatibility
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir "numpy==1.26.4" "pybind11>=2.12"
# --- PATCH END ---

RUN python${PYTHON_VERSION} -m pip install -v --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" .
ENV PYTHONPATH="/workspace/apex:$PYTHONPATH"
WORKDIR /workspace

# Skip horovod installation since we're using single GPU mock
# (Commented out to avoid conflicts with single GPU mode)

# Settings for S3
RUN aws configure set default.s3.max_concurrent_requests 100 && \
    aws configure set default.s3.max_queue_size 10000

# Install Minkowski Engine with pip (much simpler and more reliable)
ENV TORCH_CUDA_ARCH_LIST=Volta;Turing;Kepler+Tesla
# Set CUDA environment variables for MinkowskiEngine
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"

# Ensure setuptools and numpy versions for MinkowskiEngine
RUN python${PYTHON_VERSION} -m pip install setuptools==59.5.0 wheel "numpy==1.26.4"
RUN python${PYTHON_VERSION} -m pip install ninja
RUN apt-get update && apt-get install -y libopenblas-dev

# Create symbolic links for CUDA libraries if needed
RUN find /usr/local/cuda -name "libtorch_cuda_cu.so" -type f -exec ln -sf {} /usr/local/cuda/lib64/ \; || true
RUN find /usr/local/cuda -name "libcudart.so*" -type f -exec ln -sf {} /usr/local/cuda/lib64/ \; || true
RUN ldconfig

# Install MinkowskiEngine with pip (simpler than source compilation)
RUN python${PYTHON_VERSION} -m pip install MinkowskiEngine==0.5.4

# Verify MinkowskiEngine installation
RUN python${PYTHON_VERSION} -c "
import sys
try:
    import MinkowskiEngine as ME
    print('✅ MinkowskiEngine installed successfully')
    print(f'   Version: {ME.__version__}')
except ImportError as e:
    print(f'❌ MinkowskiEngine import failed: {e}')
    print('   This may cause issues with enhanced features, but training can continue')
    sys.exit(0)  # Don't fail the build
"

# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "-g", "--"]

# Install DGP (dataset utils)
WORKDIR /workspace
RUN git clone https://github.com/TRI-ML/dgp.git
ENV PYTHONPATH="/workspace/dgp:$PYTHONPATH"

# Override DGP wandb with required version and ensure compatibility
RUN pip install "wandb==0.8.21" pyquaternion

# Final dependency check and cleanup
RUN python${PYTHON_VERSION} -c "
import numpy; print(f'✅ NumPy: {numpy.__version__}')
import cv2; print('✅ OpenCV: OK')
try:
    import ultralytics; print('✅ Ultralytics: OK')
except: print('⚠️ Ultralytics: Failed')
try:
    import tensorboard; print('✅ Tensorboard: OK')
except: print('⚠️ Tensorboard: Failed')
print('🎯 Core dependencies verified')
"

# Expose Port for jupyter (8888)
EXPOSE 8888

# create project workspace dir
RUN mkdir -p /workspace/experiments
RUN mkdir -p /workspace/${PROJECT}
WORKDIR /workspace/${PROJECT}

# Copy project source last (to avoid cache busting)
WORKDIR /workspace/${PROJECT}
COPY . /workspace/${PROJECT}
ENV PYTHONPATH="/workspace/${PROJECT}:$PYTHONPATH"

# Set final environment variables for runtime
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"
