# Copyright 2020 Toyota Research Institute.  All rights reserved.

# CUDA_VERSION is defined in the image, so add prefix
ARG PSFM_CUDA_MAJOR_VERSION=11
ARG PSFM_CUDA_MINOR_VERSION=1
ARG PSFM_CUDA_VERSION=${PSFM_CUDA_MAJOR_VERSION}.${PSFM_CUDA_MINOR_VERSION}
ARG SOURCE_IMAGE=nvidia/cuda:11.3.1-devel-ubuntu20.04
########################################
FROM $SOURCE_IMAGE as base
# Reïntroduce variables from before:
ARG PSFM_CUDA_MAJOR_VERSION
ARG PSFM_CUDA_MINOR_VERSION
ARG PSFM_CUDA_VERSION
ARG SOURCE_IMAGE

ENV PROJECT=packnet-sfm

# https://lambdalabs.com/blog/install-tensorflow-and-pytorch-on-rtx-30-series/
# 3090 => CUDA 11.1 OK
#    => cuDNN 8.2.1

# From https://github.com/pytorch/vision:
# Torch 1.9.0 <=> Torchvision 0.10.0

# From https://pytorch.org/get-started/locally/
# Torch 1.9.0 => CUDA 11.1 || CUDA 10.2

#ENV CUDNN_VERSION=7.6.5.32-1+cuda10.2
#ENV NCCL_VERSION=2.7.8-1+cuda10.2

ENV CUDNN_VERSION=8.0.5.39-1+cuda${PSFM_CUDA_VERSION} 
# Use the variable as a sanity check that we're using the right CUDA
ARG CUDNN_PACKAGE=libcudnn8=${CUDNN_VERSION}
ENV NCCL_VERSION=2.7.8-1+cuda${PSFM_CUDA_VERSION}
ARG NCCL_PACKAGE=libnccl2=${NCCL_VERSION} # Note: This ARG might not be directly used in apt-get if changed below

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

# Instal Python and pip
# The following block for python3.6-distutils is specific to Python 3.6.
# For Python 3.8, python3.8-dev usually includes necessary components.
# If issues arise with pip or setuptools, consider adding python3-setuptools or python3.8-distutils.
# RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
#     apt-get install -y python${PYTHON_VERSION}-distutils; \
#     fi

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

# Install Pydata and other deps, ensuring NumPy 1.x
# Install numpy first and separately to ensure its version.
RUN python${PYTHON_VERSION} -m pip install numpy==1.21.6
RUN python${PYTHON_VERSION} -m pip install future typing pandas matplotlib jupyter h5py \
    awscli boto3 tqdm termcolor path.py pillow-simd opencv-python-headless \
    mpi4py onnx onnxruntime pycuda yacs cython

# Install apex
RUN mkdir /workspace
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex
RUN python${PYTHON_VERSION} -m pip install -v --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" .
ENV PYTHONPATH="/workspace/apex:$PYTHONPATH"
WORKDIR /workspace

# install horovod (for distributed training)
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 \
    python${PYTHON_VERSION} -m pip install --no-cache-dir git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION} && \
    ldconfig

# Settings for S3
RUN aws configure set default.s3.max_concurrent_requests 100 && \
    aws configure set default.s3.max_queue_size 10000

# Install Minkowski Engine
ENV TORCH_CUDA_ARCH_LIST=Volta;Turing;Kepler+Tesla
# Ensure setuptools is < 60 before Minkowski Engine, and numpy is 1.x
RUN python${PYTHON_VERSION} -m pip install setuptools==59.5.0 wheel numpy==1.21.6 # Re-ensure versions
RUN python${PYTHON_VERSION} -m pip install ninja
RUN apt-get update && apt-get install -y libopenblas-dev
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git
RUN cd /workspace/MinkowskiEngine && \
    python${PYTHON_VERSION} setup.py install --force_cuda

# Add Tini (cf. https://github.com/jupyter/docker-stacks)
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "-g", "--"]

# Install DGP (dataset utils)
WORKDIR /workspace
RUN git clone https://github.com/TRI-ML/dgp.git
ENV PYTHONPATH="/workspace/dgp:$PYTHONPATH"

# Override DGP wandb with required version
RUN pip install wandb==0.8.21 pyquaternion

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