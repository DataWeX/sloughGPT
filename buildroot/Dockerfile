FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies for Buildroot
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    autoconf \
    automake \
    libtool \
    patch \
    pkg-config \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    bc \
    bison \
    flex \
    libncurses-dev \
    libssl-dev \
    libelf-dev \
    cpio \
    unzip \
    rsync \
    file \
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN useradd -m -s /bin/bash builder
WORKDIR /home/builder

# Copy Buildroot configs
COPY --chown=builder:builder buildroot/ /home/builder/buildroot/

# Set permissions
RUN chmod +x /home/builder/buildroot/post-build.sh \
    && chmod +x /home/builder/buildroot/post-image.sh

# Default command
CMD ["bash", "-c", "cd /home/builder/buildroot && make sloughgpt_defconfig && make -j$(nproc)"]
