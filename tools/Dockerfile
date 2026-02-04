# Use the official Ubuntu base image
FROM ubuntu:22.04

# Avoid prompts from apt during build
ARG DEBIAN_FRONTEND=noninteractive

# Install necessary tools and Python in fewer layers
RUN apt-get update && \
    apt-get install -y \
    wget \
    software-properties-common \
    python3-pip \
    libsqlite3-dev \
    binutils \
    patchelf \
    upx-ucl && \
    # Install LLVM/Clang 20
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 20 && \
    ln -sf /usr/bin/clang-20 /usr/bin/clang && \
    ln -sf /usr/bin/lld-20 /usr/bin/lld && \
    # Add deadsnakes PPA for Python 3.11
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    libpython3.11-dev && \
    # Install libc++ and libc++abi for clang-20
    apt-get install -y libc++-20-dev libc++abi-20-dev && \
    # Clean up to reduce image size
    rm -rf /var/lib/apt/lists/* llvm.sh

# Set Python 3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create a virtual environment in a separate directory
RUN python3.11 -m venv /opt/build-env

# Install build tools in the virtual environment
RUN /opt/build-env/bin/pip install --upgrade pip
RUN /opt/build-env/bin/pip install pyinstaller

# Copy requirements.txt and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN /opt/build-env/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory in the container
WORKDIR /build

# Set environment variables
ENV CC=clang
ENV VIRTUAL_ENV=/opt/build-env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Command to run when the container starts
CMD ["bash"]
