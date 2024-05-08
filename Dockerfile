# Use the official Ubuntu base image
FROM ubuntu:22.04

# Avoid prompts from apt during build
ARG DEBIAN_FRONTEND=noninteractive

# Install necessary tools and Python
RUN apt-get update && \
    apt-get install -y wget software-properties-common python3-pip && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 && \
    ln -s /usr/bin/clang-18 /usr/bin/clang && \
    ln -s /usr/bin/lld-18 /usr/bin/lld && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-distutils python3.11-venv libpython3.11-dev

# Install libc++ and libc++abi for clang-18
RUN apt-get install -y libc++-18-dev libc++abi-18-dev libsqlite3-dev lld

# Set Python 3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create a virtual environment in a separate directory
RUN python3.11 -m venv /opt/build-env

# Install Cython in the virtual environment
RUN /opt/build-env/bin/pip install cython

# Set the working directory in the container
WORKDIR /build

# Set environment variables
ENV CC=clang
# Environment variable to activate the virtual environment on container start
ENV VIRTUAL_ENV=/opt/build-env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Command to run when the container starts.
CMD ["bash"]
