ARG UBUNTU=22.04
FROM ubuntu:${UBUNTU}

ARG LLVM_VERSION=22
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    wget \
    lsb-release \
    software-properties-common \
    gnupg \
    python3-venv && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh ${LLVM_VERSION} && \
    # llvm-profdata for --pgo, static libc++ for the link
    apt-get install -y \
    llvm-${LLVM_VERSION} \
    libc++-${LLVM_VERSION}-dev \
    libc++abi-${LLVM_VERSION}-dev && \
    rm -rf /var/lib/apt/lists/* llvm.sh

# zstandard for the bundle, chess for --pgo sampling
RUN python3 -m venv /opt/build-env && \
    /opt/build-env/bin/pip install --no-cache-dir zstandard chess

WORKDIR /build

ENV VIRTUAL_ENV=/opt/build-env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV CXX=clang++-${LLVM_VERSION}

CMD ["bash"]
