#!/bin/bash
# Run from the repo root; arguments go to build.py, e.g. --pgo
set -e
LLVM_VERSION=${LLVM_VERSION:-22}
IMAGE=sturddle-native-builder:llvm-${LLVM_VERSION}

docker build --build-arg LLVM_VERSION=${LLVM_VERSION} -t ${IMAGE} tools/native-bundle
docker run --rm \
    -v "$(pwd)":/build \
    -u $(id -u):$(id -g) \
    -e PYTHONUNBUFFERED=1 \
    ${IMAGE} python3 tools/native-bundle/build.py "$@"
