#!/bin/bash
docker run --rm \
    -v $(pwd):/build \
    -u $(id -u):$(id -g) \
    -e PYTHONUNBUFFERED=1 \
    pyinstaller-builder bash -c "
    CXX=clang++-20 tools/build.py
"
