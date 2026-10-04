#!/usr/bin/env python3
"""
Copy the move head of with_moves.bin into no_moves.bin, quantize-round export to out.bin.

Usage:
    ./add_move_head.py with_moves.bin no_moves.bin out.bin
"""

import sys

import numpy as np

from verify import LAYERS, MOVE_LAYERS, get_constraint_params


def size(layers):
    return int(sum(np.prod(k) + np.prod(b) for _, k, b, _ in layers))


def split(path):
    data = np.fromfile(path, dtype=np.float32)
    for layers in (LAYERS, [l for l in LAYERS if l[0] != "hidden_1c"]):
        n = size(layers)
        if data.size in (n, n + size(MOVE_LAYERS)):
            return layers, data[:n], data[n:]
    sys.exit(f"{path}: unrecognized size {data.size}")


def quantize(data, layers):
    out, off = [], 0
    for _, k, b, c in layers:
        qmax, qscale = get_constraint_params(c)
        for n in (int(np.prod(k)), int(np.prod(b))):
            a = data[off : off + n]
            off += n
            out.append(a if qmax is None else np.clip(np.round(a * qscale) / qscale, -qmax, qmax))
    return np.concatenate(out).astype(np.float32)


def main(with_moves, no_moves, out):
    head = split(with_moves)[2]
    layers, base, existing = split(no_moves)
    if not head.size:
        sys.exit(f"{with_moves}: no move head")
    if existing.size:
        sys.exit(f"{no_moves}: already has a move head")
    quantize(np.concatenate([base, head]), layers + MOVE_LAYERS).tofile(out)
    print(f"{out}: {base.size} + {head.size} floats")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} with_moves.bin no_moves.bin out.bin")
    main(*sys.argv[1:])
