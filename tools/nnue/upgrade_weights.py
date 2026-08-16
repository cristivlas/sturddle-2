#!/usr/bin/env python3
"""
Upgrade a 2.6.0 weights.bin to the 2.6.1 layout: insert the STM pool block
(2 x ACCUMULATOR_SIZE floats at 1/POOL_SIZE == average pooling) between
hidden_1b and hidden_2. Bit-exact eval at init. Passes a move head through.

Usage:
    ./upgrade_weights.py old.bin new.bin
"""

import sys

import numpy as np

ACTIVE_INPUTS = 769
ACCUMULATOR_SIZE = 2048
POOL_SIZE = 8
MAIN_BUCKETS = 16
POOLED = ACCUMULATOR_SIZE // POOL_SIZE

H1A = MAIN_BUCKETS * ACTIVE_INPUTS * ACCUMULATOR_SIZE + ACCUMULATOR_SIZE
H1B = 256 * POOLED + POOLED
TAIL = (POOLED * 16 + 16) + (16 * 16 + 16) + (16 * 1 + 1)  # hidden_2, hidden_3, out
BASE = H1A + H1B + TAIL
MOVE = (ACTIVE_INPUTS * 256 + 256) + (256 * 4096 + 4096)  # move_acc, move
POOL = 2 * ACCUMULATOR_SIZE


def main(src, dst):
    data = np.fromfile(src, dtype=np.float32)
    if data.size in (BASE + POOL, BASE + POOL + MOVE):
        sys.exit(f"{src}: already has a pool block ({data.size} floats)")
    if data.size not in (BASE, BASE + MOVE):
        sys.exit(f"{src}: expected {BASE} or {BASE + MOVE} floats, got {data.size}")

    cut = H1A + H1B
    pool = np.full(POOL, 1.0 / POOL_SIZE, dtype=np.float32)
    np.concatenate([data[:cut], pool, data[cut:]]).tofile(dst)
    print(f"{dst}: {data.size} -> {data.size + POOL} floats"
          f" ({'with' if data.size == BASE + MOVE else 'no'} move head)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} old.bin new.bin")
    main(sys.argv[1], sys.argv[2])
