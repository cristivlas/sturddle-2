#!/usr/bin/env python3
"""
Add a zero hidden_1c (bishops + occupancy, 256 x POOLED + POOLED bias) between
hidden_1b and pool, for warm-starting train_torch.py. Bit-exact eval at init.
Drops the move head, which the torch trainer does not support.

Usage:
    ./add_hidden_1c.py old.bin new.bin
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
H1C = 256 * POOLED + POOLED
POOL = 2 * ACCUMULATOR_SIZE
TAIL = (POOLED * 16 + 16) + (16 * 16 + 16) + (16 * 1 + 1)  # hidden_2, hidden_3, out
BASE = H1A + H1B + POOL + TAIL
MOVE = (ACTIVE_INPUTS * 256 + 256) + (256 * 4096 + 4096)  # move_acc, move


def main(src, dst):
    data = np.fromfile(src, dtype=np.float32)
    if data.size in (BASE + H1C, BASE + H1C + MOVE):
        sys.exit(f"{src}: already has hidden_1c ({data.size} floats)")
    if data.size not in (BASE, BASE + MOVE):
        sys.exit(f"{src}: expected {BASE} or {BASE + MOVE} floats, got {data.size}")

    cut = H1A + H1B
    h1c = np.zeros(H1C, dtype=np.float32)
    np.concatenate([data[:cut], h1c, data[cut:BASE]]).tofile(dst)
    print(f"{dst}: {data.size} -> {BASE + H1C} floats{' (move head dropped)' if data.size > BASE else ''}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} old.bin new.bin")
    main(sys.argv[1], sys.argv[2])
