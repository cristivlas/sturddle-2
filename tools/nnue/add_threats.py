#!/usr/bin/env python3
"""
Upgrade a weights.bin to the --threats layout: insert a hidden_1c block
(THREAT_INPUTS x SIZE kernel + bias) between hidden_1b and pool, and append
SIZE zero input rows to the hidden_2 kernel. The hidden_1c kernel is random
(He, seeded): all-zero would be a symmetric saddle (its gradient gated by the
zero hidden_2 rows and vice versa) that fine-tuning can never leave. Eval is
still bit-exact at init since the hidden_2 rows are zero. Passes a move head
through.

Usage:
    ./add_threats.py old.bin new.bin [SIZE=32]
"""

import sys

import numpy as np

ACTIVE_INPUTS = 769
ACCUMULATOR_SIZE = 2048
POOL_SIZE = 8
MAIN_BUCKETS = 16
POOLED = ACCUMULATOR_SIZE // POOL_SIZE
THREAT_INPUTS = 768

H1A = MAIN_BUCKETS * ACTIVE_INPUTS * ACCUMULATOR_SIZE + ACCUMULATOR_SIZE
H1B = 256 * POOLED + POOLED
POOL = 2 * ACCUMULATOR_SIZE
H2K = POOLED * 16
TAIL = (H2K + 16) + (16 * 16 + 16) + (16 * 1 + 1)  # hidden_2, hidden_3, out
BASE = H1A + H1B + POOL + TAIL
MOVE = (ACTIVE_INPUTS * 256 + 256) + (256 * 4096 + 4096)  # move_acc, move


def main(src, dst, size):
    data = np.fromfile(src, dtype=np.float32)
    if data.size not in (BASE, BASE + MOVE):
        sys.exit(f"{src}: expected {BASE} or {BASE + MOVE} floats, got {data.size}")

    rng = np.random.default_rng(0x5EED)
    kernel = rng.normal(0.0, np.sqrt(2.0 / THREAT_INPUTS), THREAT_INPUTS * size).astype(np.float32)
    h1c = np.concatenate([kernel, np.zeros(size, dtype=np.float32)])
    h2_rows = np.zeros(size * 16, dtype=np.float32)

    cut_1c = H1A + H1B
    cut_h2 = cut_1c + POOL + H2K
    np.concatenate([data[:cut_1c], h1c, data[cut_1c:cut_h2], h2_rows, data[cut_h2:]]).tofile(dst)
    print(
        f"{dst}: {data.size} -> {data.size + h1c.size + h2_rows.size} floats, hidden_1c size {size}"
        f" ({'with' if data.size == BASE + MOVE else 'no'} move head)"
    )


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(f"Usage: {sys.argv[0]} old.bin new.bin [SIZE=32]")
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 32)
