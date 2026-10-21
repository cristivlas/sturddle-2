#!/usr/bin/env python3
import argparse

import numpy as np

ACCUMULATOR_SIZE = 1024
INPUT_ROWS = 32 * 768
L2_INPUTS = 1024
HIDDEN_2 = 32
HIDDEN_3 = 32
BANDS = 8
QUEEN_STATES = 4

HEAD = INPUT_ROWS * ACCUMULATOR_SIZE + ACCUMULATOR_SIZE
STACK = L2_INPUTS * HIDDEN_2 + HIDDEN_2 + HIDDEN_2 * HIDDEN_3 + HIDDEN_3 + HIDDEN_3 + 1


def main():
    p = argparse.ArgumentParser(
        description="Convert a fiberglass weights.bin to hourglass: each piece band's white-to-move stack, "
        "with mover-first hidden_2 rows, copied into its 4 queen-state stacks."
    )
    p.add_argument("input", help="fiberglass weights.bin")
    p.add_argument("output", help="hourglass weights.bin")
    args = p.parse_args()

    data = np.fromfile(args.input, dtype=np.float32)
    expected = HEAD + BANDS * 2 * STACK
    if data.size != expected:
        raise ValueError(f"{args.input}: expected {expected} floats, got {data.size}")

    parts = [data[:HEAD]]
    for band in range(BANDS):
        start = HEAD + (band * 2 + 1) * STACK
        stack = data[start : start + STACK]
        rows = stack[: L2_INPUTS * HIDDEN_2].reshape(L2_INPUTS, HIDDEN_2)
        half = L2_INPUTS // 2
        rows = np.concatenate([rows[half:], rows[:half]])  # mover first
        parts += [rows.ravel(), stack[L2_INPUTS * HIDDEN_2 :]] * QUEEN_STATES

    np.concatenate(parts).tofile(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
