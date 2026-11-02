#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile

import h5py
import numpy as np
from tqdm import tqdm


def first_empty(data, batch_size):
    for start in tqdm(range(0, len(data), batch_size), desc="Scanning", unit=" batch"):
        empty = np.flatnonzero(~data[start : start + batch_size].any(axis=1))
        if len(empty):
            return start + empty[0]
    return len(data)


def truncate(input_path, output_path, batch_size, verbose):
    with h5py.File(input_path, "r") as src:
        data = src["data"]
        total = len(data)
        if verbose:
            print(f"{input_path}: {total:,} records, shape {data.shape}")

        rows = first_empty(data, batch_size)
        if rows == total:
            print("No empty records found.")
            return False
        if rows == 0:
            print("All records are empty.")
            return False
        print(f"First empty record at {rows:,}: keeping {rows:,} of {total:,} records.")

        with h5py.File(output_path, "w") as dst:
            dst.attrs.update(src.attrs)
            out = dst.create_dataset("data", shape=(rows,) + data.shape[1:], dtype=data.dtype)
            for start in tqdm(range(0, rows, batch_size), desc="Copying", unit=" batch"):
                end = min(start + batch_size, rows)
                out[start:end] = data[start:end]
    print(f"Wrote {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Truncate an H5 file at its first all-zero record.")
    parser.add_argument("input")
    parser.add_argument("-b", "--batch-size", type=int, default=10000, help="default: 10000")
    parser.add_argument("-o", "--output", help="default: overwrite the input")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.output and args.output != args.input:
        truncate(args.input, args.output, args.batch_size, args.verbose)
        return

    fd, tmp = tempfile.mkstemp(suffix=".h5", dir=os.path.dirname(os.path.abspath(args.input)))
    os.close(fd)
    try:
        if truncate(args.input, tmp, args.batch_size, args.verbose):
            shutil.move(tmp, args.input)
            print(f"Replaced {args.input}")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


if __name__ == "__main__":
    main()
