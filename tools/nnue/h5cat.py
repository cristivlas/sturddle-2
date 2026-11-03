#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser(description="Virtually concatenate H5 files with the same columns and dtype.")
    parser.add_argument("input_files", nargs="*", help="default: every .h5 under --folder")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--folder", default=".", help="where to look when no input files are given")
    args = parser.parse_args()

    if args.input_files:
        inputs = [str(Path(p).resolve()) for p in args.input_files]
    else:
        folder = Path(args.folder)
        if not folder.is_dir():
            parser.error(f"{args.folder} is not a folder")
        output = Path(args.output).resolve()
        inputs = [str(p.resolve()) for p in sorted(folder.rglob("*.h5")) if p.resolve() != output]
    if not inputs:
        parser.error("no input files")
    print(f"Found {len(inputs)} files.")

    shapes = []
    for path in inputs:
        with h5py.File(path, "r") as f:
            if args.dataset not in f:
                sys.exit(f"{path}: no dataset '{args.dataset}'")
            shapes.append((f[args.dataset].shape, f[args.dataset].dtype))
    if len({(shape[1:], dtype) for shape, dtype in shapes}) != 1:
        sys.exit("All input files must have the same number of columns and dtype")

    rows = sum(shape[0] for shape, _ in shapes)
    first_shape, dtype = shapes[0]
    layout = h5py.VirtualLayout(shape=(rows,) + first_shape[1:], dtype=dtype)
    start = 0
    for path, (shape, _) in zip(inputs, shapes):
        layout[start : start + shape[0]] = h5py.VirtualSource(path, args.dataset, shape=shape)
        start += shape[0]

    with h5py.File(args.output, "w", libver="latest") as out:
        out.create_virtual_dataset(args.dataset, layout, fillvalue=None)
    print(f"{args.output}: {rows:,} rows from {len(inputs)} files")


if __name__ == "__main__":
    main()
