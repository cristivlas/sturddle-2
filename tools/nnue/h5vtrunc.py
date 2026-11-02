#!/usr/bin/env python3
import argparse
import os
import sys

import h5py


def main():
    parser = argparse.ArgumentParser(description="Virtual H5 file over the first N records of a dataset.")
    parser.add_argument("input_file")
    parser.add_argument("-n", "--num-records", type=int, required=True)
    parser.add_argument("-o", "--output-file", help="default: <input>_truncated.h5")
    parser.add_argument("-d", "--dataset-name", default="data")
    args = parser.parse_args()

    if args.num_records <= 0:
        parser.error("--num-records must be > 0")
    if not os.path.isfile(args.input_file):
        parser.error(f"{args.input_file} not found")

    source = os.path.abspath(args.input_file)
    output = args.output_file or os.path.splitext(os.path.basename(source))[0] + "_truncated.h5"
    name = args.dataset_name

    with h5py.File(source, "r") as src:
        if name not in src:
            sys.exit(f"{args.input_file}: no dataset '{name}' (has: {list(src.keys())})")
        data = src[name]
        total = data.shape[0]
        if args.num_records > total:
            sys.exit(f"{args.input_file}: requested {args.num_records:,} records, has {total:,}")

        layout = h5py.VirtualLayout(shape=(args.num_records,) + data.shape[1:], dtype=data.dtype)
        layout[:] = h5py.VirtualSource(source, name, shape=data.shape)[: args.num_records]
        with h5py.File(output, "w") as dst:
            dst.create_virtual_dataset(name, layout)
            dst[name].attrs.update(data.attrs)

    print(f"{output}: {args.num_records:,} of {total:,} records from {source}")


if __name__ == "__main__":
    main()
