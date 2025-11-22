#!/usr/bin/env python3

import argparse
import sys
import h5py
from pathlib import Path


def create_virtual_truncation(input_file: str, num_records: int, output_file: str, dataset_name: str):
    """
    Create a virtual HDF5 file that maps to a truncated view of the original dataset.

    Args:
        input_file: Path to the original HDF5 file
        num_records: Number of records to include in the virtual view
        output_file: Path for the output virtual file (default: input_truncated.h5)
        dataset_name: Name of the dataset to truncate (default: 'data')
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    if output_file is None:
        output_file = input_path.stem + '_truncated.h5'

    # Open source file and create virtual mapping
    with h5py.File(input_file, 'r') as src:
        if dataset_name not in src:
            print(f"Error: Dataset '{dataset_name}' not found in '{input_file}'.", file=sys.stderr)
            print(f"Available datasets: {list(src.keys())}", file=sys.stderr)
            sys.exit(1)

        src_dataset = src[dataset_name]
        src_shape = src_dataset.shape
        src_dtype = src_dataset.dtype

        if num_records > src_shape[0]:
            print(f"Error: Requested {num_records} records but dataset only has {src_shape[0]}.", file=sys.stderr)
            sys.exit(1)

        # Create new shape with truncated first dimension
        new_shape = (num_records,) + src_shape[1:]

        # Create virtual layout
        layout = h5py.VirtualLayout(shape=new_shape, dtype=src_dtype)

        # Map to source - need to use absolute path for the source file
        vsource = h5py.VirtualSource(str(input_path.absolute()), dataset_name, shape=src_shape)

        # Select the truncated region
        if len(src_shape) == 1:
            layout[:] = vsource[:num_records]
        else:
            layout[:] = vsource[:num_records, ...]

        # Create output file with virtual dataset
        with h5py.File(output_file, 'w') as dst:
            dst.create_virtual_dataset(dataset_name, layout)

            # Copy attributes from source dataset
            for attr_name, attr_value in src_dataset.attrs.items():
                dst[dataset_name].attrs[attr_name] = attr_value

        print(f"Created virtual file: {output_file}")
        print(f"  Source: {input_path.absolute()}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Original shape: {src_shape}")
        print(f"  Virtual shape: {new_shape}")
        print(f"  Records: {num_records:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a virtual HDF5 file with truncated view of original data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input_file', help='Input HDF5 file')
    parser.add_argument('-n', '--num-records', type=int, help='Number of records to keep')
    parser.add_argument('-o', '--output-file', help='Output file (default: input_truncated.h5)')
    parser.add_argument('-d', '--dataset-name', nargs='?', default='data', help='Dataset name (default: data)')

    args = parser.parse_args()

    if args.num_records <= 0:
        print("Error: Number of records must be positive.", file=sys.stderr)
        sys.exit(1)

    create_virtual_truncation(
        args.input_file,
        args.num_records,
        args.output_file,
        args.dataset_name
    )


if __name__ == '__main__':
    main()
