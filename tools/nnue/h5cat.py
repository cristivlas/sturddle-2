#!/usr/bin/env python3
"""
Virtually concatenate h5 files.
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def find_h5_files(folder_path):
    """
    Generator that recursively finds all .h5 files in a folder.

    Args:
        folder_path: Path to the folder to search

    Yields:
        Absolute path strings of .h5 files
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder '{folder_path}' does not exist")

    # Recursively find all .h5 files and yield their absolute paths as strings
    for h5_file in sorted(folder.rglob('*.h5')):
        yield str(h5_file.resolve())


def check_shapes_and_dtypes(input_files, dataset_name):
    dtypes = []
    column_counts = []
    for file_path in input_files:
        with h5py.File(file_path, 'r') as f:
            if dataset_name not in f:
                raise ValueError(f"Dataset '{dataset_name}' not found in {file_path}")
            column_counts.append(f[dataset_name].shape[1])
            dtypes.append(f[dataset_name].dtype)

    if len(set(column_counts)) != 1 or len(set(dtypes)) != 1:
        raise ValueError('All input files must have the same number of columns and dtype for the specified dataset')


def concatenate(input_files, output_file, dataset_name):
    if not isinstance(input_files, list):
        input_files = list(input_files)

    if len(input_files) == 0:
        raise ValueError("No input files found")

    print(f"Found {len(input_files)} files.")

    check_shapes_and_dtypes(input_files, dataset_name)

    with h5py.File(input_files[0], 'r') as f:
        shape = f[dataset_name].shape
        dtype = f[dataset_name].dtype

    total_rows = sum([h5py.File(f, 'r')[dataset_name].shape[0] for f in input_files])
    layout = h5py.VirtualLayout(shape=(total_rows,) + shape[1:], dtype=dtype)

    with h5py.File(output_file, 'w', libver='latest') as out_f:
        current_row = 0
        for file_path in input_files:
            with h5py.File(file_path, 'r') as in_f:
                in_dset = in_f[dataset_name]
                num_rows = in_dset.shape[0]
                vsource = h5py.VirtualSource(file_path, dataset_name, shape=in_dset.shape)
                layout[current_row:current_row + num_rows, ...] = vsource[...]
                current_row += num_rows

        out_f.create_virtual_dataset(dataset_name, layout, fillvalue=None)


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description='Virtually concatenate multiple H5 files with the same dataset and datatype',
        epilog='If input_files is not specified, all .h5 files in the current directory (recursively) will be used.'
    )
    parser.add_argument('input_files', nargs='*', help='List of input H5 file paths (or omit to auto-discover)')
    parser.add_argument('-o', '--output', required=True, help='Output H5 file path')
    parser.add_argument('--dataset', default='data', help='Name of the dataset to concatenate')
    parser.add_argument('--folder', default='.', help='Folder to search for .h5 files (default: current directory)')
    args = parser.parse_args()

    # If no input files specified, find all .h5 files in the folder
    if not args.input_files:
        print(f"No input files specified, searching for .h5 files in '{args.folder}'...")
        input_files = find_h5_files(args.folder)
    else:
        input_files = args.input_files

    concatenate(input_files, args.output, args.dataset)
    print(f"Successfully created virtual dataset in {args.output}")


if __name__ == '__main__':
    main()
