#!/usr/bin/env python3

import argparse
import os

import h5py
import numpy as np


def check_shapes_and_dtypes(input_files, dataset_name):
    dtypes = []
    column_counts = []
    for file_path in input_files:
        with h5py.File(file_path, 'r') as f:
            column_counts.append(f[dataset_name].shape[1])
            dtypes.append(f[dataset_name].dtype)

    if len(set(column_counts)) != 1 or len(set(dtypes)) != 1:
        raise ValueError('All input files must have the same number of columns and dtype for the specified dataset')


def concatenate(input_files, output_file, dataset_name):
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
    parser = argparse.ArgumentParser(description='Virtually concatenate multiple H5 files with the same dataset and datatype')
    parser.add_argument('input_files', nargs='+', help='List of input H5 file paths')
    parser.add_argument('-o', '--output', required=True, help='Output H5 file path')
    parser.add_argument('--dataset', default='data', help='Name of the dataset to concatenate')
    args = parser.parse_args()

    concatenate(args.input_files, args.output, args.dataset)


if __name__ == '__main__':
    main()
