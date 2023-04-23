#!/usr/bin/env python3

import argparse
import h5py
import numpy as np


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description='Concatenate multiple npy files into a virtual H5 dataset')
    parser.add_argument('input_files', nargs='+',
                        help='List of input npy file paths')
    parser.add_argument('-o', '--output', required=True,
                        help='Output H5 file path')
    parser.add_argument('--hot_encoding', type=int, default=769,
                        help='Size of the one-hot encoding vectors')
    parser.add_argument('--half', action='store_true',
                        help='Use float16 dtype')
    args = parser.parse_args()

    # Determine the dtype to use
    dtype = np.float16 if args.half else np.float32

    # Determine the shape of each row
    row_shape = (args.hot_encoding + 1,)

    # Calculate the number of rows in each input file
    row_counts = [np.memmap(file_path, dtype=dtype, mode='r').shape[0] //
                  (args.hot_encoding + 1) for file_path in args.input_files]

    # Create a new H5 file
    with h5py.File(args.output, 'w') as f:
        # Create a virtual dataset with the same shape as the original dataset
        vds_shape = (
            sum(row_counts),
            args.hot_encoding + 1,
        )
        vds_layout = h5py.VirtualLayout(shape=vds_shape, dtype=dtype)

        # Iterate through the input files and populate the virtual dataset
        start_row = 0
        for file_path, num_rows in zip(args.input_files, row_counts):
            data = np.memmap(file_path, dtype=dtype, mode='r')
            intermediate_h5_file = f'{file_path}.h5'

            with h5py.File(intermediate_h5_file, 'w') as h5f:
                h5f.create_dataset('data', data=data)

            vsource = h5py.VirtualSource(intermediate_h5_file, 'data', shape=(
                num_rows,) + row_shape, dtype=dtype)

            # Populate the virtual dataset with the virtual source
            vds_layout[start_row: start_row + num_rows, :] = vsource

            start_row += num_rows

        f.create_virtual_dataset('data', vds_layout)


if __name__ == '__main__':
    main()
