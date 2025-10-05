#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import os
import shutil
import tempfile
from tqdm import tqdm

def find_first_empty_record(dataset):
    """Find the index of the first empty record in the dataset."""
    total_records = dataset.shape[0]

    print("Scanning for first empty record...")
    with tqdm(total=total_records, desc="Scanning records") as pbar:
        # Process in batches to improve performance
        batch_size = 5000
        for start in range(0, total_records, batch_size):
            end = min(start + batch_size, total_records)
            batch = dataset[start:end]

            # Find empty records in this batch (all zeros)
            empty_indices = np.where(~np.any(batch != 0, axis=1))[0]

            if len(empty_indices) > 0:
                # We found an empty record in this batch
                first_empty = start + empty_indices[0]
                pbar.update(pbar.total - pbar.n)  # Complete the progress bar
                return first_empty

            pbar.update(end - start)

    # No empty records found
    return total_records

def truncate_h5_file(input_path, output_path, verbose=False):
    """Truncate an HDF5 file to remove empty records at the end."""
    with h5py.File(input_path, 'r') as in_file:
        # Assume 'data' is the main dataset
        dataset = in_file['data']
        total_records = dataset.shape[0]

        if verbose:
            print(f"Original file has {total_records} records with shape {dataset.shape}")

        # Find the first empty record
        first_empty_index = find_first_empty_record(dataset)

        if first_empty_index == total_records:
            print("No empty records found in the file.")
            return False

        if first_empty_index == 0:
            print("All records are empty in the file!")
            return False

        valid_record_count = first_empty_index
        print(f"Found {valid_record_count} valid records out of {total_records} total.")
        print(f"First empty record found at index {first_empty_index}")

        # Create a new file with only the valid records
        with h5py.File(output_path, 'w') as out_file:
            # Copy over all attributes and other datasets/groups
            for key in in_file.attrs:
                out_file.attrs[key] = in_file.attrs[key]

            # Create a new dataset with the correct size
            new_dataset = out_file.create_dataset(
                'data',
                shape=(valid_record_count, dataset.shape[1]),
                dtype=dataset.dtype
            )

            # Copy data in batches
            batch_size = 10000
            with tqdm(total=valid_record_count, desc="Copying valid records") as pbar:
                for i in range(0, valid_record_count, batch_size):
                    end = min(i + batch_size, valid_record_count)
                    new_dataset[i:end] = dataset[i:end]
                    pbar.update(end - i)

            print(f"Successfully truncated file from {total_records} to {valid_record_count} records.")
            print(f"Output saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Truncate HDF5 file to remove empty records at the end')
    parser.add_argument('input', help='Input HDF5 file path')
    parser.add_argument('-o', '--output', help='Output HDF5 file path (defaults to overwriting input)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # If no output specified, overwrite the input file
    output_path = args.output if args.output else args.input

    # If overwriting, create a temporary file first
    if args.input == output_path:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        temp_file.close()
        try:
            if truncate_h5_file(args.input, temp_file.name, args.verbose):
                shutil.move(temp_file.name, args.input)
                print(f"Original file successfully replaced with truncated version")
            else:
                 os.unlink(temp_file.name)
        except Exception as e:
            print(f"Error during truncation: {e}")
            os.unlink(temp_file.name)
            raise
    else:
        truncate_h5_file(args.input, output_path, args.verbose)

if __name__ == '__main__':
    main()
