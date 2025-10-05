#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import sys
import os
from tqdm import tqdm

def count_high_scores(h5_file_path, threshold, batch_size=100000):
    """
    Count records in H5 file where abs(score) > threshold with progress bar

    Args:
        h5_file_path: Path to the H5 file
        threshold: Score threshold value
        batch_size: Number of records to process at once
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            data = f['data']
            total_count = data.shape[0]

            print(f"Total records: {total_count:,}")
            print(f"Processing in batches of {batch_size:,}")
            print(f"Counting records where abs(score) >= {threshold}")
            print("-" * 50)

            count_above = 0

            # Process in batches with progress bar
            with tqdm(total=total_count, desc="Processing", unit=" rec") as pbar:
                for start_idx in range(0, total_count, batch_size):
                    end_idx = min(start_idx + batch_size, total_count)

                    # Load batch of scores (index 13) and cast to signed int64
                    batch_scores = data[start_idx:end_idx, 13].astype(np.int64)

                    # Count in this batch
                    batch_count = np.sum(np.abs(batch_scores) >= threshold)
                    count_above += batch_count

                    # Update progress
                    processed = end_idx
                    percentage = (count_above / processed * 100) if processed > 0 else 0

                    pbar.set_postfix({
                        'Found': f'{count_above:,}',
                        'Rate': f'{percentage:.2f}%'
                    })
                    pbar.update(end_idx - start_idx)

            final_percentage = (count_above / total_count * 100) if total_count > 0 else 0

            print("\nFinal Results:")
            print(f"Records with abs(score) >= {threshold}: {count_above:,}")
            print(f"Percentage of total: {final_percentage:.2f}%")

    except FileNotFoundError:
        print(f"Error: File '{h5_file_path}' not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: 'data' dataset not found in '{h5_file_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Count records in H5 chess data file where abs(score) >= threshold'
    )
    parser.add_argument(
        'input_file',
        help='Path to the H5 file to analyze'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=3000,
        help='Score threshold (count records where abs(score) > threshold) (default: 3000)'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=100000,
        help='Batch size for processing (default: 100000)'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    print(f"Analyzing file: {args.input_file}")

    count_high_scores(args.input_file, args.threshold, args.batch_size)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
