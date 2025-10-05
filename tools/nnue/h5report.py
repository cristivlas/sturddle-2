#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import sys
import os
from tqdm import tqdm


def analyze_chess_data(h5_file_path, threshold, batch_size):
    """
    Analyze chess data in H5 file with detailed evaluation and outcome statistics

    Args:
        h5_file_path: Path to the H5 file
        threshold: Score threshold value for high score counting
        batch_size: Number of records to process at once
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            data = f['data']
            total_count = data.shape[0]

            print(f"Total records: {total_count:,}")
            print(f"Processing in batches of {batch_size:,}")
            print(f"Analyzing evaluation scores (col 13) and outcomes (col 14)")
            print("-" * 60)

            # Initialize counters
            count_above_threshold = 0
            count_zero = 0
            count_positive = 0
            count_negative = 0

            # Outcome counters (2=win, 1=draw, 0=loss)
            count_wins = 0
            count_draws = 0
            count_losses = 0

            # Process in batches with progress bar
            with tqdm(total=total_count, desc="Processing", unit=" rec") as pbar:
                for start_idx in range(0, total_count, batch_size):
                    end_idx = min(start_idx + batch_size, total_count)

                    # Load batch of scores (index 13) and outcomes (index 14)
                    # Cast uint64 to int64 for proper signed arithmetic
                    batch_scores = data[start_idx:end_idx, 13].astype(np.int64)
                    batch_outcomes = data[start_idx:end_idx, 14].astype(np.int64)

                    # Count high scores (abs value >= threshold)
                    batch_count_above = np.sum(np.abs(batch_scores) >= threshold)
                    count_above_threshold += batch_count_above

                    # Count evaluation score categories
                    batch_zero = np.sum(batch_scores == 0)
                    batch_positive = np.sum(batch_scores > 0)
                    batch_negative = np.sum(batch_scores < 0)

                    count_zero += batch_zero
                    count_positive += batch_positive
                    count_negative += batch_negative

                    # Count game outcomes (2=win, 1=draw, 0=loss)
                    batch_wins = np.sum(batch_outcomes == 2)
                    batch_draws = np.sum(batch_outcomes == 1)
                    batch_losses = np.sum(batch_outcomes == 0)

                    count_wins += batch_wins
                    count_draws += batch_draws
                    count_losses += batch_losses

                    # Update progress
                    processed = end_idx
                    high_score_rate = (count_above_threshold / processed * 100) if processed > 0 else 0

                    pbar.set_postfix({
                        'High': f'{count_above_threshold:,}',
                        'Rate': f'{high_score_rate:.1f}%'
                    })
                    pbar.update(end_idx - start_idx)

            # Calculate percentages
            def safe_percentage(count, total):
                return (count / total * 100) if total > 0 else 0

            print("\n" + "="*60)
            print("EVALUATION SCORE ANALYSIS")
            print("="*60)
            print(f"Records with abs(score) >= {threshold:,}: {count_above_threshold:,} ({safe_percentage(count_above_threshold, total_count):.2f}%)")
            print()
            print("Score Distribution:")
            print(f"  Zero scores (score = 0):     {count_zero:,} ({safe_percentage(count_zero, total_count):.2f}%)")
            print(f"  Positive scores (score > 0): {count_positive:,} ({safe_percentage(count_positive, total_count):.2f}%)")
            print(f"  Negative scores (score < 0): {count_negative:,} ({safe_percentage(count_negative, total_count):.2f}%)")

            # Verify totals
            score_total = count_zero + count_positive + count_negative
            print(f"  Total verified:              {score_total:,}")
            if score_total != total_count:
                print(f"  ⚠️  WARNING: Score totals don't match! Missing: {total_count - score_total:,}")

            print("\n" + "="*60)
            print("GAME OUTCOME ANALYSIS")
            print("="*60)
            print("Outcome Distribution:")
            print(f"  Wins (outcome = 2):   {count_wins:,} ({safe_percentage(count_wins, total_count):.2f}%)")
            print(f"  Draws (outcome = 1):  {count_draws:,} ({safe_percentage(count_draws, total_count):.2f}%)")
            print(f"  Losses (outcome = 0): {count_losses:,} ({safe_percentage(count_losses, total_count):.2f}%)")

            # Verify outcome totals
            outcome_total = count_wins + count_draws + count_losses
            print(f"  Total verified:       {outcome_total:,}")
            if outcome_total != total_count:
                print(f"  ⚠️  WARNING: Outcome totals don't match! Missing: {total_count - outcome_total:,}")

                # Check for unexpected outcome values
                print("\n  Checking for unexpected outcome values...")
                unexpected_outcomes = 0
                with tqdm(total=total_count, desc="Checking outcomes", unit=" rec") as pbar:
                    for start_idx in range(0, total_count, batch_size):
                        end_idx = min(start_idx + batch_size, total_count)
                        batch_outcomes = data[start_idx:end_idx, 14].astype(np.int64)

                        # Count outcomes that are not 0, 1, or 2
                        batch_unexpected = np.sum(~np.isin(batch_outcomes, [0, 1, 2]))
                        unexpected_outcomes += batch_unexpected

                        pbar.update(end_idx - start_idx)

                if unexpected_outcomes > 0:
                    print(f"  Found {unexpected_outcomes:,} records with unexpected outcome values")

            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Total records analyzed: {total_count:,}")
            print(f"High absolute scores (>= {threshold:,}): {count_above_threshold:,}")
            print(f"Win rate: {safe_percentage(count_wins, total_count):.2f}%")
            print(f"Draw rate: {safe_percentage(count_draws, total_count):.2f}%")
            print(f"Loss rate: {safe_percentage(count_losses, total_count):.2f}%")

    except FileNotFoundError:
        print(f"Error: File '{h5_file_path}' not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: 'data' dataset not found in '{h5_file_path}'.")
        sys.exit(1)
    except IndexError:
        print(f"Error: Data doesn't have enough columns. Expected at least 15 columns.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze chess evaluation scores and game outcomes in H5 data file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.h5
  %(prog)s data.h5 -t 5000 -b 50000

Note: Evaluation scores are in column 13, outcomes in column 14.
Outcomes: 2=win, 1=draw, 0=loss
        """
    )
    parser.add_argument(
        'input_file',
        help='Path to the H5 file to analyze'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=3000,
        help='Score threshold for high score counting (default: 3000)'
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

    # Validate arguments
    if args.threshold < 0:
        print(f"Error: Threshold must be non-negative, got {args.threshold}")
        sys.exit(1)

    if args.batch_size <= 0:
        print(f"Error: Batch size must be positive, got {args.batch_size}")
        sys.exit(1)

    print(f"Analyzing file: {args.input_file}")
    print(f"Score threshold: {args.threshold:,}")
    print(f"Batch size: {args.batch_size:,}")
    print()

    analyze_chess_data(args.input_file, args.threshold, args.batch_size)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
