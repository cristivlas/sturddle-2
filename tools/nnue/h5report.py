#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import sys
import os
from tqdm import tqdm


def analyze_chess_data(h5_file_path, threshold, batch_size, generate_histogram=False, hist_bins=100, hist_range=None):
    """
    Analyze chess data in H5 file with detailed evaluation and outcome statistics

    Args:
        h5_file_path: Path to the H5 file
        threshold: Score threshold value for high score counting
        batch_size: Number of records to process at once
        generate_histogram: Whether to generate a histogram of eval scores
        hist_bins: Number of bins for the histogram
        hist_range: Tuple of (min, max) for histogram range, or None for auto
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            data = f['data']
            total_count = data.shape[0]

            print(f"Total records: {total_count:,}")
            print(f"Processing in batches of {batch_size:,}")
            print(f"Analyzing turns (col 12), evals (col 13), and outcomes (col 14)")
            print("-" * 64)

            # Initialize counters
            count_above_threshold = 0
            count_zero = 0
            count_positive = 0
            count_negative = 0

            # Outcome counters (2=win, 1=draw, 0=loss)
            count_wins = 0
            count_draws = 0
            count_losses = 0

            # Side-to-move counters (1=white, 0=black)
            count_white_to_move = 0
            count_black_to_move = 0

            # Win counters by color (side-to-move when position was evaluated)
            count_white_wins = 0  # White to move and wins
            count_black_wins = 0  # Black to move and wins
            count_white_draws = 0
            count_black_draws = 0
            count_white_losses = 0
            count_black_losses = 0

            # For histogram generation
            all_scores = [] if generate_histogram else None

            # Process in batches with progress bar
            with tqdm(total=total_count, desc="Processing", unit=" rec") as pbar:
                for start_idx in range(0, total_count, batch_size):
                    end_idx = min(start_idx + batch_size, total_count)

                    # Load batch: side-to-move (12), scores (13), outcomes (14)
                    batch_stm = data[start_idx:end_idx, 12].astype(np.int64)
                    batch_scores = data[start_idx:end_idx, 13].astype(np.int64)
                    batch_outcomes = data[start_idx:end_idx, 14].astype(np.int64)

                    # Collect scores for histogram
                    if generate_histogram:
                        all_scores.append(batch_scores)

                    # Count high scores (abs value > threshold)
                    batch_count_above = np.sum(np.abs(batch_scores) > threshold)
                    count_above_threshold += batch_count_above

                    # Count evaluation score categories
                    batch_zero = np.sum(batch_scores == 0)
                    batch_positive = np.sum(batch_scores > 0)
                    batch_negative = np.sum(batch_scores < 0)

                    count_zero += batch_zero
                    count_positive += batch_positive
                    count_negative += batch_negative

                    # Count game outcomes
                    batch_wins = np.sum(batch_outcomes == 2)
                    batch_draws = np.sum(batch_outcomes == 1)
                    batch_losses = np.sum(batch_outcomes == 0)

                    count_wins += batch_wins
                    count_draws += batch_draws
                    count_losses += batch_losses

                    # Count side-to-move (1=white, 0=black)
                    batch_white = np.sum(batch_stm == 1)
                    batch_black = np.sum(batch_stm == 0)
                    count_white_to_move += batch_white
                    count_black_to_move += batch_black

                    # Count wins/draws/losses by color
                    white_mask = batch_stm == 1
                    black_mask = batch_stm == 0

                    count_white_wins += np.sum(white_mask & (batch_outcomes == 2))
                    count_white_draws += np.sum(white_mask & (batch_outcomes == 1))
                    count_white_losses += np.sum(white_mask & (batch_outcomes == 0))

                    count_black_wins += np.sum(black_mask & (batch_outcomes == 2))
                    count_black_draws += np.sum(black_mask & (batch_outcomes == 1))
                    count_black_losses += np.sum(black_mask & (batch_outcomes == 0))

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

            print("\n" + "="*64)
            print("SIDE-TO-MOVE DISTRIBUTION")
            print("="*64)
            print(f"White to move (stm = 1): {count_white_to_move:,} ({safe_percentage(count_white_to_move, total_count):.2f}%)")
            print(f"Black to move (stm = 0): {count_black_to_move:,} ({safe_percentage(count_black_to_move, total_count):.2f}%)")

            stm_total = count_white_to_move + count_black_to_move
            print(f"Total verified:          {stm_total:,}")
            if stm_total != total_count:
                print(f"⚠️  WARNING: Side-to-move totals don't match! Missing: {total_count - stm_total:,}")

            print("\n" + "="*64)
            print("EVALUATION SCORE ANALYSIS")
            print("="*64)
            print(f"Records with abs(score) > {threshold:,}: {count_above_threshold:,} ({safe_percentage(count_above_threshold, total_count):.2f}%)")
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

            print("\n" + "="*64)
            print("GAME OUTCOME ANALYSIS")
            print("="*64)
            print("Side-to-Move Outcome Distribution")
            print(f"  Wins:   {count_wins:,} ({safe_percentage(count_wins, total_count):.2f}%)")
            print(f"  Draws:  {count_draws:,} ({safe_percentage(count_draws, total_count):.2f}%)")
            print(f"  Losses: {count_losses:,} ({safe_percentage(count_losses, total_count):.2f}%)")

            # Verify outcome totals
            outcome_total = count_wins + count_draws + count_losses
            print(f"  Total:  {outcome_total:,}")
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

            print("\n" + "="*64)
            print("OUTCOMES BY COLOR")
            print("="*64)
            print("White to Move")
            print(f"  Wins:   {count_white_wins:,} ({safe_percentage(count_white_wins, count_white_to_move):.2f}%)")
            print(f"  Draws:  {count_white_draws:,} ({safe_percentage(count_white_draws, count_white_to_move):.2f}%)")
            print(f"  Losses: {count_white_losses:,} ({safe_percentage(count_white_losses, count_white_to_move):.2f}%)")

            white_outcome_total = count_white_wins + count_white_draws + count_white_losses
            print(f"  Total:  {white_outcome_total:,}")
            if white_outcome_total != count_white_to_move:
                print(f"  ⚠️  WARNING: White outcome totals don't match!")

            print("\nBlack to Move")
            print(f"  Wins:   {count_black_wins:,} ({safe_percentage(count_black_wins, count_black_to_move):.2f}%)")
            print(f"  Draws:  {count_black_draws:,} ({safe_percentage(count_black_draws, count_black_to_move):.2f}%)")
            print(f"  Losses: {count_black_losses:,} ({safe_percentage(count_black_losses, count_black_to_move):.2f}%)")

            black_outcome_total = count_black_wins + count_black_draws + count_black_losses
            print(f"  Total:  {black_outcome_total:,}")
            if black_outcome_total != count_black_to_move:
                print(f"  ⚠️  WARNING: Black outcome totals don't match!")

            print("\n" + "="*64)
            print("SUMMARY")
            print("="*64)
            print(f"Total records analyzed: {total_count:,}")
            print(f"High absolute scores (> {threshold:,}): {count_above_threshold:,}")
            print()
            print(f"Win rate:  {safe_percentage(count_wins, total_count):.2f}%")
            print(f"Draw rate: {safe_percentage(count_draws, total_count):.2f}%")
            print(f"Loss rate: {safe_percentage(count_losses, total_count):.2f}%")
            print()
            print(f"White to move: wins {safe_percentage(count_white_wins, count_white_to_move):.2f}%")
            print(f"Black to move: wins {safe_percentage(count_black_wins, count_black_to_move):.2f}%")

            # Generate histogram if requested
            if generate_histogram:
                print("\n" + "="*64)
                print("GENERATING HISTOGRAM")
                print("="*64)
                generate_eval_histogram(all_scores, hist_bins, hist_range, h5_file_path)

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


def generate_eval_histogram(all_scores, bins, score_range, h5_file_path):
    """
    Generate and save a histogram of evaluation scores

    Args:
        all_scores: List of numpy arrays containing scores
        bins: Number of bins for the histogram
        score_range: Tuple of (min, max) or None for auto range
        h5_file_path: Original H5 file path (used for output naming)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for histogram generation.")
        print("Install it with: pip install matplotlib")
        return

    print("Concatenating score arrays...")
    scores = np.concatenate(all_scores)

    print(f"Computing histogram with {bins} bins...")

    # Compute statistics
    min_score = np.min(scores)
    max_score = np.max(scores)
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)

    print(f"Score statistics:")
    print(f"  Min:    {min_score:,}")
    print(f"  Max:    {max_score:,}")
    print(f"  Mean:   {mean_score:,.2f}")
    print(f"  Median: {median_score:,.2f}")
    print(f"  StdDev: {std_score:,.2f}")

    # Create histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Full range histogram
    if score_range:
        range_tuple = score_range
    else:
        range_tuple = (min_score, max_score)

    counts, bin_edges, patches = ax1.hist(scores, bins=bins, range=range_tuple,
                                          edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Evaluation Score', fontsize=12)
    ax1.set_ylabel('Frequency (log scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title(f'Evaluation Scores Distribution (n={len(scores):,})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:,.0f}')
    ax1.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:,.0f}')
    ax1.legend()

    # Format y-axis with commas
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Zoomed histogram (centered around 0, excluding extreme values)
    # Use 99th percentile to exclude outliers
    p99 = np.percentile(np.abs(scores), 99)
    zoom_range = (-p99, p99)
    mask = (scores >= zoom_range[0]) & (scores <= zoom_range[1])
    zoomed_scores = scores[mask]

    counts2, bin_edges2, patches2 = ax2.hist(zoomed_scores, bins=bins, range=zoom_range,
                                             edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Evaluation Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Zoomed Distribution (±99th percentile, ±{p99:,.0f})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()

    # Generate output filename
    base_name = os.path.splitext(os.path.basename(h5_file_path))[0]
    output_file = f"{base_name}_eval_histogram.png"

    print(f"Saving histogram to: {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Histogram saved successfully!")

    # Also save histogram data as text file
    hist_data_file = f"{base_name}_eval_histogram.txt"
    print(f"Saving histogram data to: {hist_data_file}")

    with open(hist_data_file, 'w') as f:
        f.write(f"Evaluation Score Histogram Data\n")
        f.write(f"{'='*60}\n")
        f.write(f"Source file: {h5_file_path}\n")
        f.write(f"Total scores: {len(scores):,}\n")
        f.write(f"Bins: {bins}\n")
        f.write(f"Range: {range_tuple}\n")
        f.write(f"\nStatistics:\n")
        f.write(f"  Min:    {min_score:,}\n")
        f.write(f"  Max:    {max_score:,}\n")
        f.write(f"  Mean:   {mean_score:,.2f}\n")
        f.write(f"  Median: {median_score:,.2f}\n")
        f.write(f"  StdDev: {std_score:,.2f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Bin Data (Full Range):\n")
        f.write(f"{'Bin Start':>15} {'Bin End':>15} {'Count':>15} {'Percentage':>12}\n")
        f.write(f"{'-'*60}\n")

        for i in range(len(counts)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            count = int(counts[i])
            percentage = (count / len(scores)) * 100
            f.write(f"{bin_start:>15,.0f} {bin_end:>15,.0f} {count:>15,} {percentage:>11.2f}%\n")

    print(f"✓ Histogram data saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze chess evaluation scores and game outcomes in H5 data file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.h5
  %(prog)s data.h5 -t 5000 -b 50000
  %(prog)s data.h5 --histogram --bins 200
  %(prog)s data.h5 --histogram --range -10000 10000

Note: Column 12 = side-to-move (1=white, 0=black)
      Column 13 = evaluation scores
      Column 14 = outcomes from STM perspective (2=win, 1=draw, 0=loss)
        """
    )
    parser.add_argument(
        'input_file',
        help='Path to the H5 file to analyze'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=10000,
        help='Score threshold for high score counting (default: 10000)'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=100000,
        help='Batch size for processing (default: 100000)'
    )
    parser.add_argument(
        '--histogram',
        action='store_true',
        help='Generate histogram of evaluation scores (requires matplotlib)'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=100,
        help='Number of bins for histogram (default: 100)'
    )
    parser.add_argument(
        '--range',
        nargs=2,
        type=int,
        metavar=('MIN', 'MAX'),
        help='Score range for histogram (default: auto)'
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

    if args.bins <= 0:
        print(f"Error: Number of bins must be positive, got {args.bins}")
        sys.exit(1)

    if args.range and args.range[0] >= args.range[1]:
        print(f"Error: Range minimum must be less than maximum")
        sys.exit(1)

    print(f"Analyzing file: {args.input_file}")
    print(f"Score threshold: {args.threshold:,}")
    print(f"Batch size: {args.batch_size:,}")
    if args.histogram:
        print(f"Histogram: enabled ({args.bins} bins)")
        if args.range:
            print(f"Histogram range: {args.range[0]:,} to {args.range[1]:,}")
    print()

    hist_range = tuple(args.range) if args.range else None
    analyze_chess_data(args.input_file, args.threshold, args.batch_size,
                      args.histogram, args.bins, hist_range)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
