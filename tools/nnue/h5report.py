#!/usr/bin/env python3
import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

OUTCOMES = ("Losses", "Draws", "Wins")


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def analyze(path, threshold, batch_size, histogram, bins, hist_range, mismatch):
    with h5py.File(path, "r") as f:
        if "data" not in f:
            sys.exit(f"{path}: no 'data' dataset")
        data = f["data"]
        if data.ndim != 2 or data.shape[1] < 15:
            sys.exit(f"{path}: expected at least 15 columns, got shape {data.shape}")
        total = data.shape[0]
        print(f"{path}: {total:,} records")

        outcomes = np.zeros(3, dtype=np.int64)
        wdl = np.zeros((2, 3), dtype=np.int64)  # [side to move][outcome]
        bad_stm = bad_outcome = 0
        above = zero = positive = negative = 0
        mismatched = mismatched_kept = 0
        scores = [] if histogram else None

        with tqdm(total=total, unit=" rec") as pbar:
            for start in range(0, total, batch_size):
                block = data[start : start + batch_size, 12:15].astype(np.int64)
                stm, score, outcome = block[:, 0], block[:, 1], block[:, 2]
                if histogram:
                    scores.append(score.copy())  # a view would keep the whole block alive

                above += np.count_nonzero(np.abs(score) > threshold)
                zero += np.count_nonzero(score == 0)
                positive += np.count_nonzero(score > 0)
                negative += np.count_nonzero(score < 0)

                stm_ok = (stm == 0) | (stm == 1)
                outcome_ok = (outcome >= 0) & (outcome <= 2)
                bad_stm += np.count_nonzero(~stm_ok)
                bad_outcome += np.count_nonzero(~outcome_ok)
                outcomes += np.bincount(outcome[outcome_ok], minlength=3)
                ok = stm_ok & outcome_ok
                wdl += np.bincount(stm[ok] * 3 + outcome[ok], minlength=6).reshape(2, 3)

                if mismatch:
                    # sign-invariant, so the side-to-move eval works as is
                    m = ((score > mismatch) & (outcome == 0)) | ((score < -mismatch) & (outcome == 2))
                    mismatched += np.count_nonzero(m)
                    mismatched_kept += np.count_nonzero(m & (np.abs(score) <= threshold))

                pbar.update(len(block))
                pbar.set_postfix(high=f"{above:,}", rate=f"{pct(above, start + len(block)):.1f}%")

    section("SIDE TO MOVE")
    print(f"White: {wdl[1].sum():,} ({pct(wdl[1].sum(), total):.2f}%)")
    print(f"Black: {wdl[0].sum():,} ({pct(wdl[0].sum(), total):.2f}%)")
    if bad_stm:
        print(f"WARNING: {bad_stm:,} records with side to move other than 0 or 1")

    section("EVALUATION")
    print(f"|score| > {threshold:,}: {above:,} ({pct(above, total):.2f}%)")
    print(f"Zero:     {zero:,} ({pct(zero, total):.2f}%)")
    print(f"Positive: {positive:,} ({pct(positive, total):.2f}%)")
    print(f"Negative: {negative:,} ({pct(negative, total):.2f}%)")

    section("OUTCOMES, FROM THE SIDE TO MOVE")
    for label, row in (("All", outcomes), ("White to move", wdl[1]), ("Black to move", wdl[0])):
        print(label)
        for i in (2, 1, 0):
            print(f"  {OUTCOMES[i] + ':':<8}{row[i]:,} ({pct(row[i], row.sum()):.2f}%)")
    if bad_outcome:
        print(f"WARNING: {bad_outcome:,} records with outcome other than 0, 1 or 2")

    if mismatch:
        survivors = total - above
        section("EVAL/OUTCOME MISMATCH (--discard-mismatch)")
        print(f"Predicate: |eval| > {mismatch:,} and outcome disagrees")
        print(f"All mismatches: {mismatched:,} ({pct(mismatched, total):.2f}% of total)")
        print(
            f"After |eval| <= {threshold:,} filter: {mismatched_kept:,}"
            f" ({pct(mismatched_kept, survivors):.2f}% of {survivors:,} survivors)"
        )

    if histogram:
        save_histogram(np.concatenate(scores), bins, hist_range, path)


def save_histogram(scores, bins, score_range, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("--histogram needs matplotlib")
        return

    low, high = scores.min(), scores.max()
    mean, median = scores.mean(), np.median(scores)
    summary = [
        f"  Min:    {low:,}",
        f"  Max:    {high:,}",
        f"  Mean:   {mean:,.2f}",
        f"  Median: {median:,.2f}",
        f"  StdDev: {scores.std():,.2f}",
    ]
    section("HISTOGRAM")
    print("\n".join(summary))

    full_range = score_range or (low, high)
    commas = FuncFormatter(lambda x, _: f"{int(x):,}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    counts, edges, _ = ax1.hist(scores, bins=bins, range=full_range, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.set_xlabel("Evaluation Score", fontsize=12)
    ax1.set_ylabel("Frequency (log scale)", fontsize=12)
    ax1.set_yscale("log")
    ax1.set_title(f"Evaluation Scores Distribution (n={len(scores):,})", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:,.0f}")
    ax1.axvline(median, color="green", linestyle="--", linewidth=2, label=f"Median: {median:,.0f}")
    ax1.legend()
    ax1.yaxis.set_major_formatter(commas)

    p99 = np.percentile(np.abs(scores), 99)
    ax2.hist(scores[np.abs(scores) <= p99], bins=bins, range=(-p99, p99), edgecolor="black", alpha=0.7, color="coral")
    ax2.set_xlabel("Evaluation Score", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(f"Zoomed Distribution (+/-99th percentile, +/-{p99:,.0f})", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax2.yaxis.set_major_formatter(commas)

    plt.tight_layout()
    base = os.path.splitext(os.path.basename(path))[0]
    plt.savefig(f"{base}_eval_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(f"{base}_eval_histogram.txt", "w") as out:
        out.write(f"Source file: {path}\nTotal scores: {len(scores):,}\nBins: {bins}\nRange: {full_range}\n\n")
        out.write("\n".join(summary) + "\n\n")
        out.write(f"{'Bin Start':>15} {'Bin End':>15} {'Count':>15} {'Percentage':>12}\n")
        for start, end, count in zip(edges[:-1], edges[1:], counts.astype(np.int64)):
            out.write(f"{start:>15,.0f} {end:>15,.0f} {count:>15,} {pct(count, len(scores)):>11.2f}%\n")
    print(f"Saved {base}_eval_histogram.png and {base}_eval_histogram.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Eval and outcome statistics for an H5 training file.",
        epilog="Columns: 12 = side to move (1 = white), 13 = eval, 14 = outcome for the side to move (2 = win).",
    )
    parser.add_argument("input_file")
    parser.add_argument("-t", "--threshold", type=int, default=10000, help="high |score| threshold (default: 10000)")
    parser.add_argument("-b", "--batch-size", type=int, default=100000, help="default: 100000")
    parser.add_argument("--histogram", action="store_true", help="save eval histogram .png and .txt (needs matplotlib)")
    parser.add_argument("--bins", type=int, default=100, help="histogram bins (default: 100)")
    parser.add_argument("--range", nargs=2, type=int, metavar=("MIN", "MAX"), help="histogram range (default: data)")
    parser.add_argument(
        "--mismatch", type=int, default=0, help="count eval/outcome disagreements above this cp, as --discard-mismatch"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        parser.error(f"{args.input_file} not found")
    if args.threshold < 0:
        parser.error("--threshold must be >= 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.bins <= 0:
        parser.error("--bins must be > 0")
    if args.range and args.range[0] >= args.range[1]:
        parser.error("--range MIN must be less than MAX")

    analyze(
        args.input_file,
        args.threshold,
        args.batch_size,
        args.histogram,
        args.bins,
        tuple(args.range) if args.range else None,
        args.mismatch,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
