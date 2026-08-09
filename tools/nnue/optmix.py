#!/usr/bin/env python3
"""
Assemble an "optimum" training mix from a collection of h5 files.

Given a text file listing candidate h5 files (one path per line), profile each
one (sampled; cached in a JSON sidecar), then greedily select whole files --
optionally duplicated -- to maximize graded eval signal where it matters most
(low pawn counts), while capping extreme and zero-eval label shares, under a
total position budget.

Score of a candidate mix, computed per pawn band (0-4, 5-8, 9-12, 13-16), in
rows (volume-based, so adding good data always helps and the budget gets used):
    + BAND_WEIGHT[b] * graded-label rows
    - penalties for ext/zero rows above the per-band cap shares
    + endgame-share bonus, capped at EG_TARGET

Usage:
    ./optmix.py candidates.txt --max-positions 8e9 --out mix.txt
"""

import argparse
import json
import os

import h5py
import numpy as np

FEATURE_COUNT = 13
FILES_EFGH = np.uint64(0xF0F0F0F0F0F0F0F0)

BAND_WEIGHT = [3.0, 2.0, 1.0, 0.5]  # reward graded labels most at low pawn counts
EXT_CAP = [0.10, 0.10, 0.08, 0.03]  # tolerated extreme-eval share per band
ZERO_CAP = [0.35, 0.05, 0.03, 0.03]  # tolerated zero-eval share per band
EXT_PENALTY = 10.0
ZERO_PENALTY = 5.0
EG_TARGET = 0.10  # reward 0-4 pawn share up to this fraction of the mix
EG_BONUS = 2.0


def popcount(bb):
    bb = bb.astype(np.uint64)
    bb = bb - ((bb >> np.uint64(1)) & np.uint64(0x5555555555555555))
    bb = (bb & np.uint64(0x3333333333333333)) + ((bb >> np.uint64(2)) & np.uint64(0x3333333333333333))
    bb = (bb + (bb >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((bb * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def pawn_bands(x):
    """x: (batch, 13) uint64. Band 0..3 by pawn count, same edges as the trainer buckets."""
    pawns = popcount(x[:, 2]) + popcount(x[:, 3])
    return np.where(pawns <= 4, 0, np.minimum((pawns - 1) // 4, 3))


def profile_file(path, sample, batch_size, cache):
    key = f"{path}:{sample}:{batch_size}"
    with h5py.File(path, "r") as hf:
        data = hf["data"]
        rows = len(data)
        if key in cache and cache[key]["rows"] == rows:
            return cache[key]

        num_batches = max(1, rows // batch_size)
        indices = np.arange(num_batches)
        if sample and num_batches > 1:
            k = max(1, int(num_batches * sample))
            indices = np.random.choice(num_batches, k, replace=False)
            indices.sort()

        counts = np.zeros(4, dtype=np.int64)
        zeros = np.zeros(4, dtype=np.int64)
        mids = np.zeros(4, dtype=np.int64)
        exts = np.zeros(4, dtype=np.int64)

        for i in indices:
            start = i * batch_size
            block = data[start : start + batch_size, : FEATURE_COUNT + 1]
            bands = pawn_bands(block[:, :FEATURE_COUNT])
            cp = block[:, FEATURE_COUNT].astype(np.int64)
            abs_c = np.abs(cp)

            counts += np.bincount(bands, minlength=4)
            zeros += np.bincount(bands, weights=(cp == 0), minlength=4).astype(np.int64)
            mids += np.bincount(bands, weights=(abs_c > 50) & (abs_c <= 500), minlength=4).astype(np.int64)
            exts += np.bincount(bands, weights=(abs_c > 1260), minlength=4).astype(np.int64)

    total = counts.sum()
    prof = {
        "rows": rows,
        "band_share": (counts / total).tolist(),
        "zero": (zeros / np.maximum(counts, 1)).tolist(),
        "mid": (mids / np.maximum(counts, 1)).tolist(),
        "ext": (exts / np.maximum(counts, 1)).tolist(),
    }
    cache[key] = prof
    return prof


def contribution(profile):
    """Per-file absolute row counts as a (4 stats x 4 bands) matrix: band, zero, mid, ext rows."""
    share = np.array(profile["band_share"]) * profile["rows"]
    return np.stack([share, share * profile["zero"], share * profile["mid"], share * profile["ext"]])


def score(stats, max_positions):
    band_rows, zero_rows, mid_rows, ext_rows = stats
    total = band_rows.sum()
    if total == 0:
        return 0.0
    s = 0.0
    for b in range(4):
        s += BAND_WEIGHT[b] * mid_rows[b]
        s -= EXT_PENALTY * max(0.0, ext_rows[b] - EXT_CAP[b] * band_rows[b])
        s -= ZERO_PENALTY * max(0.0, zero_rows[b] - ZERO_CAP[b] * band_rows[b])
    s /= max_positions
    s += EG_BONUS * min(band_rows[0] / total, EG_TARGET) / EG_TARGET
    return s


def main(args):
    paths = [line.strip() for line in open(args.input) if line.strip()]

    cache = {}
    if args.cache and os.path.exists(args.cache):
        cache = json.load(open(args.cache))

    profiles = {}
    for path in paths:
        print(f"profiling {path} ...", flush=True)
        profiles[path] = profile_file(path, args.sample, args.batch_size, cache)
    if args.cache:
        json.dump(cache, open(args.cache, "w"))

    print(f"\n{'rows':>16} {'eg%':>6} {'mid0%':>6} {'ext0%':>6} {'zero0%':>6}  file  (band-0 = 0-4 pawns)")
    for path, p in sorted(profiles.items(), key=lambda kv: -kv[1]["mid"][0]):
        print(
            f"{p['rows']:>16,} {100 * p['band_share'][0]:>6.2f} {100 * p['mid'][0]:>6.2f}"
            f" {100 * p['ext'][0]:>6.2f} {100 * p['zero'][0]:>6.2f}  {path}"
        )

    contribs = {path: contribution(prof) for path, prof in profiles.items()}
    selection = {}
    stats = np.zeros((4, 4))
    used = 0
    while True:
        base = score(stats, args.max_positions)
        best, best_metric = None, 0.0
        for path, prof in profiles.items():
            if selection.get(path, 0) >= args.max_dup:
                continue
            if used + prof["rows"] > args.max_positions:
                continue
            gain = score(stats + contribs[path], args.max_positions) - base
            metric = gain / prof["rows"] if args.density else gain
            if gain > 0 and metric > best_metric:
                best, best_metric = path, metric
        if best is None:
            break
        selection[best] = selection.get(best, 0) + 1
        stats += contribs[best]
        used += profiles[best]["rows"]

    band_rows, zero_rows, mid_rows, ext_rows = stats
    total = band_rows.sum()
    print(
        f"\nselected {sum(selection.values())} file(s), {total:,.0f} positions,"
        f" score {score(stats, args.max_positions):.4f}"
    )
    print(f"{'band':>6} {'share%':>7} {'zero%':>6} {'mid%':>6} {'ext%':>6}")
    for b, label in enumerate(["0-4", "5-8", "9-12", "13-16"]):
        print(
            f"{label:>6} {100 * band_rows[b] / total:>7.2f} {100 * zero_rows[b] / band_rows[b]:>6.2f}"
            f" {100 * mid_rows[b] / band_rows[b]:>6.2f} {100 * ext_rows[b] / band_rows[b]:>6.2f}"
        )

    lines = [path for path, dups in selection.items() for _ in range(dups)]
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    else:
        print()
        print("\n".join(f"{'':2}{line} x{selection[line]}" for line in selection))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help="text file listing candidate h5 files")
    parser.add_argument("--max-positions", type=float, default=8e9, help="position budget for the mix")
    parser.add_argument("--max-dup", type=int, default=1, help="max times a file may be repeated")
    parser.add_argument("--density", action="store_true", help="greedy by gain per position (favors small rich files)")
    parser.add_argument("--sample", type=float, default=0.01, help="profiling sample ratio")
    parser.add_argument("-b", "--batch-size", type=int, default=16384)
    parser.add_argument("--cache", default="optmix_cache.json", help="profile cache (JSON); '' disables")
    parser.add_argument("--out", help="write selected mix list to this file")
    args = parser.parse_args()

    main(args)
