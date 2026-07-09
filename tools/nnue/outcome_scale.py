#!/usr/bin/env python3
"""
Fit the optimal --outcome-scale from training data.

The blend loss calibrates evals to game outcomes via P(win) = sigmoid(eval / S).
This tool fits S directly: single-parameter logistic regression (Newton on
beta = 1/S) minimizing cross-entropy between sigmoid(beta * cp) and the
recorded outcome, overall and per bucket. Evals and outcomes are both STM-POV,
matching the loss.

Evals are first divided by the same per-bucket label-scale ratios the trainers
apply: per-member sidecar profiles (<member>.h5.profile.json) are auto-resolved
for virtual datasets, so the fitted S is directly the value to train with.

Usage:
    ./outcome_scale.py mix.h5 --sample 0.02
"""

import argparse
import json
import os

import h5py
import numpy as np

FEATURE_COUNT = 13
FILES_EFGH = np.uint64(0xF0F0F0F0F0F0F0F0)

KING_LABELS = ["QQ", "QK", "KQ", "KK"]  # white side / black side of the board


def popcount(bb):
    bb = bb.astype(np.uint64)
    bb = bb - ((bb >> np.uint64(1)) & np.uint64(0x5555555555555555))
    bb = (bb & np.uint64(0x3333333333333333)) + ((bb >> np.uint64(2)) & np.uint64(0x3333333333333333))
    bb = (bb + (bb >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((bb * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def bucket_ids(x):
    """x: (batch, 13) uint64. Columns: 0=black king, 1=white king, 2/3=pawns."""
    pawns = popcount(x[:, 2]) + popcount(x[:, 3])
    pawn_id = np.where(pawns <= 4, 0, np.minimum((pawns - 1) // 4, 3))

    wk_right = (x[:, 1] & FILES_EFGH) != 0
    bk_right = (x[:, 0] & FILES_EFGH) != 0
    king_id = wk_right.astype(np.int64) * 2 + bk_right.astype(np.int64)

    return pawn_id * 4 + king_id


def load_profile(path):
    with open(path) as f:
        profile = json.load(f)
    ratios = profile["ratios"] if isinstance(profile, dict) else profile
    ratios = np.asarray(ratios, dtype=np.float32)
    assert ratios.shape == (16,) and np.all(ratios > 0.0), f"{path}: expected 16 positive ratios"
    return ratios


def member_profile_table(data, filepath):
    """(member start rows, (members x 16) ratio matrix, member paths) — same resolution as
    the trainers: per-member sidecar, else the container's sidecar, else all-ones. One level."""
    if data.is_virtual:
        # HDF5 resolves relative member paths against the container's directory
        base = os.path.dirname(os.path.abspath(filepath))
        members = sorted((vs.vspace.get_select_bounds()[0][0], vs.file_name) for vs in data.virtual_sources())
        members = [
            (start, filepath if path == "." else (path if os.path.isabs(path) else os.path.join(base, path)))
            for start, path in members
        ]
    else:
        members = [(0, filepath)]
    assert members[0][0] == 0, "first member must start at row 0"

    default = np.ones(16, dtype=np.float32)
    own_sidecar = filepath + ".profile.json"
    if os.path.exists(own_sidecar):
        default = load_profile(own_sidecar)

    cache = {}
    matched = 0
    profiles = []
    for _, path in members:
        sidecar = path + ".profile.json"
        if sidecar not in cache:
            if path != filepath and os.path.exists(sidecar):
                cache[sidecar] = load_profile(sidecar)
                matched += 1
            else:
                cache[sidecar] = default
        profiles.append(cache[sidecar])
    print(f"{filepath}: {len(members)} member(s), {matched} with own profile")
    return np.array([m[0] for m in members], dtype=np.int64), np.stack(profiles), [m[1] for m in members]


def fit_scale(cp, y, beta=1.0 / 400.0):
    """Newton's method on beta = 1/S; the 1-D logistic BCE is convex in beta."""
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(beta * cp, -30.0, 30.0)))
        grad = np.dot(p - y, cp)
        hess = np.dot(p * (1.0 - p), cp * cp)
        if hess <= 0.0:
            return float("nan")
        step = grad / hess
        beta -= step
        if abs(step) < 1e-12 * max(abs(beta), 1e-12):
            break
    return 1.0 / beta if beta > 0.0 else float("nan")


def main(args):
    cp_parts, y_parts, bucket_parts, member_parts = [], [], [], []
    member_paths = []
    for path in args.input:
        with h5py.File(path, "r") as hf:
            data = hf["data"]
            if args.raw:
                member_starts = np.zeros(1, dtype=np.int64)
                member_profiles = np.ones((1, 16), dtype=np.float32)
                member_paths = [path]
            else:
                member_starts, member_profiles, member_paths = member_profile_table(data, path)
            num_batches = max(1, len(data) // args.batch_size)
            indices = np.arange(num_batches)
            if args.sample:
                k = max(1, int(num_batches * args.sample))
                indices = np.random.choice(num_batches, k, replace=False)
                indices.sort()  # sequential reads are faster in h5
            print(f"{path}: {len(data):,} rows, using {len(indices)} of {num_batches} batches of {args.batch_size}")
            for n, i in enumerate(indices):
                start = i * args.batch_size
                block = data[start : start + args.batch_size, : FEATURE_COUNT + 2]
                cp = block[:, FEATURE_COUNT].astype(np.int64)  # STM POV
                y = block[:, FEATURE_COUNT + 1].astype(np.float32) / 2.0  # STM POV: 0=loss, 0.5=draw, 1=win
                mask = np.abs(cp) <= args.filter if args.filter else np.ones(len(cp), dtype=bool)
                buckets = bucket_ids(block[:, :FEATURE_COUNT])[mask]
                member_ids = (
                    np.searchsorted(member_starts, np.arange(start, start + len(block))[mask], side="right") - 1
                )
                cp_parts.append(cp[mask].astype(np.float32) / member_profiles[member_ids, buckets])
                y_parts.append(y[mask])
                bucket_parts.append(buckets)
                member_parts.append(member_ids)
                if (n + 1) % 50 == 0 or n + 1 == len(indices):
                    print(f"\r{n + 1}/{len(indices)} batches", end="", flush=True)
            print()

    cp = np.concatenate(cp_parts)
    y = np.concatenate(y_parts)
    buckets = np.concatenate(bucket_parts)
    member_ids = np.concatenate(member_parts)
    if args.max_rows and len(cp) > args.max_rows:
        keep = np.random.choice(len(cp), args.max_rows, replace=False)
        cp, y, buckets, member_ids = cp[keep], y[keep], buckets[keep], member_ids[keep]

    print(f"{'bucket':>6} {'pawns':>7} {'kings':>5} {'n':>12} {'draw%':>6} {'|cp|':>6} {'S':>7}")
    for b in range(16):
        sel = buckets == b
        n = int(sel.sum())
        pawn_id, king_id = divmod(b, 4)
        pawns = "0-4" if pawn_id == 0 else f"{pawn_id * 4 + 1}-{pawn_id * 4 + 4}"
        row = f"{b:>6} {pawns:>7} {KING_LABELS[king_id]:>5} {n:>12,}"
        if n:
            sb = fit_scale(cp[sel], y[sel])
            row += f" {100.0 * np.mean(y[sel] == 0.5):>6.2f} {np.mean(np.abs(cp[sel])):>6.0f} {sb:>7.1f}"
        print(row)

    s = fit_scale(cp, y)
    print(f"{'overall':>33} {len(cp):>12,} {100.0 * np.mean(y == 0.5):>6.2f} {np.mean(np.abs(cp)):>6.0f} {s:>7.1f}")

    if args.update_profile:
        # Fit each member on its own rows and write its sidecar (thin buckets -> member overall)
        for m, mpath in enumerate(member_paths):
            msel = member_ids == m
            if not msel.any():
                print(f"{mpath}: no sampled rows, skipped")
                continue
            ms = fit_scale(cp[msel], y[msel])
            if not np.isfinite(ms):
                ms = s  # degenerate member (labels uncorrelated with outcomes): global fit
            if not np.isfinite(ms):
                print(f"{mpath}: fit diverged, skipped")
                continue
            scales = []
            for b in range(16):
                sel = msel & (buckets == b)
                sb = fit_scale(cp[sel], y[sel]) if sel.sum() >= args.min_n else float("nan")
                scales.append(round(float(sb if np.isfinite(sb) else ms), 1))
            sidecar = mpath + ".profile.json"
            profile = {}
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    profile = json.load(f)
            profile.setdefault("ratios", [1.0] * 16)  # trainers require ratios; scale-1 for a fresh sidecar
            profile["outcome_scale"] = scales
            with open(sidecar, "w") as f:
                json.dump(profile, f, indent=2)
            print(f"{sidecar}: overall S {ms:.1f}")

    # Empirical calibration: actual mean score per eval bin vs the fitted sigmoid.
    print(f"\n{'eval bin':>16} {'n':>12} {'score%':>7} {'fit%':>7}")
    edges = [0, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1200, 10**9]
    signed_y = np.where(cp < 0, 1.0 - y, y)  # fold to positive evals
    acp = np.abs(cp)
    for lo, hi in zip(edges, edges[1:]):
        sel = (acp >= lo) & (acp < hi)
        n = int(sel.sum())
        if not n:
            continue
        mid = np.mean(acp[sel])
        fit = 100.0 / (1.0 + np.exp(-mid / s))
        label = f"{lo}-{hi if hi < 10**9 else ''}"
        print(f"{label:>16} {n:>12,} {100.0 * np.mean(signed_y[sel]):>7.2f} {fit:>7.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", nargs="+", help="h5 data file(s); the fit pools all of them")
    parser.add_argument("-b", "--batch-size", type=int, default=16384)
    parser.add_argument("--sample", type=float, help="sampling ratio, same as the trainers")
    parser.add_argument("--filter", type=int, help="drop rows with |raw eval| above this, like the trainers' -F")
    parser.add_argument("--max-rows", type=int, default=20_000_000, help="cap on pooled rows (random thinning)")
    parser.add_argument("--raw", action="store_true", help="ignore sidecar profiles, fit on raw labels")
    parser.add_argument(
        "--update-profile",
        action="store_true",
        help="fit each VDS member separately and write outcome_scale into its sidecar",
    )
    parser.add_argument("--min-n", type=int, default=50_000, help="buckets with fewer rows fall back to the overall S")
    args = parser.parse_args()

    if args.update_profile and (args.raw or len(args.input) != 1):
        parser.error("--update-profile requires a single input file and no --raw (S must be in corrected domain)")

    if args.sample:
        args.sample = max(1e-3, min(1.0, args.sample))

    main(args)
