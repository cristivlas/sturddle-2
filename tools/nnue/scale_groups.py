#!/usr/bin/env python3
import argparse
import bisect
import math
import os
import re


def bin_key(path):
    name = os.path.basename(path)
    stem = name[:-3] if name.endswith(".h5") else name
    if re.fullmatch(r"[0-9a-fA-F]{12,}", stem):
        return "guid"
    prefix = re.split(r"[-_.\d]", name)[0] or "misc"
    return f"{os.path.dirname(path)}/{prefix}"


def parse_reports(report_paths):
    """summary lines: '  3.28  0.984  DIVRG  /path.h5' -> {path: (ratio, corr)}"""
    line_rgx = re.compile(r"^\s*([0-9.]+|nan)\s+(-?[0-9.]+|nan)\s+(DIVRG\s+)?(\S.*?)\s*$")
    checked = {}
    for rp in report_paths:
        in_summary = False
        for line in open(rp):
            if re.match(r"^\s*ratio\s+corr\s+flag\s+file", line):
                in_summary = True
                continue
            if not in_summary:
                continue
            m = line_rgx.match(line)
            if not m:
                continue
            ratio, corr, path = float(m.group(1)), float(m.group(2)), m.group(4)
            checked.setdefault(path, []).append((ratio, corr))
    return checked


def auto_boundaries(ratios, gap):
    """Cut between consecutive sorted ratios where the jump is >= gap (geometric midpoint)."""
    rs = sorted(ratios)
    return [math.sqrt(a * b) for a, b in zip(rs, rs[1:]) if b / a >= gap]


def main(args):
    candidates = [line.strip() for line in open(args.input) if line.strip()]
    checked = parse_reports(args.reports)

    bins = {}
    for path, samples in checked.items():
        ratios = [r for r, _ in samples]
        corrs = [c for _, c in samples]
        bins.setdefault(bin_key(path), []).append((sum(ratios) / len(ratios), min(corrs)))

    verdict = {}  # bin key -> (ratio, corr)
    for key, samples in bins.items():
        verdict[key] = (sum(v[0] for v in samples) / len(samples), min(v[1] for v in samples))

    good = [ratio for ratio, corr in verdict.values() if corr >= args.min_corr and not math.isnan(ratio)]
    boundaries = sorted(args.split) if args.split else auto_boundaries(good, args.gap)
    if boundaries:
        print("boundaries: " + " ".join(f"{b:.2f}" for b in boundaries))

    groups = {}
    out = {"excluded": [], "unknown": []}
    for path in candidates:
        v = verdict.get(bin_key(path))
        if v is None:
            out["unknown"].append(path)
            continue
        ratio, corr = v
        if math.isnan(ratio) or math.isnan(corr) or corr < args.min_corr:
            out["excluded"].append(path)
        else:
            groups.setdefault(bisect.bisect(boundaries, ratio), []).append((ratio, path))

    for i, g in enumerate(sorted(groups)):
        members = groups[g]
        lo, hi = min(r for r, _ in members), max(r for r, _ in members)
        dest = f"{args.out_prefix}_{i + 1}.txt"
        with open(dest, "w") as f:
            f.write("\n".join(p for _, p in members) + "\n")
        print(f"{len(members):>5}  {dest}  (ratio {lo:.2f}-{hi:.2f})")

    for name, paths in out.items():
        dest = f"{args.out_prefix}_{name}.txt"
        with open(dest, "w") as f:
            f.write("\n".join(paths) + ("\n" if paths else ""))
        print(f"{len(paths):>5}  {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help="text file listing all candidate h5 paths")
    parser.add_argument("-r", "--reports", nargs="+", required=True, help="label_check report file(s)")
    parser.add_argument("--split", type=float, nargs="+", help="manual scale boundaries (disables auto grouping)")
    parser.add_argument(
        "--gap", type=float, default=1.4, help="auto mode: cut groups where consecutive bin ratios jump by this factor"
    )
    parser.add_argument("--min-corr", type=float, default=0.7, help="exclude bins with correlation below this")
    parser.add_argument("--out-prefix", default="scale", help="prefix for output list files")
    args = parser.parse_args()

    main(args)
