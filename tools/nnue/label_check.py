#!/usr/bin/env python3
"""
Assess label quality of h5 training data against a reference UCI engine.

Takes h5 files and/or text list files of h5 paths. For each file, stratified-
samples positions per pawn band (0-4, 5-8, 9-12, 13-16), has the engine
evaluate each, and compares stored labels to engine scores in WDL space --
broken down by the 16 pawn x king-file buckets. Reports per-file scale ratio
(mean |label| / mean |engine|) and flags files whose scale diverges across
buckets (max/min bucket ratio > --divergence).

Castling rights and en-passant are not stored in the data; both are assumed
absent.

Usage:
    ./label_check.py files.txt -e ./stockfish --per-band 250 --depth 12
"""

import argparse
import math
import os
import random
import re

import chess
import chess.engine
import h5py
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

FEATURE_COUNT = 13
BAND_LABELS = ["0-4", "5-8", "9-12", "13-16"]
KING_LABELS = ["QQ", "QK", "KQ", "KK"]  # white side / black side of the board
MIN_BUCKET_N = 25  # buckets with fewer samples are excluded from the divergence check


def decode_position(row):
    """12 bitboards (black/white per piece) + turn -> python-chess Board."""
    bb = [int(x) for x in row[:12]]
    board = chess.Board(fen=None)
    for b in bb:
        board.occupied |= b
    for b in bb[::2]:
        board.occupied_co[chess.BLACK] |= b
    for b in bb[1::2]:
        board.occupied_co[chess.WHITE] |= b
    board.kings = bb[0] | bb[1]
    board.pawns = bb[2] | bb[3]
    board.knights = bb[4] | bb[5]
    board.bishops = bb[6] | bb[7]
    board.rooks = bb[8] | bb[9]
    board.queens = bb[10] | bb[11]
    board.turn = bool(row[12])
    return board


def bucket_of(board):
    p = chess.popcount(board.pawns)
    band = 0 if p <= 4 else min((p - 1) // 4, 3)
    wk_right = chess.square_file(board.king(chess.WHITE)) >= 4
    bk_right = chess.square_file(board.king(chess.BLACK)) >= 4
    return band * 4 + wk_right * 2 + bk_right


def wdl(cp, scale):
    return 1.0 / (1.0 + math.exp(-cp / scale))


def analyse_file(path, engine, limit, args):
    """Returns list of (bucket, label_wdl, engine_wdl, label_cp, engine_cp)."""
    rng = np.random.default_rng()
    band_counts = [0] * 4
    records = []
    scanned = 0

    progress = tqdm(total=args.per_band * 4, unit="pos", desc=path.rsplit("/", 1)[-1]) if tqdm else None

    with h5py.File(path, "r") as hf:
        data = hf["data"]
        rows = len(data)

        while min(band_counts) < args.per_band and scanned < args.per_band * 400:
            row = data[rng.integers(0, rows)]
            scanned += 1
            if progress and scanned % 100 == 0:
                progress.set_postfix_str(f"{scanned} scanned")

            board = decode_position(row)
            if not board.is_valid():
                continue
            bucket = bucket_of(board)
            if band_counts[bucket // 4] >= args.per_band:
                continue

            label_cp = int(np.int64(row[FEATURE_COUNT]))  # STM POV
            if not board.turn:
                label_cp = -label_cp  # white POV

            try:
                info = engine.analyse(board, limit)
            except chess.engine.EngineError:
                continue
            engine_cp = info["score"].white().score(mate_score=args.mate_score)

            band_counts[bucket // 4] += 1
            records.append((bucket, wdl(label_cp, args.scale), wdl(engine_cp, args.scale), label_cp, engine_cp))
            if progress:
                progress.update(1)

    if progress:
        progress.close()
    return records


def stats(rec):
    """rec: (n, 5) slice -> (mae, bias, corr, mean|label|, mean|engine|, ratio)"""
    lw, ew, lcp, ecp = rec[:, 1], rec[:, 2], rec[:, 3], rec[:, 4]
    mae = np.mean(np.abs(lw - ew))
    bias = np.mean(lw - ew)
    corr = np.corrcoef(lw, ew)[0, 1] if len(rec) > 2 else float("nan")
    ml, me = np.mean(np.abs(lcp)), np.mean(np.abs(ecp))
    ratio = ml / me if me else float("nan")
    return mae, bias, corr, ml, me, ratio


def report(path, records, args):
    """Per-bucket table; returns (overall_ratio, overall_corr, divergent)."""
    rec = np.array(records)
    print(f"\n{path}")
    print(
        f"{'bucket':>6} {'pawns':>7} {'kings':>5} {'n':>5} {'wdl_mae':>8} {'bias':>8} {'corr':>6} {'|label|':>8} {'|engine|':>9} {'ratio':>6}"
    )

    ratios = []
    for b in range(16):
        sub = rec[rec[:, 0] == b]
        band, king = divmod(b, 4)
        pawns = BAND_LABELS[band]
        if len(sub) == 0:
            print(f"{b:>6} {pawns:>7} {KING_LABELS[king]:>5} {0:>5}")
            continue
        mae, bias, corr, ml, me, ratio = stats(sub)
        if len(sub) >= MIN_BUCKET_N and not math.isnan(ratio):
            ratios.append(ratio)
        print(
            f"{b:>6} {pawns:>7} {KING_LABELS[king]:>5} {len(sub):>5} {mae:>8.5f} {bias:>+8.5f} {corr:>6.3f}"
            f" {ml:>8.0f} {me:>9.0f} {ratio:>6.2f}"
        )

    mae, bias, corr, ml, me, ratio = stats(rec)
    divergent = bool(ratios) and max(ratios) / min(ratios) > args.divergence
    flag = "  ** SCALE-DIVERGENT ACROSS BUCKETS **" if divergent else ""
    print(
        f"{'overall':>26} {len(rec):>5} {mae:>8.5f} {bias:>+8.5f} {corr:>6.3f} {ml:>8.0f} {me:>9.0f} {ratio:>6.2f}{flag}"
    )
    return ratio, corr, divergent


def main(args):
    paths = []
    for arg in args.input:
        if arg.endswith(".h5"):
            paths.append(arg)
        else:
            paths.extend(line.strip() for line in open(arg) if line.strip())

    if args.per_group:
        # pre-binning: one bin per source family, folder-independent; guids all land in one bin
        groups = {}
        for p in paths:
            name = os.path.basename(p)
            stem = name[:-3] if name.endswith(".h5") else name
            if re.fullmatch(r"[0-9a-fA-F]{12,}", stem):
                key = "guid"
            else:
                prefix = re.split(r"[-_.\d]", name)[0] or "misc"
                key = f"{os.path.dirname(p)}/{prefix}"
            groups.setdefault(key, []).append(p)

        print(f"{len(groups)} bins:")
        for key, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"{len(g):>5}  {key}")

        paths = [p for g in groups.values() for p in random.sample(g, min(args.per_group, len(g)))]
        print(f"checking {len(paths)} representative file(s)")

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    limit = chess.engine.Limit(depth=args.depth) if args.depth else chess.engine.Limit(nodes=args.nodes)

    summary = []
    try:
        for name, value in [("Threads", args.threads), ("Hash", args.hash)]:
            try:
                engine.configure({name: value})
            except chess.engine.EngineError:
                pass

        for path in paths:
            records = analyse_file(path, engine, limit, args)
            if not records:
                print(f"\n{path}: no valid positions sampled")
                continue
            ratio, corr, divergent = report(path, records, args)
            summary.append((path, ratio, corr, divergent))
    finally:
        engine.quit()

    if len(summary) > 1:
        print(f"\n{'ratio':>6} {'corr':>6} {'flag':>6}  file")
        for path, ratio, corr, divergent in sorted(summary, key=lambda s: (math.isnan(s[1]), -s[1])):
            print(f"{ratio:>6.2f} {corr:>6.3f} {'DIVRG' if divergent else '':>6}  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", nargs="+", help="h5 file(s) and/or text file(s) listing h5 paths")
    parser.add_argument("-e", "--engine", required=True, help="path to UCI engine used as reference")
    parser.add_argument("--per-band", type=int, default=250, help="positions to evaluate per pawn band per file")
    parser.add_argument("--depth", type=int, default=12, help="engine search depth (0 to use --nodes)")
    parser.add_argument("--nodes", type=int, default=100000, help="node limit when --depth 0")
    parser.add_argument("--hash", type=int, default=256, help="engine hash MB")
    parser.add_argument("--threads", type=int, default=1, help="engine threads")
    parser.add_argument("--scale", type=float, default=390.0, help="cp -> WDL sigmoid scale")
    parser.add_argument("--mate-score", type=int, default=3000, help="cp value for mate scores")
    parser.add_argument(
        "--divergence", type=float, default=2.0, help="flag file if max/min bucket scale ratio exceeds this"
    )
    parser.add_argument(
        "--per-group", type=int, default=0, help="check only N random files per source group (folder + name prefix)"
    )
    args = parser.parse_args()

    main(args)
