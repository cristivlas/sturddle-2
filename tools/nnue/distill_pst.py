#!/usr/bin/env python3
"""
Distill piece-square tables from a trained NNUE.

Runs batched white-POV inference over H5 training positions, subtracts the
graded material component (WEIGHT + ADJUST[pawn_bucket], duplicated from
chess.h), and fits the residual onto signed piece-square occupancy features
by streaming least squares. Output has the exact tables.h shape: 6 x 64
SQUARE_TABLE plus ENDGAME_KING_SQUARE_TABLE, with the king routed to the
endgame table when total piece count <= ENDGAME_PIECE_COUNT.

Usage:
    python tools/nnue/distill_pst.py -m weights.bin data.h5 [more.h5 ...] [-o tables_out.h]
"""

import argparse
import os
import re
import sys

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_engine_constants():
    """Pull WEIGHT/ADJUST/bucket/endgame constants out of chess.h and common.h."""
    with open(os.path.join(REPO_ROOT, "common.h"), encoding="utf-8") as f:
        common = f.read()
    with open(os.path.join(REPO_ROOT, "chess.h"), encoding="utf-8") as f:
        chess = f.read()

    grading = re.search(r"#define EVAL_PIECE_GRADING\s+(\w+)", common).group(1) == "true"
    endgame = int(re.search(r"constexpr int ENDGAME_PIECE_COUNT\s*=\s*(\d+)", common).group(1))
    buckets = int(re.search(r"constexpr int PAWN_BUCKETS\s*=\s*(\d+)", chess).group(1))

    values = re.findall(r"#define PIECE_VALUES\s*\{([^}]*)\}", chess)
    weight = [int(v) for v in values[0 if grading else 1].split(",")]
    assert len(weight) == 7, weight

    if not grading:
        return weight, [[0] * 7 for _ in range(buckets)], buckets, endgame

    macro = chess[chess.index("#define GRADING_ADJUST") :]
    macro = macro[: re.search(r"\n[^\\\n]*\n", macro).start()]  # macro ends with the last '\'-continued line
    macro = re.sub(r"/\*.*?\*/", "", macro, flags=re.S)
    adjust = [[int(v) for v in row.split(",") if v.strip()] for row in re.findall(r"\{([^{}]*)\}", macro)]
    assert len(adjust) == buckets and all(len(row) == 7 for row in adjust), adjust

    return weight, adjust, buckets, endgame


WEIGHT, GRADING_ADJUST, PAWN_BUCKETS, ENDGAME_PIECE_COUNT = _parse_engine_constants()

# H5 column base per piece type (+0 black, +1 white); PST emit order and chess.h PieceType index
H5_BASE = {"PAWN": 2, "KNIGHT": 4, "BISHOP": 6, "ROOK": 8, "QUEEN": 10, "KING": 0}
PST_ORDER = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]
PIECE_TYPE = {"PAWN": 1, "KNIGHT": 2, "BISHOP": 3, "ROOK": 4, "QUEEN": 5, "KING": 6}

PARAMS = 7 * 64  # 6 MG tables + endgame king


def pawn_bucket(pawn_count):
    return np.where(pawn_count <= 4, 0, np.minimum((pawn_count - 1) // 4, PAWN_BUCKETS - 1))


def unpack(bb):
    """(B,) uint64 -> (B, 64) float64 with array index == chess square (A1=0)."""
    bits = (bb[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)
    return bits.astype(np.float64)


def mirror(bits):
    """Flip ranks: square index sq -> sq ^ 56 (white PST lookup)."""
    return bits.reshape(-1, 8, 8)[:, ::-1, :].reshape(-1, 64)


def features_and_material(rows):
    """Build (B, PARAMS) signed PST features and the graded material score per row."""
    bb = rows[:, :12].astype(np.uint64)
    n = rows.shape[0]

    black = {}
    white = {}
    for name, base in H5_BASE.items():
        black[name] = unpack(bb[:, base])
        white[name] = unpack(bb[:, base + 1])

    total = sum(b.sum(axis=1) + w.sum(axis=1) for b, w in zip(black.values(), white.values()))
    eg = (total <= ENDGAME_PIECE_COUNT).astype(np.float64)[:, None]

    buckets = pawn_bucket((black["PAWN"].sum(axis=1) + white["PAWN"].sum(axis=1)).astype(np.int64))
    adjust = np.asarray(GRADING_ADJUST, dtype=np.float64)[buckets]  # (B, 7)

    x = np.empty((n, PARAMS))
    material = np.zeros(n)
    for i, name in enumerate(PST_ORDER):
        block = mirror(white[name]) - black[name]
        if name == "KING":
            x[:, i * 64 : (i + 1) * 64] = block * (1.0 - eg)
            x[:, (i + 1) * 64 :] = block * eg
        else:
            x[:, i * 64 : (i + 1) * 64] = block
            value = WEIGHT[PIECE_TYPE[name]] + adjust[:, PIECE_TYPE[name]]
            material += value * (white[name].sum(axis=1) - black[name].sum(axis=1))
    return x, material


def load_model(bin_path, device):
    import torch
    import train_torch as tt

    model = tt.NNUE()
    tt.load_bin(model, bin_path)
    model.eval()
    return model.to(device)


TABLES_H_HEAD = """#pragma once

#include "common.h"
/*
 * Piece-square tables.
 * https://www.chessprogramming.org/Simplified_Evaluation_Function
 */
#if USE_PIECE_SQUARE_TABLES

static
#if !PS_PAWN_TUNING_ENABLED && !PS_KNIGHT_TUNING_ENABLED && !PS_BISHOP_TUNING_ENABLED && \\
    !PS_ROOK_TUNING_ENABLED && !PS_QUEEN_TUNING_ENABLED && !PS_KING_TUNING_ENABLED
    constexpr
#endif

int SQUARE_TABLE[][64] = {
    {}/* NONE */,"""

ENDGAME_HEAD = """

static
#if !PS_KING_TUNING_ENABLED
    constexpr
#endif
int ENDGAME_KING_SQUARE_TABLE[64] = {"""


def emit(tables, out):
    def block(values, indent):
        lines = []
        for r in range(8):
            row = values[r * 8 : (r + 1) * 8]
            lines.append(indent + " ".join(f"{v:4d}," for v in row))
        return "\n".join(lines)

    print(TABLES_H_HEAD, file=out)
    for i, name in enumerate(PST_ORDER):
        print(f"    {{ /* {name} */", file=out)
        print(block(tables[i * 64 : (i + 1) * 64], "        "), file=out)
        print("    },", file=out)
    print("};", file=out)
    print(ENDGAME_HEAD, file=out)
    print(block(tables[6 * 64 :], "    "), file=out)
    print("};", file=out)
    print("\n#endif /* USE_PIECE_SQUARE_TABLES */", file=out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", nargs="+", help="H5 training data file(s)")
    p.add_argument("-m", "--model", default="weights.bin", help="weights.bin")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    p.add_argument("-b", "--batch-size", type=int, default=8192)
    p.add_argument("--limit", type=int, help="max positions per file")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--device", help="default: cuda if available, else cpu")
    args = p.parse_args()

    import torch

    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}", file=sys.stderr)

    model = load_model(args.model, args.device)

    gram = np.zeros((PARAMS, PARAMS))
    rhs = np.zeros(PARAMS)
    count = 0
    sum_sq = 0.0

    with torch.no_grad():
        for path in args.data:
            with h5py.File(path, "r") as hf:
                data = hf["data"]
                n = min(data.shape[0], args.limit) if args.limit else data.shape[0]
                print(f"{path}: {n:,} rows", file=sys.stderr)
                for s in tqdm(range(0, n - n % args.batch_size, args.batch_size), unit="batch"):
                    rows = data[s : s + args.batch_size]
                    packed = rows[:, :13].astype(np.int64)
                    y_nn = model(torch.from_numpy(packed).to(args.device))[:, 0].cpu().numpy() * 100.0

                    x, material = features_and_material(rows)
                    y = y_nn.astype(np.float64) - material

                    gram += x.T @ x
                    rhs += x.T @ y
                    count += rows.shape[0]
                    sum_sq += float(y @ y)

    theta = np.linalg.solve(gram + args.ridge * np.eye(PARAMS), rhs)
    fit_sq = sum_sq - 2 * theta @ rhs + theta @ gram @ theta

    # tables carry square-dependent deltas only: the constant per table is piece value
    # (gauge shared with WEIGHT/ADJUST, which the engine pins) -- center it away
    means = np.empty(7)
    means[0] = theta[8:56].mean()  # pawn: 48 legal squares
    theta[8:56] -= means[0]
    theta[:8] = theta[56:64] = 0.0
    for i in range(1, 7):
        means[i] = theta[i * 64 : (i + 1) * 64].mean()
        theta[i * 64 : (i + 1) * 64] -= means[i]
    tables = np.rint(theta).astype(int)

    labels = PST_ORDER + ["KING_EG"]
    deltas = ", ".join(f"{n} {m:+.0f}" for n, m in zip(labels, means))
    print(f"discarded constants (net's implied value deltas): {deltas}", file=sys.stderr)

    print(f"{count:,} positions", file=sys.stderr)
    print(
        f"residual RMSE: material-only {np.sqrt(sum_sq / count):.1f}cp, +PST {np.sqrt(fit_sq / count):.1f}cp",
        file=sys.stderr,
    )

    if args.output:
        with open(args.output, "w") as out:
            emit(tables, out)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        emit(tables, sys.stdout)


if __name__ == "__main__":
    main()
