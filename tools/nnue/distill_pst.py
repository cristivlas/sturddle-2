#!/usr/bin/env python3
import argparse
import os
import re
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

import train_torch as tt

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

KING_PRECISION = 1.0  # cp: one rounding step of the emitted tables
CONFIDENCE_Z = 1.96  # 95% two-sided
HUBER_K = 1.345  # 95% efficiency under normal noise
MAD_SCALE = 1.4826  # MAD -> std under normal noise


def pawn_bucket(pawn_count):
    return torch.where(pawn_count <= 4, 0, ((pawn_count - 1) // 4).clamp(max=PAWN_BUCKETS - 1))


def unpack(packed):
    """(B, 13) int64 tensor -> (B, 12, 64) bool, index == chess square (A1=0)."""
    return ((packed[:, :12, None] >> torch.arange(64, device=packed.device)) & 1).bool()


def mirror(bits):
    """Flip ranks: square index sq -> sq ^ 56 (white PST lookup)."""
    return bits.reshape(-1, 8, 8).flip(1).reshape(-1, 64)


def features_and_material(packed):
    """(B, 13) int64 tensor -> (B, PARAMS) float32 signed PST features and (B,) float64 graded material."""
    bits = unpack(packed).float()
    n = packed.shape[0]

    black = {name: bits[:, base] for name, base in H5_BASE.items()}
    white = {name: bits[:, base + 1] for name, base in H5_BASE.items()}

    eg = (bits.sum(dim=(1, 2)) <= ENDGAME_PIECE_COUNT).float()[:, None]

    buckets = pawn_bucket((black["PAWN"].sum(dim=1) + white["PAWN"].sum(dim=1)).long())
    adjust = torch.tensor(GRADING_ADJUST, dtype=torch.float64, device=packed.device)[buckets]  # (B, 7)

    x = torch.empty((n, PARAMS), device=packed.device)
    material = torch.zeros(n, dtype=torch.float64, device=packed.device)
    for i, name in enumerate(PST_ORDER):
        block = mirror(white[name]) - black[name]
        if name == "KING":
            x[:, i * 64 : (i + 1) * 64] = block * (1.0 - eg)
            x[:, (i + 1) * 64 :] = block * eg
        else:
            x[:, i * 64 : (i + 1) * 64] = block
            value = WEIGHT[PIECE_TYPE[name]] + adjust[:, PIECE_TYPE[name]]
            material += value * (white[name].sum(dim=1) - black[name].sum(dim=1))
    return x, material


def _leaper(steps):
    m = np.zeros((64, 64), dtype=np.float32)
    for s in range(64):
        f, r = s % 8, s // 8
        for df, dr in steps:
            if 0 <= f + df < 8 and 0 <= r + dr < 8:
                m[s, (r + dr) * 8 + f + df] = 1.0
    return m


def _rays(steps):
    rays = np.full((len(steps), 64, 7), 64)  # 64: off-board sentinel
    for d, (df, dr) in enumerate(steps):
        for s in range(64):
            f, r = s % 8, s // 8
            for k in range(1, 8):
                if not (0 <= f + df * k < 8 and 0 <= r + dr * k < 8):
                    break
                rays[d, s, k - 1] = (r + dr * k) * 8 + f + df * k
    return rays


ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT_ATTACKS = _leaper([(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)])
KING_ATTACKS = _leaper(ORTHO + DIAG)
PAWN_ATTACKS = [_leaper([(-1, -1), (1, -1)]), _leaper([(-1, 1), (1, 1)])]  # [black, white][from, to]
ORTHO_RAYS, DIAG_RAYS = _rays(ORTHO), _rays(DIAG)


def attacked(occ, side, color):
    """(B, 64) bool tensor: squares attacked by side's pieces (dict of (B, 64) bool), sliders blocked by occ."""
    dev = occ.device
    leapers = ((side["KNIGHT"], KNIGHT_ATTACKS), (side["KING"], KING_ATTACKS), (side["PAWN"], PAWN_ATTACKS[color]))
    att = sum(bits.float() @ torch.as_tensor(m, device=dev) for bits, m in leapers) > 0
    stop = torch.ones((occ.shape[0], 1), dtype=torch.bool, device=dev)
    occ_ext = torch.cat([occ, stop], dim=1)
    squares = torch.arange(64, device=dev)
    for rays, movers in ((ORTHO_RAYS, side["ROOK"] | side["QUEEN"]), (DIAG_RAYS, side["BISHOP"] | side["QUEEN"])):
        movers_ext = torch.cat([movers, ~stop], dim=1)
        for ray in torch.as_tensor(rays, device=dev):
            first = ray[squares, occ_ext[:, ray].byte().argmax(dim=2)]  # (B, 64) first blocker
            att |= torch.gather(movers_ext, 1, first)
    return att


def net_eval(model, packed, chunk):
    """(B, 13) int64 tensor -> (B,) float64 white-view net evals in cp."""
    out = [torch.zeros(0, dtype=torch.float64, device=packed.device)]
    for s in range(0, len(packed), chunk):
        p = packed[s : s + chunk]
        y = model(p)[:, 0].double() * tt.SCALE
        out.append(torch.where(p[:, 12] == 1, y, -y))
    return torch.cat(out)


def king_moves(model, packed, chunk):
    """Net eval of each king relocated to every legal square, everything else fixed.

    Returns endgame mask and, per color, (legal, evals, king) as (B, 64) in PST index order, evals from the king's side."""
    bits = unpack(packed)
    pieces = [{name: bits[:, base + c] for name, base in H5_BASE.items()} for c in (0, 1)]
    occ = bits.any(dim=1)
    eg = occ.sum(dim=1) <= ENDGAME_PIECE_COUNT

    out = []
    for c in (0, 1):
        own, enemy = pieces[c], pieces[1 - c]
        occ_c = occ & ~own["KING"]
        legal = ~occ_c & ~attacked(occ_c, enemy, 1 - c)
        own_rest = dict(own, KING=torch.zeros_like(own["KING"]))
        checks = (attacked(occ_c, own_rest, c) & enemy["KING"]).any(dim=1)
        legal &= ~checks[:, None]  # skip positions where moving the king gives check

        b, t = torch.nonzero(legal, as_tuple=True)
        moved = packed[b]
        moved[:, H5_BASE["KING"] + c] = torch.ones_like(t) << t
        evals = torch.zeros(legal.shape, dtype=torch.float64, device=packed.device)
        evals[b, t] = net_eval(model, moved, chunk) * (1 if c else -1)
        king = own["KING"]
        out.append((mirror(legal), mirror(evals), mirror(king)) if c else (legal, evals, king))
    return eg, out


def read_slices(data, starts, slice_rows):
    # one read for all slices: a VDS walks every mapping per read call
    space = data.id.get_space()
    space.select_none()
    for s in starts:
        space.select_hyperslab((int(s), 0), (slice_rows, data.shape[1]), op=h5py.h5s.SELECT_OR)
    rows = np.empty((len(starts) * slice_rows, data.shape[1]), dtype=data.dtype)
    data.id.read(h5py.h5s.create_simple(rows.shape), space, rows)
    return rows


def king_batch_stats(eg, sides, table, cluster):
    """King fixed-effects normal equations for one batch, deltas Huber-clipped on residuals vs table (2, 64).

    Returns gram, rhs, meat (per-cluster score outer products, for clustered standard errors) and king visits."""
    dev = eg.device
    stats = [torch.zeros(shape, dtype=torch.float64, device=dev) for shape in ((2, 64, 64), (2, 64), (2, 64, 64), (2, 64))]
    gram, rhs, meat, visits = stats
    fitted = torch.where(eg[:, None], table[1], table[0])  # (B, 64)
    clusters = int(cluster.max()) + 1
    for phase, sel in enumerate((~eg, eg)):
        scores = torch.zeros((clusters, 64), dtype=torch.float64, device=dev)
        for legal, evals, king in sides:
            m = legal[sel]
            if not m.any():
                continue
            size = m.sum(dim=1).clamp(min=1)[:, None].double()
            resid = evals[sel] - fitted[sel]
            resid = resid - torch.where(m, resid, float("nan")).nanmedian(dim=1, keepdim=True).values
            limit = HUBER_K * MAD_SCALE * resid[m].abs().median()
            clipped = torch.where(m, evals[sel] - resid + resid.clamp(-limit, limit), 0.0)
            centered = torch.where(m, clipped - clipped.sum(dim=1, keepdim=True) / size, 0.0)
            md = m.double()
            gram[phase] += torch.diag(md.sum(dim=0)) - (md / size).T @ md
            rhs[phase] += centered.sum(dim=0)
            visits[phase] += king[sel].sum(dim=0)
            explained = torch.where(m, fitted[sel] - (fitted[sel] * md).sum(dim=1, keepdim=True) / size, 0.0)
            scores.index_add_(0, cluster[sel], centered - explained)
        meat[phase] += scores.T @ scores
    return stats


def king_solve(gram, rhs, ridge):
    eye = torch.eye(64, dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + ridge * eye, rhs[..., None])[..., 0]


def king_error(gram, meat, visits, ridge):
    """Clustered standard error of the fitted king table, RMS over squares weighted by how often the king sits there."""
    if not visits.sum():
        return np.inf
    inv = np.linalg.inv(gram + ridge * np.eye(64))
    p = visits / visits.sum()
    proj = np.eye(64) - np.outer(np.ones(64), p)  # a constant shift cancels between the two kings
    cov = proj @ inv @ meat @ inv @ proj.T
    return np.sqrt(p @ np.diag(cov))


def load_model(bin_path, device):
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
    p = argparse.ArgumentParser(
        description="Fit piece-square tables to a trained net's evals, minus material, and write them in tables.h format."
    )
    p.add_argument("data", nargs="+", help="H5 training data file(s)")
    p.add_argument("-m", "--model", default="weights.bin", help="weights.bin")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    p.add_argument("-b", "--batch-size", type=int, default=65536, help="rows per read")
    p.add_argument("--limit", type=int, help="max positions per file, sampled as random slices")
    p.add_argument("--slice-rows", type=int, default=256, help="rows per random slice")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("--ridge", type=float, default=1.0, help="L2 penalty for the least-squares fit")
    p.add_argument("--device", help="default: cuda if available, else cpu")
    p.add_argument("--eval-chunk", type=int, default=65536, help="positions per net forward pass")
    args = p.parse_args()

    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}", file=sys.stderr)

    model = load_model(args.model, args.device)
    rng = np.random.default_rng(args.seed)

    gram = torch.zeros((PARAMS, PARAMS), dtype=torch.float64, device=args.device)
    rhs = torch.zeros(PARAMS, dtype=torch.float64, device=args.device)
    king_stats = [torch.zeros(shape, dtype=torch.float64, device=args.device) for shape in ((2, 64, 64), (2, 64), (2, 64, 64), (2, 64))]
    king_gram, king_rhs, king_meat, king_visits = king_stats  # [middlegame, endgame]
    king_done, king_rows = False, 0

    with torch.no_grad():
        for path in args.data:
            with h5py.File(path, "r") as hf:
                data = hf["data"]
                total = data.shape[0]
                if args.limit and args.limit < total:
                    slices = total // args.slice_rows
                    k = min(slices, -(-args.limit // args.slice_rows))
                    starts = np.sort(rng.choice(slices, size=k, replace=False)) * args.slice_rows
                    groups = np.array_split(starts, -(-k * args.slice_rows // args.batch_size))
                    reads = (read_slices(data, g, args.slice_rows) for g in groups)
                    n, num_batches = k * args.slice_rows, len(groups)
                else:
                    reads = (data[s : s + args.batch_size] for s in range(0, total, args.batch_size))
                    n, num_batches = total, -(-total // args.batch_size)
                print(f"{path}: {n:,} rows", file=sys.stderr)
                bar = tqdm(reads, total=num_batches, unit="batch")
                for rows in bar:
                    keep = (tt.popcount(rows[:, 0]) == 1) & (tt.popcount(rows[:, 1]) == 1)  # one king per side
                    rows = rows[keep]
                    packed = torch.from_numpy(rows[:, :13].astype(np.int64)).to(args.device)
                    x, material = features_and_material(packed)
                    y = net_eval(model, packed, args.eval_chunk) - material
                    gram += (x.T @ x).double()
                    rhs += x.double().T @ y

                    if king_done:
                        continue
                    # king squares: per position, fit eval differences across legal squares
                    eg, sides = king_moves(model, packed, args.eval_chunk)
                    cluster = torch.from_numpy(np.flatnonzero(keep) // args.slice_rows).to(args.device)  # rows of a slice are correlated
                    # heavy tails dominate the noise: Huber via reweighting, refit until the table settles
                    table = king_solve(king_gram, king_rhs, args.ridge)
                    while True:
                        batch = king_batch_stats(eg, sides, table, cluster)
                        update = king_solve(king_gram + batch[0], king_rhs + batch[1], args.ridge)
                        p = king_visits + batch[3]
                        p = p / p.sum(dim=1, keepdim=True).clamp(min=1)
                        step = update - table
                        step = step - (p * step).sum(dim=1, keepdim=True)  # constant shift is free
                        table = update
                        if (p * step**2).sum(dim=1).max().sqrt() <= KING_PRECISION / CONFIDENCE_Z:
                            break
                    for acc, part in zip(king_stats, batch):
                        acc += part
                    king_rows += len(rows)
                    stats = [a.cpu().numpy() for a in king_stats]
                    errors = [king_error(stats[0][i], stats[2][i], stats[3][i], args.ridge) for i in (0, 1)]
                    bar.set_postfix(king=f"{CONFIDENCE_Z * max(errors):.1f}cp")
                    king_done = CONFIDENCE_Z * max(errors) <= KING_PRECISION
                    if king_done:
                        print(f"king tables converged after {king_rows:,} rows", file=sys.stderr)

    # table means stay in: engine adds tables to WEIGHT, so the mean is the net's value delta
    theta = np.linalg.solve(gram.cpu().numpy() + args.ridge * np.eye(PARAMS), rhs.cpu().numpy())
    theta[:8] = theta[56:64] = 0.0  # no pawns on ranks 1 and 8
    king = PST_ORDER.index("KING") * 64
    theta[king : king + 128] = king_solve(king_gram, king_rhs, args.ridge).cpu().numpy().ravel()
    tables = np.rint(theta).astype(int)

    if args.output:
        with open(args.output, "w") as out:
            emit(tables, out)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        emit(tables, sys.stdout)


if __name__ == "__main__":
    main()
