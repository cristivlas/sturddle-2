#!/usr/bin/env python3
"""
Feasibility check for NNUE_L2_INT16 (hidden_2 in int16, madd_epi16 dot) with the
current weights.bin, no trainer changes.

Static: pool and hidden_2 weights must fit int16 at WQ_SCALE; hidden_2 same-sign column
sums bound the int32 accumulator when inputs are int16-capped.
Empirical: run positions (EPD suites + random continuations) through a numpy forward pass
mirroring train_torch.py and report the ranges that must fit int16 / int32.

    python tools/nnue/l2_int16_check.py weights.bin test/TEST_SUITES/*.epd
"""
import argparse
import glob
import random

import chess
import numpy as np

ACTIVE_INPUTS = 769
ACCUMULATOR_SIZE = 2048
POOL_SIZE = 8
POOLED = ACCUMULATOR_SIZE // POOL_SIZE
MAIN_BUCKETS = 16
INPUTS_B = 256
HIDDEN_2 = 16
HIDDEN_3 = 16

Q_SCALE = 1024
# mirror nnue.h: hidden_2 input (AQSCALE) and pool / hidden_2 weights (WQSCALE)
AQ_SCALE = Q_SCALE * 4
WQ_SCALE = Q_SCALE * 4
ACC_CAP = 32767 / Q_SCALE  # accumulator and hidden_1b, int16 at Q_SCALE
INT16_CAP = 32767 / AQ_SCALE  # hidden_2 input, int16 at AQ_SCALE
W_CAP = 32767 / WQ_SCALE  # pool and hidden_2 weights, int16 at WQ_SCALE
INT32_CAP = 2**31 / (AQ_SCALE * WQ_SCALE)  # hidden_2 int32 sums


EXPORT = [
    ("hidden_1a", MAIN_BUCKETS * ACTIVE_INPUTS, ACCUMULATOR_SIZE, ACCUMULATOR_SIZE),
    ("hidden_1b", INPUTS_B, POOLED, POOLED),
    ("pool", 2 * ACCUMULATOR_SIZE, 1, 0),
    ("hidden_2", POOLED, HIDDEN_2, HIDDEN_2),
    ("hidden_3", HIDDEN_2, HIDDEN_3, HIDDEN_3),
    ("out", HIDDEN_3, 1, 1),
]


def load_bin(path):
    data = np.fromfile(path, dtype=np.float32)
    expected = sum(i * o + b for _, i, o, b in EXPORT)
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} floats, got {data.size}")
    layers = {}
    off = 0
    for name, i, o, bn in EXPORT:
        k = data[off : off + i * o].reshape(i, o)
        off += i * o
        b = data[off : off + bn]
        off += bn
        layers[name] = (k, b)
    return layers


def unpack_bits(packed):
    bitboards = packed[:, :-1].astype(np.uint64)
    shifts = np.arange(63, -1, -1, dtype=np.uint64)
    bits = (bitboards[:, :, None] >> shifts) & np.uint64(1)
    feats = bits.reshape(bits.shape[0], -1).astype(np.float32)
    return np.concatenate([feats, packed[:, -1:].astype(np.float32)], axis=1)


_RIGHT = np.array([1.0 if ((63 - i) % 8) >= 4 else 0.0 for i in range(64)], dtype=np.float32)


def bucket_id(f):
    pawns = f[:, 128:256].sum(axis=1)
    pawn_id = np.where(pawns <= 4.0, 0.0, np.minimum(np.floor((pawns - 1.0) / 4.0), 3.0)).astype(int)
    wk = (f[:, 64:128] * _RIGHT).sum(axis=1).astype(int)
    bk = (f[:, 0:64] * _RIGHT).sum(axis=1).astype(int)
    return pawn_id * 4 + wk * 2 + bk


def encode(board):
    mb, mw = board.occupied_co[chess.BLACK], board.occupied_co[chess.WHITE]
    bbs = [
        [p & mb, p & mw]
        for p in (board.kings, board.pawns, board.knights, board.bishops, board.rooks, board.queens)
    ]
    return np.append(np.asarray(bbs, dtype=np.uint64).ravel(), np.uint64(board.turn))


def positions(paths, plies, seed):
    rng = random.Random(seed)
    boards = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                board = chess.Board(" ".join(line.split()[:4]) + " 0 1")
                boards.append(board.copy())
                for n in plies:
                    b = board.copy()
                    for _ in range(n):
                        moves = list(b.legal_moves)
                        if not moves:
                            break
                        b.push(rng.choice(moves))
                    if b.is_valid():
                        boards.append(b)
    return boards


def forward(layers, packed):
    feats = unpack_bits(packed)
    w1a, b1a = layers["hidden_1a"]
    blocks = w1a.reshape(MAIN_BUCKETS, ACTIVE_INPUTS, ACCUMULATOR_SIZE)
    bid = bucket_id(feats)
    pre = np.empty((feats.shape[0], ACCUMULATOR_SIZE), dtype=np.float32)
    for b in range(MAIN_BUCKETS):
        rows = np.flatnonzero(bid == b)
        if rows.size:
            pre[rows] = feats[rows] @ blocks[b]
    pre += b1a
    acc = np.maximum(pre, 0)

    w1b, b1b = layers["hidden_1b"]
    mod = feats[:, :INPUTS_B] @ w1b + b1b

    pool = layers["pool"][0].reshape(2, POOLED, POOL_SIZE)
    stm = packed[:, -1].astype(int)
    pooled = (acc.reshape(-1, POOLED, POOL_SIZE) * pool[stm]).sum(axis=-1)
    residual = pooled * (1 + mod)

    w2, b2 = layers["hidden_2"]
    pre2 = residual @ w2 + b2
    abs2 = np.abs(residual) @ np.abs(w2)  # bounds every partial sum of the int32 accumulators
    return dict(pre=pre, acc=acc, mod=mod, pooled=pooled, residual=residual, pre2=pre2, abs2=abs2)


def report_static(layers):
    print("== static (weights only) ==")
    pool = layers["pool"][0].reshape(2, POOLED, POOL_SIZE)
    print(f"pool     max|w| {np.abs(pool).max():.4f}  int16@WQ cap {W_CAP:.2f}")
    group = np.abs(pool).sum(axis=-1).max()
    print(f"pool     max group sum|w| {group:.4f}; int32 safe if < {2**31 / (32767 * WQ_SCALE):.2f} (static, engine checks at load)")
    err = np.abs(pool - np.round(pool * WQ_SCALE) / WQ_SCALE)
    print(f"pool     max rounding err @WQ {err.max():.2e}  (relative to 1/{POOL_SIZE}: {err.max() * POOL_SIZE:.2e})")

    w2, b2 = layers["hidden_2"]
    print(f"hidden_2 max|w| {np.abs(w2).max():.4f}  max|b| {np.abs(b2).max():.4f}  int16@WQ cap {W_CAP:.2f}")
    pos = np.maximum(w2, 0).sum(axis=0)
    neg = -np.minimum(w2, 0).sum(axis=0)
    col = np.maximum(pos, neg)
    print(
        f"hidden_2 max same-sign column sum {col.max():.4f}; int32 safe if < {INT32_CAP / INT16_CAP:.1f} "
        f"with int16-capped inputs (static worst case)"
    )
    err = np.abs(w2 - np.round(w2 * WQ_SCALE) / WQ_SCALE)
    print(f"hidden_2 max rounding err @WQ {err.max():.2e}, mean {err.mean():.2e}")


def report_empirical(layers, boards, batch):
    print(f"== empirical ({len(boards)} positions) ==")
    packed = np.stack([encode(b) for b in boards])
    stats = {}

    def track(name, v, signed=True):
        lo, hi = float(v.min()), float(v.max())
        s = stats.setdefault(name, [lo, hi])
        s[0], s[1] = min(s[0], lo), max(s[1], hi)

    for i in range(0, packed.shape[0], batch):
        r = forward(layers, packed[i : i + batch])
        track("acc pre-relu", r["pre"])
        track("mod (1b)", r["mod"])
        track("pooled", r["pooled"])
        track("residual = pooled*(1+mod)", r["residual"])
        track("hidden_2 pre-act", r["pre2"])
        track("hidden_2 sum|x||w|", r["abs2"])

    caps = {
        "acc pre-relu": ACC_CAP,
        "mod (1b)": ACC_CAP,
        "pooled": INT16_CAP,
        "residual = pooled*(1+mod)": INT16_CAP,
        "hidden_2 pre-act": INT32_CAP,
        "hidden_2 sum|x||w|": INT32_CAP,
    }
    for name, (lo, hi) in stats.items():
        cap = caps[name]
        worst = max(abs(lo), abs(hi))
        flag = "  OVERFLOW" if worst >= cap else f"  headroom x{cap / max(worst, 1e-9):.1f}"
        print(f"{name:28s} [{lo:10.4f}, {hi:10.4f}]  cap {cap:8.2f}{flag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("epd", nargs="*", help="EPD files (globs ok)")
    p.add_argument("--plies", default="0,8,16,32,64", help="random continuations per EPD position")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch", type=int, default=512)
    a = p.parse_args()

    layers = load_bin(a.weights)
    print(a.weights)
    report_static(layers)

    paths = [f for g in a.epd for f in glob.glob(g)]
    if paths:
        plies = [int(x) for x in a.plies.split(",") if int(x) > 0]
        boards = positions(paths, plies, a.seed)
        report_empirical(layers, boards, a.batch)


if __name__ == "__main__":
    main()
