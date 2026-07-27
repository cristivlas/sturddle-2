#!/usr/bin/env python3
"""
Color-mirror symmetry check: eval(pos) vs -eval(color-mirrored pos).

The net standardizes evals to white POV, so perfect color symmetry means
eval(mirror) == -eval(pos). Per position pair this reports
bias = (eval(pos) + eval(mirror)) / 2 in centipawns; a positive mean says
the evaluation is systematically white-optimistic.

Positions come from an .h5 training set (batch-sampled like outcome_scale)
or an EPD/FEN file. Mirroring is the trainer's flip: swap colors, mirror
ranks, flip STM. Engine FENs are synthesized with no castling/ep rights
(the net ignores them; both sides of a pair are treated identically).

Backends (exactly one):
  -e/--engine: UCI engine command; scores from `go depth N` (STM-POV cp,
      converted to white POV). Use a --dev-mode build to expose WeightsFile.
  -m/--model: weights.bin evaluated with the torch model (float, pre-quant).

Usage:
    ./mirror_check.py mix.h5 --sample 0.001 --limit 5000 -e "./sturddle --dev-mode" -w weights.bin
    ./mirror_check.py mix.h5 --sample 0.01 -m weights.bin
"""

import argparse
import os
import shlex
import subprocess
import sys

import numpy as np

from tqdm import tqdm

PIECES = "kpnbrq"  # column order: (kings, pawns, ..., queens) x (black, white), turn last
BATCH_SIZE = 16384


def parse_fen(fen):
    """FEN -> (13,) uint64 packed row [12 bitboards + turn], trainer layout."""
    fields = fen.split()
    bb = np.zeros(13, dtype=np.uint64)
    sq = 56  # a8
    for ch in fields[0]:
        if ch == "/":
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            col = PIECES.index(ch.lower()) * 2 + (1 if ch.isupper() else 0)
            bb[col] |= np.uint64(1) << np.uint64(sq)
            sq += 1
    bb[12] = fields[1] == "w"
    return bb


def row_to_fen(row):
    """Packed row -> FEN with no castling/ep rights (the net ignores them)."""
    ranks = []
    for rank in range(7, -1, -1):
        s, empty = "", 0
        for file in range(8):
            sq = rank * 8 + file
            ch = None
            for col in range(12):
                if (int(row[col]) >> sq) & 1:
                    ch = PIECES[col // 2].upper() if col % 2 else PIECES[col // 2]
                    break
            if ch:
                if empty:
                    s, empty = s + str(empty), 0
                s += ch
            else:
                empty += 1
        if empty:
            s += str(empty)
        ranks.append(s)
    return "/".join(ranks) + (" w - - 0 1" if row[12] else " b - - 0 1")


def vertical_mirror(bb):
    """Mirror bitboards vertically (rank 1 <-> rank 8). uint64 array in/out."""
    b = bb.astype(np.uint64)
    return (
        ((b >> np.uint64(56)) & np.uint64(0x00000000000000FF))
        | ((b >> np.uint64(40)) & np.uint64(0x000000000000FF00))
        | ((b >> np.uint64(24)) & np.uint64(0x0000000000FF0000))
        | ((b >> np.uint64(8)) & np.uint64(0x00000000FF000000))
        | ((b << np.uint64(8)) & np.uint64(0x000000FF00000000))
        | ((b << np.uint64(24)) & np.uint64(0x0000FF0000000000))
        | ((b << np.uint64(40)) & np.uint64(0x00FF000000000000))
        | ((b << np.uint64(56)) & np.uint64(0xFF00000000000000))
    )


def flip_rows(rows):
    """Color-flip: swap piece colors, vertical mirror, flip STM."""
    flipped = np.empty_like(rows)
    for t in range(6):
        flipped[:, t * 2] = vertical_mirror(rows[:, t * 2 + 1])
        flipped[:, t * 2 + 1] = vertical_mirror(rows[:, t * 2])
    flipped[:, 12] = rows[:, 12] ^ np.uint64(1)
    return flipped


def popcount(bb):
    bb = bb.astype(np.uint64)
    bb = bb - ((bb >> np.uint64(1)) & np.uint64(0x5555555555555555))
    bb = (bb & np.uint64(0x3333333333333333)) + ((bb >> np.uint64(2)) & np.uint64(0x3333333333333333))
    bb = (bb + (bb >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((bb * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def load_rows(path, sample, limit):
    """(N, 13) uint64 packed rows from an .h5 training set or an EPD/FEN file."""
    if path.endswith(".h5"):
        import h5py

        with h5py.File(path, "r") as hf:
            data = hf["data"]
            num_batches = max(1, len(data) // BATCH_SIZE)
            k = max(1, int(num_batches * sample)) if sample else num_batches
            indices = np.random.choice(num_batches, min(k, num_batches), replace=False)
            indices.sort()  # sequential reads are faster in h5
            if limit:  # no point reading batches past the limit
                indices = indices[: (limit + BATCH_SIZE - 1) // BATCH_SIZE]
            print(f"{path}: {len(data):,} rows, using {len(indices)} of {num_batches} batches of {BATCH_SIZE}")
            parts = []
            for i in tqdm(indices, desc="sampling"):
                parts.append(data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, :13].astype(np.uint64))
        rows = np.concatenate(parts)
        if limit and len(rows) > limit:
            keep = np.random.choice(len(rows), limit, replace=False)
            rows = rows[keep]
        return rows

    rows = []
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 2:
                rows.append(parse_fen(" ".join(fields[:2])))
                if limit and len(rows) >= limit:
                    break
    return np.stack(rows)


class EngineBackend:
    def __init__(self, cmd, weights, depth):
        self.depth = depth
        if os.name == "nt":
            argv = [t.strip('"') for t in shlex.split(cmd, posix=False)]
        else:
            argv = shlex.split(cmd)
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._send("uci")
        self._wait("uciok")
        if weights:
            self._send(f"setoption name WeightsFile value {weights}")
        self._sync()

    def _send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def _wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine terminated")
            if line.split() and line.split()[0] == token:
                return

    def _sync(self):
        self._send("isready")
        self._wait("readyok")

    def eval_all(self, rows):
        """White-POV cp per row; NaN on mate scores."""
        out = np.full(len(rows), np.nan)
        for i, row in enumerate(tqdm(rows, desc="eval")):
            self._send("ucinewgame")
            self._sync()
            self._send(f"position fen {row_to_fen(row)}")
            self._send(f"go depth {self.depth}")
            score = None
            while True:
                toks = self.p.stdout.readline().split()
                if not toks:
                    raise RuntimeError("engine terminated")
                if "score" in toks:
                    j = toks.index("score")
                    score = int(toks[j + 2]) if toks[j + 1] == "cp" else None
                if toks[0] == "bestmove":
                    break
            if score is not None:
                out[i] = score if row[12] else -score
        return out

    def close(self):
        self._send("quit")
        self.p.wait(timeout=5)


class ModelBackend:
    def __init__(self, weights):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import torch
        from train_torch import NNUE, load_bin

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NNUE().to(self.device)
        load_bin(self.model, weights)
        self.model.eval()

    def eval_all(self, rows):
        x_all = rows.astype(np.int64)
        out = np.empty(len(rows))
        with self.torch.no_grad():
            for i in tqdm(range(0, len(rows), 4096), desc="eval"):
                x = self.torch.from_numpy(x_all[i : i + 4096]).to(self.device)
                out[i : i + 4096] = self.model(x).cpu().numpy().ravel() * 100.0
        return out

    def close(self):
        pass


PAWN_LABELS = ["0-4", "5-8", "9-12", "13-16"]


def main(args):
    rows = load_rows(args.positions, args.sample, args.limit)
    print(f"{len(rows)} positions")
    mirrors = flip_rows(rows)

    if args.engine:
        backend = EngineBackend(args.engine, args.weights, args.depth)
    else:
        backend = ModelBackend(args.model)
    try:
        v = backend.eval_all(rows)
        vm = backend.eval_all(mirrors)
    finally:
        backend.close()

    ok = ~(np.isnan(v) | np.isnan(vm))
    skipped = int((~ok).sum())
    if skipped:
        print(f"{skipped} pair(s) skipped (mate scores)")
    bias = (v + vm) / 2.0

    pawns = popcount(rows[:, 2]) + popcount(rows[:, 3])
    buckets = np.where(pawns <= 4, 0, np.minimum((pawns - 1) // 4, 3))

    print(f"{'pawns':>7} {'n':>8} {'bias':>8} {'mae':>8} {'|eval|':>8}")
    for b in range(4):
        sel = ok & (buckets == b)
        n = int(sel.sum())
        if n:
            print(
                f"{PAWN_LABELS[b]:>7} {n:>8} {np.mean(bias[sel]):>8.1f}"
                f" {np.mean(np.abs(bias[sel])):>8.1f} {np.mean(np.abs(v[sel])):>8.1f}"
            )
    print(
        f"{'all':>7} {int(ok.sum()):>8} {np.mean(bias[ok]):>8.1f}"
        f" {np.mean(np.abs(bias[ok])):>8.1f} {np.mean(np.abs(v[ok])):>8.1f}"
    )

    if args.top:
        order = np.argsort(-np.abs(np.where(ok, bias, 0.0)))[: args.top]
        print("\nworst offenders (bias, eval, eval-mirror, fen):")
        for i in order:
            if ok[i]:
                print(f"{bias[i]:>8.1f} {v[i]:>8.1f} {vm[i]:>8.1f}  {row_to_fen(rows[i])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("positions", help=".h5 training set, or EPD/FEN file with one position per line")
    parser.add_argument("-e", "--engine", help="UCI engine command (quote to pass flags, e.g. --dev-mode)")
    parser.add_argument("-m", "--model", help="weights.bin for the torch model backend")
    parser.add_argument("-w", "--weights", help="engine WeightsFile (needs a --dev-mode build)")
    parser.add_argument("-d", "--depth", type=int, default=1, help="engine search depth")
    parser.add_argument("--sample", type=float, help="h5 batch sampling ratio, like outcome_scale")
    parser.add_argument("--limit", type=int, default=100_000, help="max positions, 0 = all (random thinning)")
    parser.add_argument("--top", type=int, default=5, help="print the N worst offenders")
    args = parser.parse_args()

    if bool(args.engine) == bool(args.model):
        parser.error("exactly one of --engine / --model is required")
    if args.weights and not args.engine:
        parser.error("--weights applies to the engine backend")

    main(args)
