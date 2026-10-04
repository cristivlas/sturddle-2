#! /usr/bin/env python3
"""Estimate material values per pawn bucket, consistent with the NNUE runtime eval."""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

# H5 bitboard columns: [kings, pawns, knights, bishops, rooks, queens] x [black, white], then turn.
COL_BK, COL_WK = 0, 1
COL_BP, COL_WP = 2, 3
COL_BN, COL_WN = 4, 5
COL_BB, COL_WB = 6, 7
COL_BR, COL_WR = 8, 9
COL_BQ, COL_WQ = 10, 11
COL_TURN = 12

PIECES = ("PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN")
PIECE_COLS = ((COL_WP, COL_BP), (COL_WN, COL_BN), (COL_WB, COL_BB), (COL_WR, COL_BR), (COL_WQ, COL_BQ))

WHITE_COLS = (COL_WK, COL_WP, COL_WN, COL_WB, COL_WR, COL_WQ)
BLACK_COLS = (COL_BK, COL_BP, COL_BN, COL_BB, COL_BR, COL_BQ)

NUM_BUCKETS = 4
PAWN_RANGES = ("0-4", "5-8", "9-12", "13-16")

ENDGAME_PIECE_COUNT = 12  # common.h

DEFAULT_ANCHOR_VALUE = 100


def popcount64(a):
    v = a.astype(np.uint64)
    v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((v * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def compute_buckets(packed):
    """Pawn dimension of nnue.h get_bucket(); the king dimension tracks position quality, not material."""
    pawns = popcount64(packed[:, COL_WP]) + popcount64(packed[:, COL_BP])
    return np.where(pawns <= 4, 0, np.minimum((pawns - 1) // 4, 3))


def parse_square_tables(path):
    """Read SQUARE_TABLE and ENDGAME_KING_SQUARE_TABLE out of tables.h."""
    import re

    text = open(path).read()
    body = re.search(r"int SQUARE_TABLE\[\]\[64\] = \{(.*?)\n\};", text, re.S).group(1)

    tables = {}
    for name in ("PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"):
        block = re.search(r"\{\s*/\* " + name + r" \*/(.*?)\}", body, re.S).group(1)
        nums = [int(v) for v in re.findall(r"-?\d+", block)]
        if len(nums) != 64:
            raise ValueError(f"{path}: {name} table has {len(nums)} entries, expected 64")
        tables[name] = np.array(nums, dtype=np.float64)

    eg = re.search(r"int ENDGAME_KING_SQUARE_TABLE\[64\] = \{(.*?)\n\};", text, re.S).group(1)
    tables["KING_ENDGAME"] = np.array([int(v) for v in re.findall(r"-?\d+", eg)], dtype=np.float64)
    return tables


def build_pst_matrix(tables, endgame):
    """(12, 64) lookup keyed by H5 column: white mirrors the square, black uses it directly."""
    mat = np.zeros((12, 64))
    order = (
        ("KING", COL_WK, COL_BK),
        ("PAWN", COL_WP, COL_BP),
        ("KNIGHT", COL_WN, COL_BN),
        ("BISHOP", COL_WB, COL_BB),
        ("ROOK", COL_WR, COL_BR),
        ("QUEEN", COL_WQ, COL_BQ),
    )
    squares = np.arange(64)
    for name, wcol, bcol in order:
        table = tables["KING_ENDGAME"] if (endgame and name == "KING") else tables[name]
        mat[wcol] = table[squares ^ 0x38]  # square_mirror
        mat[bcol] = table[squares]
    return mat


def unpack_bits(packed):
    """(N, 12) bitboards -> (N, 12, 64) float bit matrix; bit i of the word is square i."""
    shifts = np.arange(64, dtype=np.uint64)
    return ((packed[:, :12, None] >> shifts) & np.uint64(1)).astype(np.float64)


def pst_scores(packed, mats):
    """White-POV piece-square sum, picking the endgame table by total piece count."""
    bits = unpack_bits(packed)
    total = popcount64(packed[:, :12]).sum(axis=1)

    out = np.empty(len(packed))
    for endgame, mat in mats.items():
        rows = (total <= ENDGAME_PIECE_COUNT) == endgame
        if rows.any():
            per_col = np.einsum("nps,ps->np", bits[rows], mat)
            out[rows] = per_col[:, WHITE_COLS].sum(axis=1) - per_col[:, BLACK_COLS].sum(axis=1)
    return out


def piece_features(packed):
    """Signed piece counts, white minus black."""
    return np.stack(
        [popcount64(packed[:, w]) - popcount64(packed[:, b]) for w, b in PIECE_COLS],
        axis=1,
    ).astype(np.float64)


# Backends all return centipawns from WHITE's POV, matching nnue::eval_fen.
class TorchBackend:
    name = "torch"

    def __init__(self, weights, device=None):
        import torch

        import train_torch as tt

        self.torch = torch
        self.model = tt.NNUE()
        tt.load_bin(self.model, weights)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def eval_batch(self, packed):
        x = self.torch.from_numpy(packed[:, :13].astype(np.int64)).to(self.device)
        with self.torch.no_grad():
            out = self.model(x)[:, 0]
        return out.float().cpu().numpy().astype(np.float64) * 100.0


class TFBackend:
    name = "tf"

    def __init__(self, model_path):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "ACCUMULATOR_SIZE": 2048,
                "POOL_SIZE": 8,
                "combined_loss": None,
                "scaled_sparse_categorical_crossentropy": None,
                "top": None,
                "top_3": None,
                "top_5": None,
            },
        )

    def eval_batch(self, packed):
        out = self.model.predict(packed[:, :13], verbose=0)
        res = out[0] if isinstance(out, (list, tuple)) else out
        return np.asarray(res).reshape(-1).astype(np.float64) * 100.0


class EngineBackend:
    name = "engine"

    def __init__(self):
        import chess_engine

        self.nnue_eval_fen = chess_engine.nnue_eval_fen

    def eval_batch(self, packed, progress=False):
        rows = tqdm(packed, desc="Engine eval", unit=" pos", leave=False) if progress else packed
        return np.array([self.nnue_eval_fen(packed_to_fen(row)) for row in rows], dtype=np.float64)


def make_backend(args):
    if args.backend == "torch":
        return TorchBackend(args.weights, args.device)
    if args.backend == "tf":
        if not args.model:
            sys.exit("--backend tf requires --model <keras dir>")
        return TFBackend(args.model)
    return EngineBackend()


def packed_to_fen(row):
    import chess

    board = chess.Board(None)
    for col, piece, color in (
        (COL_WK, chess.KING, chess.WHITE),
        (COL_BK, chess.KING, chess.BLACK),
        (COL_WP, chess.PAWN, chess.WHITE),
        (COL_BP, chess.PAWN, chess.BLACK),
        (COL_WN, chess.KNIGHT, chess.WHITE),
        (COL_BN, chess.KNIGHT, chess.BLACK),
        (COL_WB, chess.BISHOP, chess.WHITE),
        (COL_BB, chess.BISHOP, chess.BLACK),
        (COL_WR, chess.ROOK, chess.WHITE),
        (COL_BR, chess.ROOK, chess.BLACK),
        (COL_WQ, chess.QUEEN, chess.WHITE),
        (COL_BQ, chess.QUEEN, chess.BLACK),
    ):
        bb = int(row[col])
        while bb:
            lsb = bb & -bb
            board.set_piece_at(lsb.bit_length() - 1, chess.Piece(piece, color))
            bb ^= lsb
    board.turn = bool(row[COL_TURN])
    return board.fen()


def sample_rows(path, count, seed, chunk):
    """Contiguous blocks at random offsets: sequential reads, spread over the file."""
    import h5py

    with h5py.File(path, "r") as f:
        data = f["data"]
        total = data.shape[0]
        count = min(count, total)
        chunk = min(chunk, count)

        rng = np.random.default_rng(seed)
        nblocks = (count + chunk - 1) // chunk
        offsets = np.sort(rng.integers(0, max(1, total - chunk), size=nblocks))

        with tqdm(total=count, desc="Sampling", unit=" pos") as bar:
            got = 0
            for off in offsets:
                take = min(chunk, count - got)
                yield data[off : off + take, :13]
                got += take
                bar.update(take)


class Accumulator:
    """Per-bucket normal equations, so memory stays flat regardless of sample count."""

    def __init__(self, num_terms):
        self.n = num_terms + 1  # + intercept
        self.gram = np.zeros((NUM_BUCKETS, self.n, self.n))
        self.rhs = np.zeros((NUM_BUCKETS, self.n))
        self.counts = np.zeros(NUM_BUCKETS, dtype=np.int64)
        self.ysum = np.zeros(NUM_BUCKETS)
        self.yy = np.zeros(NUM_BUCKETS)

    def add(self, buckets, features, evals):
        design = np.hstack([features, np.ones((len(features), 1))])
        for b in np.unique(buckets):
            rows = buckets == b
            d, y = design[rows], evals[rows]
            self.gram[b] += d.T @ d
            self.rhs[b] += d.T @ y
            self.counts[b] += len(y)
            self.ysum[b] += y.sum()
            self.yy[b] += float(y @ y)


def fit_bucket(acc, b, ridge):
    """Returns (raw values, r2, rmse), or None if the fit is degenerate."""
    n = int(acc.counts[b])
    if n < 2 * acc.n:
        return None

    # Intercept absorbs bucket-wide bias (e.g. tempo) so it stays out of the coefficients.
    gram, rhs = acc.gram[b].copy(), acc.rhs[b]
    if ridge:
        reg = np.eye(acc.n) * ridge
        reg[-1, -1] = 0.0
        gram = gram + reg

    try:
        coef = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return None

    values = dict(zip(PIECES, coef[: len(PIECES)]))

    # ||y - Xb||^2 = y'y - 2b'X'y + b'(X'X)b, all available from the moments.
    ss_res = max(0.0, acc.yy[b] - 2.0 * float(coef @ rhs) + float(coef @ acc.gram[b] @ coef))
    ss_tot = max(0.0, acc.yy[b] - acc.ysum[b] ** 2 / n)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / n))

    return values, r2, rmse


def format_table(name, values):
    body = ", ".join(str(int(round(values[p]))) for p in PIECES)
    return f"#define {name} {{ 0, {body}, 20000 }}"


class Reservoir:
    """Uniform sample of the stream, held back for the engine cross-check."""

    def __init__(self, size, seed):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.packed = np.empty((size, 13), dtype=np.uint64)
        self.evals = np.empty(size)
        self.seen = 0

    def add(self, packed, evals):
        n = len(evals)
        fill = max(0, min(self.size - self.seen, n))
        if fill > 0:
            self.packed[self.seen : self.seen + fill] = packed[:fill]
            self.evals[self.seen : self.seen + fill] = evals[:fill]

        # Each later row replaces a random slot with probability size/seen.
        if n > fill:
            pos = self.seen + fill + np.arange(n - fill)
            take = self.rng.random(n - fill) < self.size / (pos + 1)
            if take.any():
                slots = self.rng.integers(0, self.size, size=int(take.sum()))
                self.packed[slots] = packed[fill:][take]
                self.evals[slots] = evals[fill:][take]

        self.seen += n


def verify(reservoir, backend):
    if backend.name == "engine":
        print("--verify skipped: already using the engine backend")
        return

    try:
        engine = EngineBackend()
    except ImportError as e:
        print(f"--verify skipped: engine backend unavailable ({e})")
        return

    n = min(reservoir.size, reservoir.seen)
    diff = np.abs(engine.eval_batch(reservoir.packed[:n], progress=True) - reservoir.evals[:n])
    print(f"Verify vs engine on {len(diff):,}: mean |diff| {diff.mean():.2f} cp, max {diff.max():.2f} cp")
    if diff.max() > 25:
        print("  WARNING: large divergence - check that --weights matches the built-in net")


def report(fits, counts, args):
    # Population-weighted, so adjustments average to ~0 over the sampled distribution.
    total = sum(counts[b] for b in fits)
    baseline = {p: sum(fits[b][p] * counts[b] for b in fits) / total for p in PIECES}

    print()
    print("=" * 72)
    print(f"BASELINE (population-weighted over {len(fits)} buckets, {total:,} positions)")
    print("=" * 72)
    # Ratios survive across datasets better than absolutes, so show both.
    ref = baseline["KNIGHT"]
    print(f"{'piece':>8} {'fitted':>9} {'x knight':>9}")
    for piece in PIECES:
        print(f"{piece:>8} {baseline[piece]:>9.1f} {baseline[piece] / ref:>9.2f}")

    print()
    print("=" * 72)
    print("PER-BUCKET ADJUSTMENTS (fitted - baseline)")
    print("=" * 72)
    print(f"{'bucket':>6} {'pawns':>7} {'n':>9} " + " ".join(f"{p[:5]:>7}" for p in PIECES))
    for b in range(NUM_BUCKETS):
        if b not in fits:
            continue
        adj = " ".join(f"{fits[b][p] - baseline[p]:>+7.1f}" for p in PIECES)
        print(f"{b:>6} {PAWN_RANGES[b]:>7} {counts[b]:>9} {adj}")

    print()
    print("=" * 72)
    print("chess.h drop-in")
    print("=" * 72)
    print(format_table("PIECE_VALUES", baseline))
    print()
    print("/* GRADING_ADJUST[PAWN_BUCKETS][7], indexed by chess::pawn_bucket() */")
    print("#define GRADING_ADJUST { \\")
    base_int = {p: int(round(baseline[p])) for p in PIECES}
    for b in range(NUM_BUCKETS):
        if b in fits:
            adj_int = {p: int(round(fits[b][p] - baseline[p])) for p in PIECES}
            body = ", ".join(f"{adj_int[p]}" for p in PIECES)
            # comment shows base + adjust as the engine computes it, not the re-rounded fit
            note = ", ".join(f"{base_int[p] + adj_int[p]}" for p in PIECES)
        else:
            body = ", ".join("0" for _ in PIECES)
            note = "no fit"
        print(f"    {{ 0, {body}, 0 }}, /* {PAWN_RANGES[b]:>5} pawns: {note} */ \\")
    print("}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5", help="H5 dataset (or virtual dataset) to sample from")
    p.add_argument("-n", "--count", type=int, default=200000, help="positions to sample (default: 200000)")
    p.add_argument("--backend", choices=("torch", "engine", "tf"), default="torch", help="eval backend")
    p.add_argument("--weights", default="weights.bin", help="flat weights.bin for --backend torch")
    p.add_argument("--model", help="Keras model dir for --backend tf")
    p.add_argument("--device", help="torch device override (default: cuda if available)")
    p.add_argument(
        "--anchor-value",
        "--pawn-value",
        type=float,
        default=DEFAULT_ANCHOR_VALUE,
        help="cp value assigned to the anchor piece",
    )
    p.add_argument(
        "--anchor-piece",
        choices=[p.lower() for p in PIECES],
        default="pawn",
        help="piece the scale is pinned to (default: pawn)",
    )
    p.add_argument("--anchor-bucket", type=int, help="pin the scale to one bucket (default: pool all)")
    p.add_argument("--clip", type=float, default=600.0, help="drop |eval| above this, 0 to keep all")
    p.add_argument("--ridge", type=float, default=0.0, help="ridge penalty on piece coefficients")
    p.add_argument("--subtract-pst", action="store_true", help="remove engine piece-square values before fitting")
    p.add_argument("--batch", type=int, default=8192, help="eval batch size")
    p.add_argument("--chunk", type=int, default=65536, help="H5 read chunk size")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("--verify", type=int, default=0, help="cross-check N samples against the engine backend")
    args = p.parse_args()

    if args.anchor_bucket is not None and not 0 <= args.anchor_bucket < NUM_BUCKETS:
        sys.exit(f"--anchor-bucket must be 0..{NUM_BUCKETS - 1}")

    backend = make_backend(args)
    print(f"Sampling {args.count:,} from {args.h5}, backend '{backend.name}' ...", flush=True)

    pst_mats = None
    if args.subtract_pst:
        tables = parse_square_tables(os.path.join(_HERE, "..", "..", "tables.h"))
        pst_mats = {False: build_pst_matrix(tables, False), True: build_pst_matrix(tables, True)}
        print("Subtracting engine piece-square values from each eval.")

    acc = Accumulator(len(PIECES))
    reservoir = Reservoir(args.verify, args.seed + 1) if args.verify else None
    kept = seen = 0

    for chunk in sample_rows(args.h5, args.count, args.seed, args.chunk):
        for start in range(0, len(chunk), args.batch):
            packed = chunk[start : start + args.batch]
            evals = backend.eval_batch(packed)
            seen += len(packed)

            if pst_mats is not None:
                evals = evals - pst_scores(packed, pst_mats)

            if args.clip:
                # Saturated evals track "winning" more than material and would bias the fit.
                keep = np.abs(evals) <= args.clip
                packed, evals = packed[keep], evals[keep]
            if not len(evals):
                continue

            kept += len(evals)
            acc.add(compute_buckets(packed), piece_features(packed), evals)
            if reservoir is not None:
                reservoir.add(packed, evals)

    if args.clip:
        print(f"Clipping |eval| > {args.clip:g}: kept {kept:,}/{seen:,}")

    if reservoir is not None:
        verify(reservoir, backend)

    raw, stats = {}, {}
    counts = {int(b): int(acc.counts[b]) for b in range(NUM_BUCKETS)}
    for b in range(NUM_BUCKETS):
        fit = fit_bucket(acc, b, args.ridge)
        if fit is not None:
            raw[b], stats[b] = fit[0], fit[1:]

    if not raw:
        sys.exit("No bucket produced a usable fit.")

    # One global scale. A single bucket's value is the noisiest number in the fit, so pool
    # them all (sample-weighted) unless the user pins a specific bucket.
    piece = args.anchor_piece.upper()
    anchor = args.anchor_bucket
    if anchor is None:
        weight = sum(counts[b] for b in raw)
        measured = sum(raw[b][piece] * counts[b] for b in raw) / weight
        label = "pooled"
    else:
        if anchor not in raw:
            sys.exit(f"Anchor bucket {anchor} has no usable fit; pick another with --anchor-bucket.")
        measured = raw[anchor][piece]
        label = f"bucket {anchor}"

    if measured <= 0:
        sys.exit(f"Anchor ({label} {piece.lower()}) fitted a non-positive value; try another anchor.")

    scale = args.anchor_value / measured
    fits = {b: {p: v * scale for p, v in vals.items()} for b, vals in raw.items()}

    print()
    print(f"Anchor: {label} {piece.lower()} = {args.anchor_value:g} (scale {scale:.4g})")
    head = f"{'bucket':>6} {'n':>9} " + " ".join(f"{p[:5]:>7}" for p in PIECES)
    print(head + (f" {'r2':>7} {'rmse':>7}" if stats else ""))
    for b in range(NUM_BUCKETS):
        if b not in fits:
            print(f"{b:>6} {counts[b]:>9}   (insufficient or degenerate data)")
            continue
        cells = " ".join(f"{fits[b][p]:>7.1f}" for p in PIECES)
        extra = f" {stats[b][0]:>7.3f} {stats[b][1] * scale:>7.1f}" if b in stats else ""
        print(f"{b:>6} {counts[b]:>9} {cells}{extra}")

    report(fits, counts, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
