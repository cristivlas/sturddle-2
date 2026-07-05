#!/usr/bin/env python3
"""
Per-bucket report for the pawn x king-file bucketing (16 buckets).

Histogram of training samples per bucket, and optionally per-bucket eval error
for one or more trained models. Buckets match the engine (nnue.h get_bucket)
and the trainer (train-x.py compute_bucket_id).

Usage:
    python bucket_report.py data.h5                       # histogram only
    python bucket_report.py data.h5 -m models/AWD models/DDAY --sample 0.05
"""

import argparse
import queue
import threading

import h5py
import numpy as np

SCALE = 100.0
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


def load_models(paths):
    import tensorflow as tf

    custom_objects = {
        "combined_loss": None,
        "scaled_sparse_categorical_crossentropy": None,
        "top": None,
        "top_3": None,
        "top_5": None,
    }
    fns = []
    for p in paths:
        try:
            model = tf.keras.models.load_model(p, custom_objects=custom_objects, compile=False)
            # compile the forward pass once; fixed batch shape means a single trace
            fns.append(tf.function(lambda x, m=model: m(x, training=False)))
        except Exception as e:
            # Lambda layers referencing trainer globals (e.g. POOL_SIZE) break keras
            # deserialization; the raw SavedModel graph needs no Python
            print(f"{p}: {e.__class__.__name__} from keras loader, using raw SavedModel graph")
            loaded = tf.saved_model.load(p)
            fns.append(lambda x, m=loaded: m(x, training=False))
    return fns


def batch_reader(args, plan, out_queue):
    """Read batches in a dedicated thread so h5 I/O overlaps with inference."""
    for path, indices in plan:
        with h5py.File(path, "r") as hf:
            data = hf["data"]
            for i in indices:
                start = i * args.batch_size
                out_queue.put(data[start : start + args.batch_size, : FEATURE_COUNT + 2])  # features + eval + outcome
    out_queue.put(None)


def main(args):
    predict = load_models(args.model) if args.model else []

    plan = []
    for path in args.input:
        with h5py.File(path, "r") as hf:
            rows = len(hf["data"])
        num_batches = max(1, rows // args.batch_size)  # small files: one partial batch
        indices = np.arange(num_batches)
        if args.sample:
            k = max(1, int(num_batches * args.sample))
            indices = np.random.choice(num_batches, k, replace=False)
            indices.sort()  # sequential reads are faster in h5
        print(f"{path}: {rows:,} rows, using {len(indices)} of {num_batches} batches of {args.batch_size}")
        plan.append((path, indices))
    total_batches = sum(len(indices) for _, indices in plan)

    counts = np.zeros(16, dtype=np.int64)
    draws = np.zeros(16, dtype=np.int64)
    white_wins = np.zeros(16, dtype=np.int64)
    zeros = np.zeros(16, dtype=np.int64)
    mids = np.zeros(16, dtype=np.int64)
    extreme = np.zeros(16, dtype=np.int64)
    abs_cp = np.zeros(16)
    err_sum = np.zeros((len(predict), 16))

    batches = queue.Queue(maxsize=8)
    reader = threading.Thread(target=batch_reader, args=(args, plan, batches), daemon=True)
    reader.start()

    n = 0
    while (block := batches.get()) is not None:
        x = block[:, :FEATURE_COUNT]
        buckets = bucket_ids(x)
        counts += np.bincount(buckets, minlength=16)

        white_to_move = x[:, -1] == 1
        cp = block[:, FEATURE_COUNT].astype(np.int64)
        cp = np.where(white_to_move, cp, -cp)  # white POV
        outcome = block[:, FEATURE_COUNT + 1].astype(np.int64)  # STM POV: 0=loss, 1=draw, 2=win

        draws += np.bincount(buckets, weights=(outcome == 1), minlength=16).astype(np.int64)
        wwin = np.where(white_to_move, outcome == 2, outcome == 0)
        white_wins += np.bincount(buckets, weights=wwin, minlength=16).astype(np.int64)
        zeros += np.bincount(buckets, weights=(cp == 0), minlength=16).astype(np.int64)
        abs_c = np.abs(cp)
        mids += np.bincount(buckets, weights=(abs_c > args.mid[0]) & (abs_c <= args.mid[1]), minlength=16).astype(
            np.int64
        )
        extreme += np.bincount(buckets, weights=(abs_c > args.extreme), minlength=16).astype(np.int64)
        abs_cp += np.bincount(buckets, weights=np.abs(cp), minlength=16)

        if predict:
            y_eval = cp.astype(np.float32) / SCALE  # white POV, pawns
            wdl_target = 1.0 / (1.0 + np.exp(-y_eval * SCALE / args.outcome_scale))

            for m, fn in enumerate(predict):
                pred = fn(x)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred = np.squeeze(pred.numpy(), axis=-1)
                wdl_pred = 1.0 / (1.0 + np.exp(-pred * SCALE / args.outcome_scale))
                err = np.abs(wdl_pred - wdl_target)
                err_sum[m] += np.bincount(buckets, weights=err, minlength=16)

        n += 1
        if n % 10 == 0 or n == total_batches:
            print(f"\r{n}/{total_batches} batches", end="", flush=True)
    print()

    total = counts.sum()
    names = [f"wdl_mae {p}" for p in (args.model or [])]
    header = (
        f"{'bucket':>6} {'pawns':>7} {'kings':>5} {'count':>12} {'%':>6}"
        f" {'draw%':>6} {'wwin%':>6} {'zero%':>6} {'mid%':>6} {'ext%':>6} {'|cp|':>6}"
        + "".join(f" {n:>24}" for n in names)
    )
    print(header)

    def stats(count, d, w, z, mi, e, a):
        return (
            f" {100.0 * d / count:>6.2f} {100.0 * w / count:>6.2f} {100.0 * z / count:>6.2f}"
            f" {100.0 * mi / count:>6.2f} {100.0 * e / count:>6.2f} {a / count:>6.0f}"
        )

    for b in range(16):
        pawn_id, king_id = divmod(b, 4)
        pawns = "0-4" if pawn_id == 0 else f"{pawn_id * 4 + 1}-{pawn_id * 4 + 4}"
        row = f"{b:>6} {pawns:>7} {KING_LABELS[king_id]:>5} {counts[b]:>12,} {100.0 * counts[b] / total:>6.2f}"
        if counts[b]:
            row += stats(counts[b], draws[b], white_wins[b], zeros[b], mids[b], extreme[b], abs_cp[b])
        for m in range(len(predict)):
            mae = err_sum[m][b] / counts[b] if counts[b] else float("nan")
            row += f" {mae:>24.5f}"
        print(row)

    row = f"{'overall':>33} {total:>12,} {100.0:>6.2f}"
    row += stats(total, draws.sum(), white_wins.sum(), zeros.sum(), mids.sum(), extreme.sum(), abs_cp.sum())
    row += "".join(f" {err_sum[m].sum() / total:>24.5f}" for m in range(len(predict)))
    print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", nargs="+", help="h5 data file(s); stats are aggregated across all")
    parser.add_argument("-m", "--model", nargs="*", help="model checkpoint path(s) for per-bucket eval error")
    parser.add_argument("-b", "--batch-size", type=int, default=16384)
    parser.add_argument("--sample", type=float, help="sampling ratio, same as the trainers")
    parser.add_argument("--outcome-scale", type=float, default=400.0)
    parser.add_argument("--extreme", type=int, default=1260, help="centipawn threshold for the ext%% column")
    parser.add_argument(
        "--mid", type=int, nargs=2, default=[50, 500], help="centipawn range for the mid%% (graded eval) column"
    )
    args = parser.parse_args()

    if args.sample:
        args.sample = max(1e-3, min(1.0, args.sample))

    main(args)
