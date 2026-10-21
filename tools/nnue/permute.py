#!/usr/bin/env python3
import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_torch as tt

HALF = tt.ACCUMULATOR_SIZE // 2


def load_rows(path, n, chunks=64):
    with h5py.File(path, "r") as hf:
        data = hf["data"]
        if data.shape[0] <= n:
            return data[:, :13].astype(np.int64)
        size = n // chunks
        starts = np.linspace(0, data.shape[0] - size, chunks).astype(np.int64)
        return np.concatenate([data[s : s + size, :13] for s in starts]).astype(np.int64)


@torch.no_grad()
def zero_products(model, rows, batch_size=4096):
    zeros = []
    for i in range(0, len(rows), batch_size):
        acc_w, acc_b = model.accumulator(torch.from_numpy(rows[i : i + batch_size]), quant=True)
        for acc in (acc_w, acc_b):
            zeros.append((tt.fake_floor(tt.pairwise(acc), tt.Q_ACT) == 0).numpy())
    return np.concatenate(zeros)


def co_zero(z, chunk=16384):
    counts = np.zeros((z.shape[1], z.shape[1]))
    for i in range(0, len(z), chunk):
        f = z[i : i + chunk].astype(np.float32)
        counts += f.T @ f
    return counts


def greedy_pairs(z):
    counts = co_zero(z)
    i, j = np.triu_indices(len(counts), k=1)
    free = np.ones(len(counts), dtype=bool)
    pairs = []
    for k in np.argsort(-counts[i, j], kind="stable"):
        a, b = i[k], j[k]
        if free[a] and free[b]:
            free[a] = free[b] = False
            pairs.append((a, b))
    return pairs


def zero_blocks(z):
    return z.reshape(len(z), -1, 4).all(axis=2).mean()


@torch.no_grad()
def permute(model, perm):
    cols = torch.from_numpy(np.concatenate([perm, perm + HALF]))
    w, b = model.hidden_1a.weight, model.hidden_1a.bias
    w.copy_(w[:, cols])
    b.copy_(b[cols])
    w2 = model.hidden_2.weight
    w2.copy_(w2[:, cols, :])


def main():
    p = argparse.ArgumentParser(
        description="Reorder the accumulator neurons so that zero activations share 4-byte blocks"
    )
    p.add_argument("input", help="weights.bin")
    p.add_argument("data", help="h5 dataset to sample positions from")
    p.add_argument("output", help="reordered weights.bin")
    p.add_argument("-n", "--rows", type=int, default=65536)
    args = p.parse_args()

    model = tt.NNUE()
    tt.load_bin(model, args.input)
    model.eval()

    rows = load_rows(args.data, args.rows)
    z = zero_products(model, rows)

    pairs = greedy_pairs(z)
    quads = greedy_pairs(np.stack([z[:, a] & z[:, b] for a, b in pairs], axis=1))
    perm = np.array([k for p, q in quads for k in (*pairs[p], *pairs[q])])
    assert sorted(perm) == list(range(HALF))
    print(f"all-zero 4-byte blocks: {zero_blocks(z):.1%} before, {zero_blocks(z[:, perm]):.1%} after")

    sample = torch.from_numpy(rows[:4096])
    with torch.no_grad():
        ref = model(sample)
    permute(model, perm)
    with torch.no_grad():
        assert torch.equal(model(sample), ref), "eval changed"
    tt.save_bin(model, args.output)


if __name__ == "__main__":
    main()
