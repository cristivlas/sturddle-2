#!/usr/bin/env python3
"""Re-initialize dead hidden_2 / hidden_3 units of a .bin net, per stack.

A dead unit gets fresh incoming weights, a bias that makes it fire on about half
the sampled positions of its stack, and zero outgoing weights, so the net output
does not change. Dead accumulator neurons are only reported."""
import argparse
import math

import h5py
import numpy as np
import torch
from tqdm import tqdm

import train_torch as T

BATCH = 8192
MIN_SEEN = 300  # positions a stack needs before its units can be called dead


def batches(path, n, device, desc):
    with h5py.File(path, "r") as f:
        d = f["data"]
        if d.shape[0] < BATCH:
            raise SystemExit(f"{path}: {d.shape[0]} rows, need at least {BATCH}")
        starts = np.linspace(0, d.shape[0] - BATCH, max(n // BATCH, 1)).astype(int)
        for s in tqdm(starts, desc=desc):
            yield torch.from_numpy(d[s : s + BATCH].astype(np.int64)).to(device)


@torch.no_grad()
def layers(model, x):
    _, a = model(x, with_act=True)
    stack = T.stack_index(x)
    w2 = T.fake_quant(model.hidden_2.weight, T.Q_W2)
    b2 = T.fake_quant(model.hidden_2.bias, T.Q_B2)
    with T.exact_fp32(x.device, True):
        pre2 = model.hidden_2(a, stack, w2, b2)
    pre3 = model.hidden_3(torch.relu(pre2), stack)
    out = model.out(torch.relu(pre3), stack)
    return stack, pre2, pre3, out


@torch.no_grad()
def provably_dead(model):
    """Units whose largest possible pre-activation is <= 0, from input bounds."""
    w2 = T.fake_quant(model.hidden_2.weight, T.Q_W2)
    b2 = T.fake_quant(model.hidden_2.bias, T.Q_B2)
    ub2 = b2 + w2.clamp(min=0).sum(1) * T.ACT_MAX
    w3, b3 = model.hidden_3.weight, model.hidden_3.bias
    ub3 = b3 + (w3.clamp(min=0) * ub2.clamp(min=0)[:, :, None]).sum(1)
    return ub2 <= 0, ub3 <= 0


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    model = T.NNUE()
    with torch.no_grad():
        T.load_bin(model, args.net)
    model.to(device).eval()

    acc = {}
    model.hidden_1a.register_forward_hook(lambda m, i, o: acc.update(out=o))

    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.long, device=device)

    rows = 0
    fired_acc, full_acc = zeros(T.ACCUMULATOR_SIZE), zeros(T.ACCUMULATOR_SIZE)
    seen = zeros(T.STACKS)
    fired2, fired3 = zeros(T.STACKS, T.HIDDEN_2), zeros(T.STACKS, T.HIDDEN_3)
    probe = None
    for x in batches(args.data, args.positions, device, "scan"):
        probe = x if probe is None else probe
        stack, pre2, pre3, _ = layers(model, x)
        rows += acc["out"].shape[0]
        fired_acc += (acc["out"] > 0).sum(0)
        full_acc += (acc["out"] >= T.ACT_CLAMP).sum(0)
        seen += torch.bincount(stack, minlength=T.STACKS)
        fired2.index_add_(0, stack, (pre2 > 0).long())
        fired3.index_add_(0, stack, (pre3 > 0).long())

    judged = seen >= MIN_SEEN
    stacks = judged.nonzero().flatten().tolist()
    dead2 = judged[:, None] & (fired2 == 0)
    dead3 = judged[:, None] & (fired3 == 0)
    proven2, proven3 = provably_dead(model)
    print(f"positions {seen.sum().item()}, stacks judged {len(stacks)}/{T.STACKS}")
    print(f"accumulator: dead {(fired_acc == 0).sum().item()}, always saturated {(full_acc == rows).sum().item()}")
    print(f"dead hidden_2 {dead2.sum().item()} (provably {proven2.sum().item()})")
    print(f"dead hidden_3 {dead3.sum().item()} (provably {proven3.sum().item()})")
    for s in stacks:
        if dead2[s].any() or dead3[s].any():
            units2 = dead2[s].nonzero().flatten().tolist()
            units3 = dead3[s].nonzero().flatten().tolist()
            print(f"  stack {s} ({seen[s].item()} positions): hidden_2 {units2}, hidden_3 {units3}")
    if not (dead2.any() or dead3.any()):
        return

    out_before = layers(model, probe)[3]

    with torch.no_grad():
        for s, k in dead3.nonzero().tolist():
            model.hidden_3.weight[s, :, k].normal_(std=math.sqrt(2.0 / T.HIDDEN_2))
            model.hidden_3.bias[s, k] = 0.0
            model.out.weight[s, k, 0] = 0.0
        for s, j in dead2.nonzero().tolist():
            model.hidden_2.weight[s, :, j].normal_(std=math.sqrt(2.0 / T.L1_INPUTS))
            model.hidden_2.bias[s, j] = 0.0
            model.hidden_3.weight[s, j, :] = 0.0
        T.apply_constraints(model)

    pre2s = [[] for _ in range(T.STACKS)]
    pre3s = [[] for _ in range(T.STACKS)]
    for x in batches(args.data, args.positions, device, "bias"):
        stack, pre2, pre3, _ = layers(model, x)
        for s in stacks:
            m = stack == s
            pre2s[s].append(pre2[m][:, dead2[s]])
            pre3s[s].append(pre3[m][:, dead3[s]])

    with torch.no_grad():
        for s in stacks:
            if dead2[s].any():
                model.hidden_2.bias[s, dead2[s]] = -torch.cat(pre2s[s]).median(0).values
            if dead3[s].any():
                model.hidden_3.bias[s, dead3[s]] = -torch.cat(pre3s[s]).median(0).values

    out_after = layers(model, probe)[3]
    print(f"output change on {len(probe)} positions: {(out_after - out_before).abs().max().item():.3g}")
    T.save_bin(model, args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("net", help="input .bin")
    p.add_argument("data", help="h5 dataset")
    p.add_argument("-n", "--positions", type=int, required=True, help="positions to sample")
    p.add_argument("-o", "--output", required=True, help="output .bin")
    main(p.parse_args())
