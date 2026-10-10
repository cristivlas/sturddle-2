#!/usr/bin/env python3
"""
CPU smoke checks for train_torch.py:
  - flip: for flip_position(x), the accumulator halves equal the original's, swapped, bit for bit
  - buckets: black-view bucket id == white-view id with the king bits swapped
  - autocast: with QAT, accumulator / activation / hidden_2 stay fp32 under CPU bf16 autocast;
    a float run keeps mixed precision there
  - export: on-grid weights.bin round-trips to identical eval() outputs

Usage:
    python tools/nnue/smoke_torch.py data.h5 [-n ROWS]
"""

import argparse
import os
import sys
import tempfile

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_torch as tt


def check_flip(model, x):
    w, b = model.accumulator(x, quant=True)
    fx = torch.from_numpy(tt.flip_position(x.numpy()))
    fw, fb = model.accumulator(fx, quant=True)
    assert torch.equal(fw, b) and torch.equal(fb, w), "flip: accumulator halves not swapped exactly"
    print("flip: ok")


def check_buckets(x):
    white = tt.unpack_bits(x)
    ids = tt.compute_bucket_id(white)
    swapped = (ids & ~3) | ((ids & 1) << 1) | ((ids >> 1) & 1)
    assert torch.equal(tt.compute_bucket_id(tt.black_view(white)), swapped), "buckets: king bits not swapped"
    assert torch.equal(ids, torch.from_numpy(tt.scale_buckets(x.numpy()))), "buckets: differ from scale_buckets"
    print("buckets: ok")


def _autocast_on():
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:  # older torch
        return torch.is_autocast_cpu_enabled()


def check_autocast(model, x):
    seen = {}

    def hook(name):
        def f(module, inputs, output):
            seen[name] = (inputs, output, _autocast_on())

        return f

    handles = [getattr(model, n).register_forward_hook(hook(n)) for n in ("hidden_1a", "hidden_2", "hidden_3")]
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pred = model(x)
    for h in handles:
        h.remove()

    for name in ("hidden_1a", "hidden_2"):
        assert not seen[name][2], f"autocast: enabled inside {name}"
        assert seen[name][1].dtype == torch.float32, f"autocast: {name} output is {seen[name][1].dtype}"
    assert seen["hidden_3"][2], "autocast: not enabled in the tail (check is vacuous)"

    a, _, w2, _ = seen["hidden_2"][0]
    assert a.dtype == torch.float32 and w2.dtype == torch.float32, "autocast: hidden_2 inputs downcast"
    if model.qat:
        assert torch.equal(a * tt.Q_ACT, torch.round(a * tt.Q_ACT)), "activation off the 1/128 grid"
        assert torch.equal(w2 * tt.Q_W2, torch.round(w2 * tt.Q_W2)), "hidden_2 weights off the 1/Q_W2 grid"
        assert a.max().item() <= tt.ACT_MAX, "activation above 127/128"

    pred.float().sum().backward()
    assert model.hidden_1a.weight.grad is not None and model.hidden_2.weight.grad is not None, "no gradient"
    model.zero_grad()
    print("autocast: ok")


def check_float_autocast(x):
    model = tt.NNUE(qat=False)
    seen = {}
    handle = model.hidden_1a.register_forward_hook(lambda m, i, o: seen.update(on=_autocast_on()))
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pred = model(x)
    handle.remove()
    assert seen["on"], "float run: autocast disabled in hidden_1a"
    pred.float().sum().backward()
    print("float autocast: ok")


def check_export(model, x):
    model.eval()
    with torch.no_grad():
        ref = model(x)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.bin")
        tt.save_bin(model, path, quantize_round=True)
        other = tt.NNUE()
        tt.load_bin(other, path)
    other.eval()
    with torch.no_grad():
        out = other(x)
    assert torch.equal(ref, out), f"export: max diff {(ref - out).abs().max().item()}"
    print("export: ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="h5 dataset path")
    p.add_argument("-n", "--rows", type=int, default=4096)
    args = p.parse_args()

    torch.manual_seed(0)
    with h5py.File(args.input, "r") as hf:
        x = torch.from_numpy(hf["data"][: args.rows, :13].astype(np.int64))

    model = tt.NNUE(qat=True)
    check_buckets(x)
    check_flip(model, x)
    check_autocast(model, x)
    check_float_autocast(x)
    check_export(model, x)


if __name__ == "__main__":
    main()
