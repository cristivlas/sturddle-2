#!/usr/bin/env python3
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


def check_buckets():
    # white king, black king, white queens, black queens -> (block, mirrored) for the white view, then the black view
    cases = [
        (0, 56, (), (), (0, False), (0, False)),  # a1, a8
        (4, 60, (), (), (6, True), (6, True)),  # e1, e8
        (6, 57, (), (), (2, True), (2, False)),  # g1, b8
        (27, 7, (), (), (22, False), (28, True)),  # d4, h1
        (53, 34, (), (), (30, True), (20, False)),  # f7, c5
        (0, 56, (), (59,), (1, False), (0, False)),  # a1, a8, Qd8
        (4, 60, (3,), (), (6, True), (7, True)),  # e1, e8, Qd1
        (27, 7, (3,), (59,), (23, False), (29, True)),  # d4, h1, Qd1, Qd8
        (6, 57, (), (45, 18), (3, True), (2, False)),  # g1, b8, Qf6, Qc3
        (53, 34, (47, 30), (), (30, True), (21, False)),  # f7, c5, Qh6, Qg4
    ]
    x = torch.zeros(len(cases), 13, dtype=torch.int64)
    for i, (wk, bk, wq, bq, _, _) in enumerate(cases):
        x[i, 1], x[i, 0] = 1 << wk, 1 << bk
        x[i, 11], x[i, 10] = sum(1 << sq for sq in wq), sum(1 << sq for sq in bq)
    white = tt.unpack_bits(x)
    for view, col in ((white, 4), (tt.black_view(white), 5)):
        block, mirrored = tt.king_block(view)
        expected = [c[col] for c in cases]
        assert [(int(b), bool(m)) for b, m in zip(block, mirrored)] == expected, "buckets: wrong king block"
    print("buckets: ok")


def horizontal_mirror(bb):
    b = bb.astype(np.uint64)
    for shift, mask in ((1, 0x5555555555555555), (2, 0x3333333333333333), (4, 0x0F0F0F0F0F0F0F0F)):
        s, m = np.uint64(shift), np.uint64(mask)
        b = ((b >> s) & m) | ((b & m) << s)
    return b


def check_mirror(model, x):
    w, b = model.accumulator(x, quant=True)
    m = x.numpy().copy()
    m[:, :12] = horizontal_mirror(m[:, :12]).astype(m.dtype)
    mw, mb = model.accumulator(torch.from_numpy(m), quant=True)
    assert torch.equal(mw, w) and torch.equal(mb, b), "mirror: accumulator halves differ"
    print("mirror: ok")


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
    p = argparse.ArgumentParser(
        description="CPU self-checks for train_torch: king buckets, file mirror, perspective flip, autocast precision, export round trip."
    )
    p.add_argument("input", help="h5 dataset path")
    p.add_argument("-n", "--rows", type=int, default=4096)
    args = p.parse_args()

    torch.manual_seed(0)
    with h5py.File(args.input, "r") as hf:
        x = torch.from_numpy(hf["data"][: args.rows, :13].astype(np.int64))

    model = tt.NNUE(qat=True)
    check_buckets()
    check_mirror(model, x)
    check_flip(model, x)
    check_autocast(model, x)
    check_float_autocast(x)
    check_export(model, x)


if __name__ == "__main__":
    main()
