#!/usr/bin/env python3
"""
PyTorch trainer for the Sturddle Chess engine's NNUE.
Copyright (c) 2023 - 2026 Cristian Vlasceanu.

Port of train.py (TF) to PyTorch. Primary optimizer is SGD+momentum.
Loads / exports the same flat float32 weights.bin layout as the TF trainer so a
TF-trained net can be continued here and the result consumed by the C++ engine.

Layer export order (must match C++ context.cpp load order):
    hidden_1a, hidden_1b, hidden_2, hidden_3, out
Each layer: kernel (in, out) float32 row-major, then bias (out,) float32.
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- architecture constants (keep in sync with nnue.h / context.cpp) ----
ACTIVE_INPUTS = 769
ACCUMULATOR_SIZE = 2048
POOL_SIZE = 8
POOLED = ACCUMULATOR_SIZE // POOL_SIZE  # 256
MAIN_BUCKETS = 16  # 4 pawn x 4 king-file
INPUTS_B = 256  # kings + pawns
HIDDEN_2 = 16
HIDDEN_3 = 16

Q_SCALE = 1024
# 32 pieces + side-to-move + bias == 34
Q_MAX_A = 32767 / Q_SCALE / 34
# (8 pawns + 1 king) x 2 + bias == 19
Q_MAX_B = 32767 / Q_SCALE / 19

SCALE = 100.0


# ---------------------------------------------------------------------------
# Feature unpacking and bucketing (mirror tools/nnue/train.py exactly)
# ---------------------------------------------------------------------------
def unpack_bits(packed):
    """packed: (B, 13) uint64 [12 bitboards + turn] -> (B, 769) float32 features.

    Feature index idx within a 64-block holds bitboard bit (63 - idx)."""
    bitboards = packed[:, :12]  # (B, 12) int64
    turn = packed[:, 12:13]  # (B, 1)
    shifts = torch.arange(63, -1, -1, device=packed.device, dtype=torch.int64)
    bits = (bitboards.unsqueeze(-1) >> shifts) & 1  # (B, 12, 64)
    feats = bits.reshape(bits.shape[0], 12 * 64)  # (B, 768)
    return torch.cat([feats, turn], dim=1).float()  # (B, 769)


# right half = files e-h: feature idx where (63 - idx) % 8 >= 4
_RIGHT_MASK = torch.tensor([1.0 if ((63 - idx) % 8) >= 4 else 0.0 for idx in range(64)], dtype=torch.float32)


def compute_bucket_id(features):
    """features: (B, 769) -> (B,) bucket id = pawn_id * 4 + king_id."""
    right_mask = _RIGHT_MASK.to(features.device)
    # black king [0:64], white king [64:128], black pawns [128:192], white pawns [192:256]
    pawn_count = features[:, 128:256].sum(dim=1)
    pawn_id = torch.where(
        pawn_count <= 4.0,
        torch.zeros_like(pawn_count),
        torch.clamp(torch.floor((pawn_count - 1.0) / 4.0), max=3.0),
    )
    pawn_id = pawn_id.long()

    wk_right = (features[:, 64:128] * right_mask).sum(dim=1).long()
    bk_right = (features[:, 0:64] * right_mask).sum(dim=1).long()
    king_id = wk_right * 2 + bk_right
    return pawn_id * 4 + king_id


class BucketedDense(nn.Module):
    """hidden_1a: per-row bucket selects a (ACTIVE_INPUTS, units) weight block.

    Kernel stored as one (num_buckets * ACTIVE_INPUTS, units) tensor, bucket-major,
    matching the C++ inference convention base = bucket * ACTIVE_INPUTS and the flat
    weights.bin layout."""

    def __init__(self, num_buckets, in_features, units):
        super().__init__()
        self.num_buckets = num_buckets
        self.in_features = in_features
        self.units = units
        # weight laid out (num_buckets * in_features, units) to match .bin (in, out)
        self.weight = nn.Parameter(torch.empty(num_buckets * in_features, units))
        self.bias = nn.Parameter(torch.zeros(units))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

    def forward(self, features):
        bucket_id = compute_bucket_id(features)  # (B,)
        blocks = self.weight.view(self.num_buckets, self.in_features, self.units)
        out = None
        for b in range(self.num_buckets):
            rows = (bucket_id == b).nonzero(as_tuple=True)[0]
            if rows.numel():
                r = features[rows] @ blocks[b]
                if out is None:  # match matmul dtype (AMP -> Half)
                    out = features.new_empty(features.shape[0], self.units, dtype=r.dtype)
                out[rows] = r
        if out is None:
            out = features.new_zeros(features.shape[0], self.units)
        out = out + self.bias
        return F.relu(out)


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_1a = BucketedDense(MAIN_BUCKETS, ACTIVE_INPUTS, ACCUMULATOR_SIZE)
        self.hidden_1b = nn.Linear(INPUTS_B, POOLED)  # linear (no activation)
        self.hidden_2 = nn.Linear(POOLED, HIDDEN_2)
        self.hidden_3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.out = nn.Linear(HIDDEN_3, 1)
        for m in (self.hidden_1b, self.hidden_2, self.hidden_3):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, packed):
        feats = unpack_bits(packed)  # (B, 769)
        acc = self.hidden_1a(feats)  # (B, 2048) relu'd
        kp = feats[:, :INPUTS_B]
        mod = self.hidden_1b(kp)  # (B, 256) linear

        pooled = acc.view(acc.shape[0], POOLED, POOL_SIZE).mean(dim=-1)  # (B, 256)
        residual = pooled + pooled * mod  # pooled * (1 + mod)

        x = F.relu(self.hidden_2(residual))
        x = F.relu(self.hidden_3(x))
        return self.out(x)  # (B, 1)


# ---------------------------------------------------------------------------
# Quantization constraint: round to 1/Q_SCALE and clamp. Applied in-place after
# each optimizer step (PyTorch has no kernel/bias constraint hook).
# ---------------------------------------------------------------------------
def _core(model):
    """Unwrap nn.DataParallel to reach the real NNUE layers."""
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def apply_constraints(model, quantize_round):
    model = _core(model)

    def clamp(p, qmax):
        if quantize_round:
            p.copy_(torch.round(p * Q_SCALE) / Q_SCALE)
        p.clamp_(-qmax, qmax)

    clamp(model.hidden_1a.weight, Q_MAX_A)
    clamp(model.hidden_1a.bias, Q_MAX_A)
    clamp(model.hidden_1b.weight, Q_MAX_B)
    clamp(model.hidden_1b.bias, Q_MAX_B)
    # hidden_2 / hidden_3 / out are unconstrained (float in C++)


# ---------------------------------------------------------------------------
# Flat weights.bin load / save (C++-compatible: kernel (in, out) then bias)
# ---------------------------------------------------------------------------
# (name, in, out); BucketedDense uses num_buckets*in as the stored row count.
_EXPORT = [
    ("hidden_1a", MAIN_BUCKETS * ACTIVE_INPUTS, ACCUMULATOR_SIZE),
    ("hidden_1b", INPUTS_B, POOLED),
    ("hidden_2", POOLED, HIDDEN_2),
    ("hidden_3", HIDDEN_2, HIDDEN_3),
    ("out", HIDDEN_3, 1),
]


def _layer_kernel_bias(model, name):
    """Return (kernel (in,out) np.float32, bias (out,) np.float32) for export."""
    model = _core(model)
    if name == "hidden_1a":
        k = model.hidden_1a.weight.detach().cpu().numpy()  # already (in, out)
        b = model.hidden_1a.bias.detach().cpu().numpy()
    else:
        m = getattr(model, name)
        k = m.weight.detach().cpu().numpy().T  # (out,in) -> (in,out)
        b = m.bias.detach().cpu().numpy()
    return k.astype(np.float32), b.astype(np.float32)


@torch.no_grad()
def save_bin(model, path, quantize_round=False):
    def q(a, qmax):
        if quantize_round:
            a = np.round(a * Q_SCALE) / Q_SCALE
            a = np.clip(a, -qmax, qmax)
        return a.astype(np.float32)

    qmax = {"hidden_1a": Q_MAX_A, "hidden_1b": Q_MAX_B}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        for name, _, _ in _EXPORT:
            k, b = _layer_kernel_bias(model, name)
            m = qmax.get(name)
            if m is not None:
                k, b = q(k, m), q(b, m)
            k.tofile(f)
            b.tofile(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic
    print(f"Wrote weights to {path}")


@torch.no_grad()
def load_bin(model, path):
    model = _core(model)
    data = np.fromfile(path, dtype=np.float32)
    expected = sum(i * o + o for _, i, o in _EXPORT)
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} floats, got {data.size}")
    off = 0
    for name, i, o in _EXPORT:
        k = data[off : off + i * o].reshape(i, o)
        off += i * o
        b = data[off : off + o]
        off += o
        if name == "hidden_1a":
            model.hidden_1a.weight.copy_(torch.from_numpy(k.copy()))
            model.hidden_1a.bias.copy_(torch.from_numpy(b.copy()))
        else:
            m = getattr(model, name)
            m.weight.copy_(torch.from_numpy(k.T.copy()))  # (in,out)->(out,in)
            m.bias.copy_(torch.from_numpy(b.copy()))
    print(f"Loaded weights from {path}")


# ---------------------------------------------------------------------------
# Loss: combine eval (Huber on raw domain) with WDL outcome (BCE on prob domain)
# ---------------------------------------------------------------------------
def combined_loss(y_pred, y_true, args):
    eval_target = y_true[:, 0:1] / y_true[:, 3:4]  # normalize label scale per bucket
    outcome_target = y_true[:, 1:2]
    piece_ratio = y_true[:, 2:3]

    if args.clip_eval:
        cv = args.clip_eval / SCALE
        clipped = torch.clamp(eval_target, -cv, cv)
        eval_target = clipped + 0.1 * (eval_target - clipped)  # soft clip

    # BCE via logits: F.binary_cross_entropy is banned under autocast and inf-prone at sigmoid saturation
    logits = y_pred * SCALE / args.outcome_scale
    wdl_pred = torch.sigmoid(logits)
    wdl_target = torch.sigmoid(eval_target * SCALE / args.outcome_scale)

    if args.loss_mae:
        loss_eval = (wdl_pred - wdl_target).abs()
        loss_outcome = (wdl_pred - outcome_target).abs()
    elif args.loss_bce:
        loss_eval = F.binary_cross_entropy_with_logits(logits, wdl_target, reduction="none")
        loss_outcome = F.binary_cross_entropy_with_logits(logits, outcome_target, reduction="none")
    elif args.loss_blend:
        d = args.huber_delta
        err = y_pred - eval_target
        a = err.abs()
        loss_eval = torch.where(a <= d, 0.5 * err * err, d * (a - 0.5 * d))
        loss_outcome = F.binary_cross_entropy_with_logits(logits, outcome_target, reduction="none")
    else:
        loss_eval = (wdl_pred - wdl_target) ** 2
        loss_outcome = (wdl_pred - outcome_target) ** 2

    w = args.outcome_weight
    if args.dynamic_outcome_weight:
        w = w * piece_ratio
    loss = loss_eval * (1.0 - w) + loss_outcome * w
    return loss.mean()


# ---------------------------------------------------------------------------
# H5 dataset: yields whole batches (contiguous slices), shuffled / sampled.
# ---------------------------------------------------------------------------
class H5Batches(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        batch_size,
        sample=None,
        smoothing=0.025,
        filter=None,
        no_capture=False,
        discard_mismatch=0,
        balance=False,
        profile_ratios=None,
    ):
        self.path = path
        self.batch_size = batch_size
        self.sample = sample
        self.smoothing = smoothing
        self.filter = filter
        self.no_capture = no_capture
        self.discard_mismatch = discard_mismatch
        self.balance = balance
        print(f"Loading dataset {path}")
        with h5py.File(path, "r") as hf:
            n = hf["data"].shape[0]
            self.cols = hf["data"].shape[1]
            self.member_starts, self.member_profiles = member_profile_table(hf["data"], path, profile_ratios)
        self.trivial_profiles = bool(np.all(self.member_profiles == 1.0))
        print(f"{n:,} rows.")
        logging.info("dataset %s: %s rows", path, f"{n:,}")
        self.feature_count = 13
        self.num_batches = n // batch_size
        self.hf = None  # lazy per-worker
        self.reshuffle()

    def reshuffle(self):
        if self.sample:
            k = max(1, int(self.num_batches * self.sample))
            self.indices = np.random.choice(self.num_batches, k, replace=False)
        else:
            self.indices = np.random.permutation(self.num_batches)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.hf is None:  # open inside worker
            self.hf = h5py.File(self.path, "r")
        data = self.hf["data"]
        i = self.indices[idx]
        s, e = i * self.batch_size, (i + 1) * self.batch_size
        rows = data[s:e]  # (B, cols) uint64

        x = rows[:, : self.feature_count].astype(np.int64)
        wtm = x[:, 12] == 1

        # Member (source file) of each row, for per-member label-scale profiles
        member_ids = None
        if not self.trivial_profiles:
            member_ids = np.searchsorted(self.member_starts, np.arange(s, e), side="right") - 1

        y_eval = rows[:, self.feature_count].astype(np.int64).astype(np.float32) / SCALE
        y_eval = np.where(wtm, y_eval, -y_eval)

        y_out = rows[:, self.feature_count + 1].astype(np.float32) - 1.0  # 0,1,2 -> -1,0,1
        y_out = np.where(wtm, y_out, -y_out)
        y_out = (y_out + 1.0) / 2.0  # -> 0,0.5,1
        s_ = self.smoothing
        y_out = y_out * (1 - s_) + 0.5 * s_

        keep = None
        if self.filter:
            keep = np.abs(y_eval) < (self.filter / SCALE)

        if self.discard_mismatch:
            t = self.discard_mismatch / SCALE
            mismatch = ((y_eval > t) & (y_out < 0.25)) | ((y_eval < -t) & (y_out > 0.75))
            keep = ~mismatch if keep is None else (keep & ~mismatch)

        if self.no_capture and rows.shape[1] > self.feature_count + 3:
            to_square = rows[:, self.feature_count + 3]
            black_occ = np.bitwise_or.reduce(rows[:, 0:12:2], axis=1)
            white_occ = np.bitwise_or.reduce(rows[:, 1:12:2], axis=1)
            opp_occ = np.where(rows[:, 12] == 1, black_occ, white_occ)
            is_capture = (opp_occ & np.left_shift(np.uint64(1), to_square)) != 0
            keep = ~is_capture if keep is None else (keep & ~is_capture)

        if keep is not None:
            x, y_eval, y_out = x[keep], y_eval[keep], y_out[keep]
            if member_ids is not None:
                member_ids = member_ids[keep]

        # Label scale reflects how the source labeled the original position, so
        # compute before balance mirroring (which QK<->KQ swaps the bucket)
        if self.trivial_profiles:
            label_scale = np.ones(x.shape[0], dtype=np.float32)
        else:
            label_scale = self.member_profiles[member_ids, scale_buckets(x)]

        if self.balance:
            # Synthesize color-mirrored positions: swap colors + vertical mirror + flip STM.
            x = np.concatenate([x, flip_position(x)], axis=0)
            y_eval = np.concatenate([y_eval, -y_eval], axis=0)  # negate eval
            y_out = np.concatenate([y_out, 1.0 - y_out], axis=0)  # swap win/loss, keep draws
            label_scale = np.concatenate([label_scale, label_scale])

        pc = np.zeros(x.shape[0], dtype=np.float32)
        for c in range(12):  # first 12 columns of x are the piece bitboards
            pc += popcount(x[:, c])
        piece_ratio = pc / 32.0

        y = np.stack([y_eval, y_out, piece_ratio, label_scale], axis=1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


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


def flip_position(x):
    """Color-flip: swap piece colors, vertical mirror, flip STM. x: (B, 13) [12 bitboards + turn]."""
    flipped = np.empty_like(x)
    for i in range(6):  # bitboards are [black, white] per piece
        flipped[:, i * 2] = vertical_mirror(x[:, i * 2 + 1]).astype(x.dtype)
        flipped[:, i * 2 + 1] = vertical_mirror(x[:, i * 2]).astype(x.dtype)
    flipped[:, 12] = x[:, 12] ^ 1
    return flipped


if hasattr(np, "bitwise_count"):

    def popcount(bb):
        return np.bitwise_count(bb.astype(np.uint64)).astype(np.float32)

else:

    def popcount(bb):
        bb = bb.astype(np.uint64)
        bb = bb - ((bb >> np.uint64(1)) & np.uint64(0x5555555555555555))
        bb = (bb & np.uint64(0x3333333333333333)) + ((bb >> np.uint64(2)) & np.uint64(0x3333333333333333))
        bb = (bb + (bb >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return ((bb * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.float32)


# Squares on files e-h, for king-file bucketing
FILES_EFGH = np.uint64(0xF0F0F0F0F0F0F0F0)


def scale_buckets(x):
    """Per-sample pawn x king-file bucket ids from raw bitboards, same scheme as compute_bucket_id."""
    xu = x.astype(np.uint64)
    pawns = popcount(xu[:, 2]) + popcount(xu[:, 3])
    pawn_id = np.where(pawns <= 4, 0, np.minimum((pawns - 1) // 4, 3))
    wk_right = (xu[:, 1] & FILES_EFGH) != 0
    bk_right = (xu[:, 0] & FILES_EFGH) != 0
    return (pawn_id * 4 + wk_right * 2 + bk_right).astype(np.int64)


def load_profile(path):
    """Load per-bucket label scale ratios (dataset profile); eval labels are divided by these."""
    with open(path) as f:
        profile = json.load(f)
    ratios = profile["ratios"] if isinstance(profile, dict) else profile
    if not isinstance(ratios, list) or len(ratios) != MAIN_BUCKETS:
        raise ValueError(f"{path}: expected a list of {MAIN_BUCKETS} bucket ratios")
    ratios = np.array(ratios, dtype=np.float32)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError(f"{path}: ratios must be positive and finite")
    logging.info("profile %s: %s", path, " ".join(f"{r:.3f}" for r in ratios))
    return ratios


def member_profile_table(data, filepath, profile_ratios):
    """Per-member label-scale profiles: (member start rows, (members x buckets) ratio matrix).

    An explicit profile overrides everything. Otherwise each member of a virtual
    dataset (or the file itself if not virtual) uses the <member>.profile.json
    sidecar when present, falling back to the container's sidecar, then to
    all-ones. Resolves one level, not recursively.
    """
    if profile_ratios is not None:
        return np.zeros(1, dtype=np.int64), profile_ratios[np.newaxis, :]

    if data.is_virtual:
        # HDF5 resolves relative member paths against the container's directory
        base = os.path.dirname(os.path.abspath(filepath))
        members = sorted((vs.vspace.get_select_bounds()[0][0], vs.file_name) for vs in data.virtual_sources())
        members = [
            (start, filepath if path == "." else (path if os.path.isabs(path) else os.path.join(base, path)))
            for start, path in members
        ]
    else:
        members = [(0, filepath)]
    assert members[0][0] == 0, "first member must start at row 0"

    default = np.ones(MAIN_BUCKETS, dtype=np.float32)
    own_sidecar = filepath + ".profile.json"
    if os.path.exists(own_sidecar):
        default = load_profile(own_sidecar)
        print(f"{filepath}: default profile {own_sidecar}")

    cache = {}
    matched = 0
    profiles = []
    for start, path in members:
        sidecar = path + ".profile.json"
        if sidecar not in cache:
            if path != filepath and os.path.exists(sidecar):
                cache[sidecar] = load_profile(sidecar)
                print(f"{path}: using profile {sidecar}")
                matched += 1
            else:
                cache[sidecar] = default
        profiles.append(cache[sidecar])

    print(f"{len(members)} member(s), {matched} with own profile")
    return np.array([m[0] for m in members], dtype=np.int64), np.stack(profiles)


def summary(model):
    print(f'{"layer":<12}{"shape (in, out)":<22}{"params":>12}')
    print("-" * 46)
    total = 0
    for name, _, _ in _EXPORT:
        k, b = _layer_kernel_bias(model, name)
        n = k.size + b.size
        total += n
        print(f"{name:<12}{str(k.shape):<22}{n:>12,}")
    print("-" * 46)
    print(f'{"total":<34}{total:>12,}')


@torch.no_grad()
def metrics(pred, y, args):
    """Match TF: accuracy = 1 - MAE(sigmoid(cp/scale), outcome); mae = MAE(eval) * SCALE/100."""
    eval_t, out_t = y[:, 0:1] / y[:, 3:4], y[:, 1:2]
    probs = torch.sigmoid(pred * SCALE / args.outcome_scale)
    acc = 1.0 - (probs - out_t).abs().mean().item()
    mae = (pred - eval_t).abs().mean().item() * SCALE / 100
    return acc, mae


def main(args):
    logging.basicConfig(filename=args.logfile, level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"device: {device}")

    # Devices actually used: all visible under DataParallel, else just device 0.
    use_dp = device.type == "cuda" and not args.no_data_parallel and torch.cuda.device_count() > 1
    used_devices = list(range(torch.cuda.device_count())) if use_dp else ([0] if device.type == "cuda" else [])

    if args.mem_limit and used_devices:
        for i in used_devices:
            total = torch.cuda.get_device_properties(i).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_limit * 1024 * 1024 / total), device=i)

    model = NNUE().to(device)
    src = args.import_file or (args.model if args.model and os.path.exists(args.model) else None)
    if src:
        load_bin(model, src)

    if args.export:
        save_bin(model, args.export, quantize_round=args.quant_round)
        return

    # Multi-GPU: split each global batch across all visible GPUs (like TF MirroredStrategy).
    if use_dp:
        model = nn.DataParallel(model)
        print(f"DataParallel over {torch.cuda.device_count()} GPUs")

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=args.learn_rate,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.decay or 0.0,
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.learn_rate,
            betas=(0.99, 0.995),
            amsgrad=(args.optimizer == "amsgrad"),
            weight_decay=args.decay or 0.0,
        )

    sched = None
    if args.schedule:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.patience)

    use_amp = args.mixed_precision and device.type == "cuda"
    if use_amp:
        # fp16 is only accelerated on Tensor Cores (compute capability >= 7.0). Every GPU that runs
        # the forward must qualify, so gate on the devices actually in use (all under DP, else just 0).
        caps = [torch.cuda.get_device_capability(i) for i in used_devices]
        if min(c[0] for c in caps) < 7:
            use_amp = False
            print(f"mixed precision disabled (no Tensor Cores; compute capabilities {caps})")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)
    if use_amp:
        print("mixed precision enabled")

    profile_ratios = None
    if args.profile:
        profile_ratios = load_profile(args.profile)
        print(f"Loaded profile {args.profile}")

    ds = H5Batches(
        args.input,
        args.batch_size,
        sample=args.sample,
        smoothing=args.outcome_smoothing,
        filter=args.filter,
        no_capture=args.no_capture,
        discard_mismatch=args.discard_mismatch,
        balance=args.balance,
        profile_ratios=profile_ratios,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=args.workers, shuffle=False)

    summary(model)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
        print("*" * 64, file=sys.stderr)
        print("*  WARNING: tqdm is not installed -- no live progress bar.", file=sys.stderr)
        print(f"*  Falling back to a status line every {args.log_every}s.", file=sys.stderr)
        print("*  Install it for a proper bar:  pip install tqdm", file=sys.stderr)
        print("*" * 64, file=sys.stderr)

    best = math.inf
    bar = None
    try:
        for epoch in range(args.epochs):
            model.train()
            total, count = 0.0, 0
            acc_sum, mae_sum = 0.0, 0.0
            ds.reshuffle()
            bar = (
                tqdm(
                    loader,
                    desc=f"epoch {epoch+1}/{args.epochs}",
                    total=len(ds),
                    ncols=(args.ncols or None),
                )
                if tqdm
                else loader
            )
            t0 = t_last = time.time()
            for i, (x, y) in enumerate(bar):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with autocast():
                    pred = model(x)
                    loss = combined_loss(pred, y, args)
                scaler.scale(loss).backward()
                if args.clip_norm:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(opt)
                scaler.update()
                apply_constraints(model, args.quant_round)
                total += loss.item()
                count += 1
                acc, mae = metrics(pred, y, args)
                acc_sum += acc
                mae_sum += mae
                if tqdm:
                    post = dict(loss=f"{total/count:.4f}", acc=f"{acc_sum/count:.4f}", mae=f"{mae_sum/count:.4f}")
                    if use_amp and args.show_scale:
                        post["scale"] = f"{scaler.get_scale():.0f}"
                    bar.set_postfix(**post)
                elif time.time() - t_last >= args.log_every:
                    t_last = time.time()
                    rate = count / max(t_last - t0, 1e-9)
                    print(
                        f"epoch {epoch+1}/{args.epochs} step {i+1}/{len(ds)} "
                        f"loss {total/count:.4f} acc {acc_sum/count:.4f} mae {mae_sum/count:.4f} {rate:.2f} it/s"
                    )
            avg = total / max(count, 1)
            avg_acc = acc_sum / max(count, 1)
            avg_mae = mae_sum / max(count, 1)
            if sched:
                sched.step(avg)
            lr = opt.param_groups[0]["lr"]
            print(f"epoch {epoch} loss {avg:.6f} acc {avg_acc:.4f} mae {avg_mae:.4f} lr {lr:.2e}")
            # format compatible with tools/nnue/plot.py
            hyperparam = {
                "learn rate": f"{lr:.2e}",
                "batch size": args.batch_size,
                "profile": args.profile,
                "sampling ratio": args.sample,
            }
            logging.info(f"epoch={epoch} loss={avg:.6f} hyperparam={hyperparam}")
            logging.info(f"epoch={epoch} accuracy={avg_acc:.6f}")
            logging.info(f"epoch={epoch} mae={avg_mae:.6f}")
            if args.model and (args.sample or avg < best):  # sampling: loss not comparable across epochs
                best = avg
                save_bin(model, args.model)
    except KeyboardInterrupt:
        if tqdm and bar is not None:
            bar.close()
        print("\nStopping")
        sys.exit(130 if os.name != "nt" else 1)  # non-zero so scripts see interruption (130 = 128 + SIGINT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", help="h5 dataset path")
    p.add_argument("-b", "--batch-size", type=int, default=16384)
    p.add_argument("-e", "--epochs", type=int, default=100)
    p.add_argument("-m", "--model", help="output weights.bin checkpoint path (best loss)")
    # Defaults below are tuned for SGD (--optimizer sgd). For adam/amsgrad use a
    # much smaller --learn-rate (e.g. 1e-4) and lower --momentum (~0.5).
    p.add_argument("-r", "--learn-rate", type=float, default=1e-2)
    p.add_argument("-o", "--export", help="export weights.bin and exit")
    p.add_argument(
        "-q",
        "--quant-round",
        action="store_true",
        help="round weights to 1/Q_SCALE (training constraint and on export)",
    )
    p.add_argument("-s", "--outcome-smoothing", type=float, default=0.025)
    p.add_argument("-d", "--decay", type=float)
    p.add_argument("-L", "--logfile", default="train_torch.log")
    p.add_argument("--import-file", help="load weights.bin to continue training")
    p.add_argument("--optimizer", choices=["sgd", "adam", "amsgrad"], default="sgd")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--clip-eval", type=int)
    p.add_argument("--outcome-weight", type=float, default=0.1)
    p.add_argument("--outcome-scale", type=float, default=400.0)
    p.add_argument("--dynamic-outcome-weight", action="store_true")
    p.add_argument("--loss-mae", action="store_true")
    p.add_argument("--loss-bce", action="store_true")
    p.add_argument("--loss-blend", action="store_true")
    p.add_argument("--huber-delta", type=float, default=1.5)
    p.add_argument("--profile", help="JSON dataset profile with per-bucket label scale ratios")
    p.add_argument("--sample", type=float)
    p.add_argument("-F", "--filter", type=int, help="drop positions with |eval| >= this (centipawns)")
    p.add_argument("--balance", action="store_true", help="augment each batch with color-mirrored positions")
    p.add_argument("--no-capture", action="store_true", help="exclude positions whose move is a capture")
    p.add_argument(
        "--discard-mismatch",
        type=float,
        default=0,
        help="drop positions where |eval| > threshold (cp) and outcome disagrees",
    )
    p.add_argument("--mem-limit", type=int, default=0, help="GPU memory limit in MB (0 = unlimited)")
    p.add_argument("--schedule", action="store_true")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--ncols", type=int, default=120, help="progress bar width (0 = auto/full terminal)")
    p.add_argument("--log-every", type=float, default=30, help="seconds between status lines when tqdm is missing")
    p.add_argument("--gpu", dest="gpu", action="store_true", default=True)
    p.add_argument("--no-gpu", dest="gpu", action="store_false")
    p.add_argument("--no-data-parallel", action="store_true", help="use a single GPU even if several are visible")
    p.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", default=True)
    p.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    p.add_argument("--show-scale", action="store_true", help="show AMP GradScaler scale in progress bar")
    args = p.parse_args()

    if not args.input and not args.export:
        p.error("input dataset required (or use --export)")
    main(args)
