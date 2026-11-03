#!/usr/bin/env python3
"""
Inspect a training VDS: bucket coverage, eval/outcome agreement and repeated positions, per chunk.
"""
import argparse
import concurrent.futures
import itertools
import json
import logging
import math
import multiprocessing
import os
import signal
import sys
import time

import h5py
import numpy as np
import scipy.optimize
import scipy.stats
import torch

import train_torch as tt

MIN_SLICES = 32  # per chunk before its SE is trusted (SE itself known to ~1 / sqrt(2 * 31) = 13%)
RELIABILITY = 0.8  # chunk done when true spread / (true spread + noise) reaches this, the conventional "good"
TASK_SLICES = 256  # slices per worker task: one trainer batch of reads
Z95 = 1.96
ALPHA = 0.05  # spread between chunks or members counts only if significant at this level
SLICE_BITS = 40  # row key: chunk << SLICE_BITS | slice
SHOW = 1e3  # excess is displayed x1e3
MIX_ROUNDS = 20  # member caps vs implied trainer rows; converges in a few
RANGE_ROWS = 1 << 20  # rows per mapped range: ~1000 mappings per billion rows, each chunk's share spread over it
SHOW_BUCKETS = 8  # build report: buckets furthest from target

DS = None  # H5Batches, inherited by forked workers
CHUNK_START = None


def sources(data, path):
    """(dst start, rows, src start, src path) per virtual source, ascending; '.' resolves to path."""
    base = os.path.dirname(os.path.abspath(path))
    cols = data.shape[1]
    out = []
    for vs in data.virtual_sources():
        (dst, _), (dst_last, _) = vs.vspace.get_select_bounds()
        rows = dst_last - dst + 1
        whole = vs.src_space.get_select_type() == h5py.h5s.SEL_ALL
        src, src_last = (0, rows - 1) if whole else (b[0] for b in vs.src_space.get_select_bounds())
        contiguous = vs.vspace.get_select_npoints() == rows * cols and src_last - src + 1 == rows
        contiguous &= whole or vs.src_space.get_select_npoints() == rows * cols
        if not contiguous:
            raise ValueError(f"{path}: {vs.file_name}: only contiguous row ranges are supported")
        p = vs.file_name
        p = os.path.abspath(path) if p == "." else (p if os.path.isabs(p) else os.path.join(base, p))
        out.append((dst, rows, src, p))
    out.sort()
    for (a, arows, _, apath), (b, _, _, bpath) in zip(out, out[1:]):
        if b < a + arows:
            raise ValueError(f"{path}: {apath} and {bpath} map overlapping rows")
    return out


def chunk_table(path):
    """Direct members of path, and their direct sources (chunks) in container rows: (member, path, start, rows)."""
    with h5py.File(path, "r") as hf:
        data = hf["data"]
        members = sources(data, path) if data.is_virtual else [(0, data.shape[0], 0, os.path.abspath(path))]
    chunks = []
    for m, (start, rows, src, mpath) in enumerate(members):
        leaves = [(0, src + rows, 0, mpath)]
        if mpath != os.path.abspath(path):
            with h5py.File(mpath, "r") as hf:
                if hf["data"].is_virtual:
                    leaves = sources(hf["data"], mpath)
        for lstart, lrows, _, lpath in leaves:
            lo, hi = max(lstart, src), min(lstart + lrows, src + rows)  # leaf rows the container maps
            if lo < hi:
                chunks.append((m, lpath, start + lo - src, hi - lo))
    return members, chunks


def splitmix(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def position_hash(x):
    h = np.zeros(len(x), dtype=np.uint64)
    for col in x.T.astype(np.uint64):
        h = splitmix(h ^ col)
    return h


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent handles ^C
    torch.set_num_threads(1)


def sample(task):
    """Trainer-view rows of one chunk's slices: per kept row slice, wdl target, outcome, hash, bucket ids."""
    chunk, slices = task
    before = DS.invalid
    x, y, keep = DS.read_slices(CHUNK_START[chunk] + slices * DS.slice_rows)
    invalid = DS.invalid - before
    slice_of = np.repeat(slices, DS.slice_rows)[keep]
    if not len(x):
        return chunk, slices, invalid, None
    with np.errstate(over="ignore"):
        p = 1.0 / (1.0 + np.exp(-y[:, 0] / y[:, 2] * tt.SCALE / y[:, 3]))  # combined_loss wdl_target, no --clip-eval
    packed = torch.from_numpy(x)
    ids = {}
    for name, (fn, n) in tt.PARTITIONS.items():
        ids[name] = fn(packed).numpy().reshape(-1, len(x)).T  # (rows, views)
        if ((ids[name] < 0) | (ids[name] >= n)).any():
            raise ValueError(f"chunk {chunk}: {name} id out of range")
        ids[name] = ids[name].astype(np.uint8)
    return chunk, slices, invalid, (slice_of, p, y[:, 1], position_hash(x), ids)


class Store:
    def __init__(self, num_chunks):
        self.used = [np.empty(0, dtype=np.int64) for _ in range(num_chunks)]
        self.invalid = np.zeros(num_chunks, dtype=np.int64)
        self.parts = []

    def add(self, result):
        chunk, slices, invalid, rows = result
        self.used[chunk] = np.concatenate([self.used[chunk], slices])
        self.invalid[chunk] += invalid
        if rows is not None:
            slice_of, p, y, h, ids = rows
            self.parts.append(((np.int64(chunk) << SLICE_BITS) | slice_of, p, y, h, ids))

    def rows(self):
        key, p, y, h = (np.concatenate([part[i] for part in self.parts]) for i in range(4))
        ids = {name: np.concatenate([part[4][name] for part in self.parts]) for name in tt.PARTITIONS}
        return key, p, y, h, ids


def bin_edges(p):
    """Freedman-Diaconis bins, at most sqrt(n) of them."""
    q25, q75 = np.percentile(p, [25, 75])
    width = 2.0 * (q75 - q25) * len(p) ** (-1.0 / 3.0)
    span = p.max() - p.min()
    fd = math.ceil(span / width) if width > 0 else 1
    return np.histogram_bin_edges(p, bins=max(1, min(fd, math.isqrt(len(p)))))


def agreement(key, p, y, slices, available, row_weight, num_chunks):
    """Per chunk: excess Brier of wdl target vs outcome over the rest of the data at equal target (nan without
    any), its SE with and without finite-population correction, kept rows."""
    b = (p - y) ** 2
    edges = bin_edges(p)
    num_bins = len(edges) - 1
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, num_bins - 1)
    chunk_of = (key >> SLICE_BITS).astype(np.int64)
    cell = chunk_of * num_bins + bins
    wb = np.bincount(cell, weights=row_weight * b, minlength=num_chunks * num_bins).reshape(num_chunks, num_bins)
    ww = np.bincount(cell, weights=row_weight, minlength=num_chunks * num_bins).reshape(num_chunks, num_bins)
    rest_w = ww.sum(axis=0) - ww  # the chunk itself left out of its reference
    with np.errstate(divide="ignore", invalid="ignore"):
        rest = np.where(rest_w > 0, (wb.sum(axis=0) - wb) / rest_w, 0.0)
    compared = (rest_w > 0).ravel()[cell]
    excess = np.where(compared, b - rest.ravel()[cell], 0.0)

    uk, inv = np.unique(key, return_inverse=True)
    e_s, n_s = np.bincount(inv, weights=excess), np.bincount(inv, weights=compared)
    slice_chunk = (uk >> SLICE_BITS).astype(np.int64)
    kept = np.bincount(chunk_of, minlength=num_chunks).astype(np.float64)
    n_c = np.bincount(slice_chunk, weights=n_s, minlength=num_chunks)
    nonempty = np.bincount(slice_chunk, weights=n_s > 0, minlength=num_chunks)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.bincount(slice_chunk, weights=e_s, minlength=num_chunks) / n_c
        resid = e_s - np.nan_to_num(mean)[slice_chunk] * n_s
        dev2 = np.bincount(slice_chunk, weights=resid**2, minlength=num_chunks)
        se_raw = np.sqrt(slices / (slices - 1) * dev2) / n_c  # ratio estimator, slices as clusters
        se_raw[(nonempty < 2) | (n_c == 0)] = np.inf
        se = se_raw * np.sqrt(np.maximum(0.0, 1.0 - slices / available))  # finite population: 0 when all read
    se[(slices >= available) & (n_c > 0)] = 0.0
    se[n_c == 0] = np.inf
    return mean, se, se_raw, kept


def heterogeneity(mean, se, trusted):
    """Hedges' tau^2: spread of the means beyond their noise, unweighted; 0 unless Cochran's Q rejects homogeneity."""
    ok = trusted & np.isfinite(se) & (se > 0)
    if ok.sum() < 2:
        return 0.0
    m, s = mean[ok], se[ok]
    w = 1.0 / s**2
    q = np.sum(w * (m - np.sum(w * m) / np.sum(w)) ** 2)
    if scipy.stats.chi2.sf(q, len(m) - 1) >= ALPHA:
        return 0.0
    return max(0.0, np.var(m, ddof=1) - np.mean(s**2))


def squared_counts(num_groups, group, *cols):
    """Per group: sum over its distinct (group, *cols) values of count^2."""
    keys = (group,) + cols
    order = np.lexsort(keys[::-1])
    keys = [k[order] for k in keys]
    new = np.r_[True, np.any([k[1:] != k[:-1] for k in keys], axis=0)]
    run = np.diff(np.r_[np.flatnonzero(new), len(new)]).astype(np.float64)
    return np.bincount(keys[0][new], weights=run**2, minlength=num_groups)


def copies(num_groups, group, key, h, rows):
    """Per group: expected rows identical to a random one among its rows, from identical pairs across
    different sampled slices (games); sampling-rate independent. nan without such pairs."""
    same = squared_counts(num_groups, group, h) - squared_counts(num_groups, group, h, key)
    pairs = squared_counts(num_groups, group) - squared_counts(num_groups, group, key)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pairs > 0, same / pairs * rows, np.nan)


def member_shares(key, ids, n, slices, available, w, member, num_members):
    """Per member and bucket: share of trainer entries (chunks weighted by w), its SE with and without
    finite-population correction; slices as clusters."""
    uk, inv = np.unique(key, return_inverse=True)
    slice_chunk = (uk >> SLICE_BITS).astype(np.int64)
    views = ids.shape[1]
    a = np.bincount(np.repeat(inv, views) * n + ids.ravel(), minlength=len(uk) * n).reshape(len(uk), n)
    t = a.sum(axis=1).astype(np.float64)
    num_chunks = len(slices)
    a_c, t_c = np.zeros((num_chunks, n)), np.bincount(slice_chunk, weights=t, minlength=num_chunks)
    np.add.at(a_c, slice_chunk, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.nan_to_num(a_c / t_c[:, None])
        dev2 = np.zeros((num_chunks, n))
        np.add.at(dev2, slice_chunk, (a - frac[slice_chunk] * t[:, None]) ** 2)
        var_raw = (slices / (slices - 1))[:, None] * dev2 / (t_c**2)[:, None]
        var_raw[np.bincount(slice_chunk, weights=t > 0, minlength=num_chunks) < 2] = np.inf
        var = var_raw * np.maximum(0.0, 1.0 - slices / available)[:, None]
        var[slices >= available] = 0.0
        wn = w / np.maximum(np.bincount(member, weights=w, minlength=num_members), np.finfo(np.float64).tiny)[member]
        share, v, v_raw = (np.zeros((num_members, n)) for _ in range(3))
        np.add.at(share, member, wn[:, None] * frac)
        np.add.at(v, member, np.where(wn[:, None] > 0, wn[:, None] ** 2 * var, 0.0))
        # source spread: chunks with an SE estimate (>= 2 non-empty slices) only
        np.add.at(v_raw, member, np.where((wn[:, None] > 0) & np.isfinite(var_raw), wn[:, None] ** 2 * var_raw, 0.0))
    return share, np.sqrt(v), np.sqrt(v_raw)


def summarize(store, chunks, available):
    if not store.parts:
        return None
    num_chunks = len(chunks)
    slices = np.array([len(u) for u in store.used], dtype=np.float64)
    key, p, y, h, ids = store.rows()
    chunk_of = (key >> SLICE_BITS).astype(np.int64)
    rows = np.array([c[3] for c in chunks], dtype=np.float64)
    row_weight = (rows / np.maximum(slices * DS.slice_rows, 1))[chunk_of]  # inverse sampling rate
    mean, se, se_raw, kept = agreement(key, p, y, slices, available, row_weight, num_chunks)
    trainer_rows = rows * kept / np.maximum(slices * DS.slice_rows, 1)  # estimated, per chunk
    # spread of the generating sources: unshrunk SEs, chunks with trusted SEs
    trusted = slices >= MIN_SLICES
    member = np.array([c[0] for c in chunks])
    num_members = member.max() + 1
    w = np.where(trusted & np.isfinite(se_raw), trainer_rows, 0.0)  # member estimates from trusted chunks only
    w_m = np.bincount(member, weights=w, minlength=num_members)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_mean = np.bincount(member, weights=w * np.where(w > 0, mean, 0.0), minlength=num_members) / w_m
        se_w = w * np.where(w > 0, se_raw, 0.0)
        m_se = np.sqrt(np.bincount(member, weights=se_w**2, minlength=num_members)) / w_m
    m_trusted = w_m > 0
    between = heterogeneity(m_mean, m_se, m_trusted)
    # chunks resolve against the spread within their member, else between members, else none (done)
    chunk_tau2 = np.full(num_chunks, between)
    for m in range(num_members):
        sel = member == m
        within = heterogeneity(mean[sel], se_raw[sel], trusted[sel])
        if within > 0:
            chunk_tau2[sel] = within
    tau2 = between
    with np.errstate(divide="ignore", invalid="ignore"):
        reliability = np.where(se == 0, 1.0, chunk_tau2 / (chunk_tau2 + se**2))

    # members' histograms resolve as a whole against how much histograms differ between members
    shares, shares_done = {}, np.ones(num_members, dtype=bool)
    share_trusted = np.bincount(member, weights=trusted & (kept > 0), minlength=num_members) > 0
    for name, (_, n) in tt.PARTITIONS.items():
        share, s_se, s_raw = member_shares(key, ids[name], n, slices, available, trainer_rows, member, num_members)
        shares[f"share_{name}"], shares[f"share_se_{name}"] = share, s_se
        spread = sum(heterogeneity(share[:, b], s_raw[:, b], share_trusted) for b in range(n))
        noise = np.sum(s_se**2, axis=1)
        if spread > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                shares_done &= np.where(noise == 0, 1.0, spread / (spread + noise)) >= RELIABILITY

    stats = {
        "slices": slices,
        "read": slices * DS.slice_rows,
        "kept": kept,
        "invalid": store.invalid.astype(np.float64),
        "excess": mean,
        "se": se,
        "tau_ref": np.sqrt(chunk_tau2),
        "reliability": reliability,
        "draw": np.bincount(chunk_of, weights=(y == 0.5), minlength=num_chunks) / np.maximum(kept, 1),
        "copies": copies(num_chunks, chunk_of, key, h, trainer_rows),
        "tau": np.sqrt(tau2),
        "sampled": len(p),
        "shares_done": shares_done,
        **shares,
    }
    for name, (_, n) in tt.PARTITIONS.items():
        views = ids[name].shape[1]
        cell = np.repeat(chunk_of, views) * n + ids[name].ravel()
        hist = np.bincount(cell, minlength=num_chunks * n).reshape(num_chunks, n)
        entries = (trainer_rows[:, None] * views * hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)).ravel()
        dup = copies(num_chunks * n, cell, np.repeat(key, views), np.repeat(h, views), entries)
        stats[f"hist_{name}"], stats[f"copies_{name}"] = hist, dup.reshape(num_chunks, n)
    return stats


def save(path, args, members, chunks, stats):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            input=os.path.abspath(args.input),
            alt_profile_dir=os.path.abspath(args.alt_profile_dir) if args.alt_profile_dir else "",
            slice_rows=args.slice_rows,
            filter=args.filter or 0,
            no_capture=args.no_capture,
            outcome_scale=math.nan if args.outcome_scale is None else args.outcome_scale,  # nan: none needed
            partitions=np.array([f"{name}:{n}" for name, (_, n) in tt.PARTITIONS.items()]),
            member_path=np.array([m[3] for m in members]),
            member_start=np.array([m[0] for m in members]),
            member_rows=np.array([m[1] for m in members]),
            member_src=np.array([m[2] for m in members]),
            chunk_member=np.array([c[0] for c in chunks]),
            chunk_path=np.array([c[1] for c in chunks]),
            chunk_start=np.array([c[2] for c in chunks]),
            chunk_rows=np.array([c[3] for c in chunks]),
            **stats,
        )
    os.replace(tmp, path)


def mean_of(weight, values, width):
    ok = np.isfinite(values) & (weight > 0)
    return f"{np.sum(weight[ok] * values[ok]) / weight[ok].sum():>{width}.2f}" if ok.any() else f"{'-':>{width}}"


def report(members, chunks, stats):
    rows = np.array([c[3] for c in chunks], dtype=np.float64)
    member = np.array([c[0] for c in chunks])
    keep = stats["kept"] / np.maximum(stats["read"], 1)
    weight = rows * keep  # estimated trainer rows per chunk
    ci = Z95 * stats["se"] * SHOW

    print(f"\n{'member':<32} {'rows':>14} {'chunks':>6} {'keep%':>6} {'draw%':>6} {'excess':>16} {'copies':>8}")
    for m, (_, mrows, _, mpath) in enumerate(members):
        sel = (member == m) & np.isfinite(stats["excess"])
        w = weight[sel] / max(weight[sel].sum(), 1)
        ex = np.sum(w * stats["excess"][sel]) * SHOW
        exci = Z95 * np.sqrt(np.sum((w * stats["se"][sel]) ** 2)) * SHOW
        name, keep_pct = os.path.basename(mpath), 100 * weight[member == m].sum() / mrows
        draw = 100 * np.sum(w * stats["draw"][sel])
        line = f"{name:<32} {mrows:>14,} {(member == m).sum():>6}"
        if not sel.any():  # nothing sampled, all filtered, or nothing to compare with
            print(f"{line} {'-':>6} {'-':>6} {'-':>16} {'-':>8}")
            continue
        dup = mean_of(weight[member == m], stats["copies"][member == m], 8)
        print(f"{line} {keep_pct:>6.1f} {draw:>6.1f} {ex:>+8.2f} ±{exci:<6.2f} {dup}")

    print(f"\n{'chunk (worst first)':<40} {'member':<24} {'rows':>13} {'excess':>16} {'rel':>5} {'copies':>8}")
    for c in np.argsort(-np.where(np.isfinite(stats["excess"]), stats["excess"], -np.inf)):
        if not np.isfinite(stats["excess"][c]):
            continue
        name, mname = os.path.basename(chunks[c][1]), os.path.basename(members[member[c]][3])
        no_spread = stats["tau_ref"][c] == 0 and stats["se"][c] > 0  # stopped: chunks indistinguishable
        rel = f"{'-':>5}" if no_spread else f"{stats['reliability'][c]:>5.2f}"
        print(
            f"{name:<40} {mname:<24} {chunks[c][3]:>13,} {stats['excess'][c] * SHOW:>+8.2f} ±{ci[c]:<6.2f}"
            f" {rel} {mean_of(weight[c : c + 1], stats['copies'][c : c + 1], 8)}"
        )

    for name in tt.PARTITIONS:
        hist, dup = stats[f"hist_{name}"], stats[f"copies_{name}"]
        frac = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)
        cell_rows = weight[:, None] * frac  # estimated trainer rows per chunk and bucket
        share = cell_rows.sum(axis=0)
        print(f"\n{name:>6} {'share%':>7} {'copies':>8}")
        for i in range(hist.shape[1]):
            print(f"{i:>6} {100 * share[i] / share.sum():>7.2f} {mean_of(cell_rows[:, i], dup[:, i], 8)}")

    for c in np.flatnonzero(stats["invalid"]):
        bad, read = stats["invalid"][c], stats["read"][c]
        print(f"WARNING: {chunks[c][1]}: {bad:,.0f} of {read:,.0f} sampled rows lack one king per side")

    print(
        f"\nexcess: Brier(wdl target, outcome) minus the rest of the data at equal target, x{SHOW:g}, 95% CI;"
        f" rel: reliability ('-': chunks indistinguishable); copies: expected other rows identical to a row, within"
        f" its chunk (bucket), from identical pairs across sampled games; {stats['read'].sum() / rows.sum():.4%} read"
    )


def inspect(args):
    global DS, CHUNK_START
    logging.getLogger().setLevel(logging.ERROR)  # invalid rows: counted and reported instead
    members, chunks = chunk_table(args.input)
    DS = tt.H5Batches(
        args.input,
        args.slice_rows,
        args.slice_rows,
        args.alt_profile_dir,
        sample=None,
        smoothing=0.0,
        filter=args.filter,
        no_capture=args.no_capture,
        profile_ratios=None,
        outcome_scale=math.nan,  # until set below
    )
    if args.outcome_scale is not None:
        DS.set_default_outcome_scale(args.outcome_scale)
    elif np.isnan(DS.member_oscales).any():  # as the trainer: fitted, for members without their own
        args.outcome_scale, ci = tt.fit_outcome_scale(DS)
        DS.set_default_outcome_scale(args.outcome_scale)
        print(f"outcome scale {args.outcome_scale:.1f} ±{ci:.1f} (fitted)")
    with h5py.File(args.input, "r") as hf:
        unmapped = hf["data"].shape[0] - sum(c[3] for c in chunks)  # container gaps, and gaps inside members
    if unmapped:
        print(f"WARNING: {unmapped:,} rows map no data (read as zeros), not inspected")
    CHUNK_START = np.array([c[2] for c in chunks], dtype=np.int64)
    available = np.array([c[3] // args.slice_rows for c in chunks], dtype=np.float64)
    print(f"{len(members)} member(s), {len(chunks)} chunk(s)")
    pending = np.flatnonzero(available > 0)
    if not len(pending):
        sys.exit(f"{args.input}: no chunk holds a full {args.slice_rows}-row slice")

    store = Store(len(chunks))
    rng = np.random.default_rng()
    stats, t0, interrupted = None, time.time(), False
    ctx = multiprocessing.get_context("fork")
    pool = concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=ctx, initializer=init_worker)
    try:
        for rnd in itertools.count():
            tasks = []
            for c in pending:
                used = store.used[c]
                want = max(MIN_SLICES, 2 * len(used))  # doubling per round
                if want >= available[c]:
                    new = np.setdiff1d(np.arange(available[c], dtype=np.int64), used)
                else:
                    new = np.setdiff1d(rng.choice(int(available[c]), want - len(used), replace=False), used)
                if len(new):
                    tasks += [(c, s) for s in np.array_split(new, -(-len(new) // TASK_SLICES))]
            for future in concurrent.futures.as_completed([pool.submit(sample, t) for t in tasks]):
                store.add(future.result())
            round_stats = summarize(store, chunks, available)
            slices = np.array([len(u) for u in store.used], dtype=np.float64)
            done = slices >= available
            trusted = slices >= MIN_SLICES
            if round_stats is None:
                done |= trusted  # nothing kept anywhere
            else:
                stats = round_stats
                save(args.output, args, members, chunks, stats)
                quality = (stats["reliability"] >= RELIABILITY) | (stats["tau_ref"] == 0)
                shares = stats["shares_done"][[c[0] for c in chunks]]
                done |= trusted & ((quality & shares) | (stats["kept"] == 0))
            pending = np.flatnonzero(~done)
            sampled = stats["sampled"] if stats else 0
            tau = stats["tau"] * SHOW if stats else 0.0
            elapsed = time.time() - t0
            print(f"round {rnd}: {sampled:,} rows, {len(pending)} chunk(s) pending, tau x{SHOW:g} {tau:.3f}, {elapsed:.0f}s")
            if not len(pending):
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: report from the last completed round")
    finally:
        pool.shutdown(wait=not interrupted, cancel_futures=True)
    if interrupted:
        if stats is not None:
            report(members, chunks, stats)
        sys.exit(130)
    if stats is None:
        sys.exit(f"{args.input}: no rows survived the filters")
    report(members, chunks, stats)


def load_target(paths):
    """Engine bucket counts (nnue-counts-*.json files, or directories of them) -> {name: shares}, per-file shares."""
    files = [p for p in paths if not os.path.isdir(p)]
    for d in (p for p in paths if os.path.isdir(p)):
        names = sorted(f for f in os.listdir(d) if f.startswith("nnue-counts-") and f.endswith(".json"))
        files += [os.path.join(d, f) for f in names]
    if not files:
        sys.exit(f"no nnue-counts-*.json in {' '.join(paths)}")
    per_file = []
    for path in files:
        with open(path) as f:
            per_file.append({name: np.asarray(v, dtype=np.float64) for name, v in json.load(f).items()})
    total = {name: sum(c[name] for c in per_file) for name in per_file[0]}
    shares = lambda counts: {name: v / v.sum() for name, v in counts.items()}
    return shares(total), [shares(c) for c in per_file]


def target_noise(target, per_file):
    """KL by which the pooled target is uncertain: files' scatter around it, over their number (0 for one file)."""
    if len(per_file) < 2:
        return 0.0
    scatter = np.mean([sum(np.sum(f[n] * np.log(np.where(f[n] > 0, f[n] / target[n], 1.0))) for n in target) for f in per_file])
    return scatter / len(per_file)


def kl(target, shares, x):
    """Sum over partitions of KL(target || mix of member shares by x), and its gradient."""
    val, grad = 0.0, np.zeros_like(x)
    for name, t in target.items():
        q = np.maximum(x @ shares[name], np.finfo(np.float64).tiny)
        val += np.sum(np.where(t > 0, t * np.log(np.where(t > 0, t, 1.0) / q), 0.0))
        grad -= shares[name] @ (t / q)
    return val, grad


def mix(target, shares, avail, keep, n, noise):
    """Member weights over trainer rows: closest to the source's own mix among those matching target to within
    noise of the best match, each within its available rows, n VDS rows in all; -> (weights, trainer rows)."""
    source = avail / avail.sum()
    x = source
    trainer_rows = n / np.sum(x / np.where(keep > 0, keep, np.inf))
    one = {"type": "eq", "fun": lambda v: v.sum() - 1.0, "jac": lambda v: np.ones_like(v)}
    for _ in range(MIX_ROUNDS):  # caps depend on the trainer rows the mix implies
        caps = np.minimum(1.0, avail / trainer_rows)
        bounds = list(zip(np.zeros_like(caps), caps))
        best = scipy.optimize.minimize(
            lambda v: kl(target, shares, v), np.minimum(x, caps), jac=True, method="SLSQP", bounds=bounds, constraints=[one]
        )
        if not best.success:
            sys.exit(f"target fit failed: {best.message}")
        # least move from the source mix toward the best fit that lands within target noise of it (KL convex on the
        # segment): sources keep a share unless the whole move is needed
        seg = lambda a: (1.0 - a) * np.minimum(source, caps) + a * np.clip(best.x, 0.0, caps)
        lo, hi = 0.0, 1.0
        if kl(target, shares, seg(lo))[0] <= best.fun + noise:
            hi = lo
        for _ in range(np.finfo(np.float64).nmant):  # bisect to float resolution
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if kl(target, shares, seg(mid))[0] <= best.fun + noise else (mid, hi)
        x = seg(hi)
        x /= x.sum()
        rows = n / np.sum(x / np.where(keep > 0, keep, np.inf))
        if abs(rows - trainer_rows) < 1.0:
            break
        trainer_rows = rows
    return x, trainer_rows


def valid_row(data, i):
    r = data[i]
    return tt.popcount(r[0:1])[0] == 1 and tt.popcount(r[1:2])[0] == 1


def usable_rows(data, start, rows):
    """Rows of a chunk before a trailing run of invalid rows (binary search; assumes the run is trailing)."""
    if valid_row(data, start + rows - 1):
        return rows
    lo, hi = 0, rows - 1  # lo: candidate valid, hi: invalid
    while lo < hi:
        mid = (lo + hi) // 2
        if valid_row(data, start + mid):
            lo = mid + 1
        else:
            hi = mid
    return lo


def ranges(start, usable, take):
    """take rows from [start, start + usable) as evenly spaced ranges of at most RANGE_ROWS."""
    if not take:
        return []
    k = max(1, -(-take // RANGE_ROWS))
    bounds = (np.arange(k + 1) * usable) // k  # k pieces; each range starts its piece
    lens = np.diff(bounds)
    sizes = (take * lens) // usable
    sizes[np.flatnonzero(lens > sizes)[: take - sizes.sum()]] += 1  # remainder, where pieces have room
    return [(int(start + s), int(z)) for s, z in zip(bounds[:-1], sizes) if z]


def build(args):
    st = np.load(args.stats)
    target, per_file = load_target(args.target)
    num_files, noise = len(per_file), target_noise(target, per_file)
    parts = dict(p.split(":") for p in st["partitions"])
    if {k: int(v) for k, v in parts.items()} != {k: len(v) for k, v in target.items()}:
        sys.exit(f"partitions {dict(parts)} vs target {dict((k, len(v)) for k, v in target.items())}")
    member, rows = st["chunk_member"], st["chunk_rows"].astype(np.float64)
    num_members = len(st["member_path"])
    keep_c = st["kept"] / np.maximum(st["read"], 1)
    avail = np.bincount(member, weights=rows * keep_c, minlength=num_members)  # trainer rows per member
    keep = avail / np.maximum(np.bincount(member, weights=rows, minlength=num_members), 1)
    if args.rows > rows.sum():
        sys.exit(f"-n {args.rows:,} exceeds the {rows.sum():,.0f} rows inspected")
    shares = {name: st[f"share_{name}"] for name in target}
    if str(st["input"]) in map(str, st["member_path"]):
        sys.exit(f"{st['input']}: self-referencing ('.') sources are not supported by build")
    x, trainer_rows = mix(target, shares, avail, keep, args.rows, noise)
    want = np.minimum(x * trainer_rows / np.where(keep > 0, keep, np.inf), st["member_rows"])
    take_m = np.floor(want)
    room = np.flatnonzero((take_m < st["member_rows"]) & (want > 0))
    extra = room[np.argsort(take_m[room] - want[room])][: max(0, args.rows - int(take_m.sum()))]
    take_m[extra] += 1  # exactly -n rows: remainder to the largest fractions

    with h5py.File(str(st["input"]), "r") as hf:
        data = hf["data"]
        cols, dtype = data.shape[1], data.dtype
        mapped = []  # (member, container start, rows)
        for m in range(num_members):
            sel = np.flatnonzero(member == m)
            usable = np.array([usable_rows(data, int(st["chunk_start"][c]), int(rows[c])) for c in sel])
            trimmed = rows[sel] - usable
            for c, t in zip(sel, trimmed):
                if t:
                    print(f"{st['chunk_path'][c]}: trailing {t:,.0f} invalid rows left out")
            share = usable / max(usable.sum(), 1)
            want = np.floor(take_m[m] * share).astype(np.int64)
            want[np.flatnonzero(usable > want)[: int(take_m[m] - want.sum())]] += 1  # remainder, where there's room
            for c, u, w in zip(sel, usable, np.minimum(want, usable)):
                mapped += [(m, s, z) for s, z in ranges(int(st["chunk_start"][c]), int(u), int(w))]

    layout = h5py.VirtualLayout(shape=(sum(z for _, _, z in mapped), cols), dtype=dtype)
    sources, pos = {}, 0
    for m, s, z in mapped:
        mpath = str(st["member_path"][m])
        if m not in sources:
            with h5py.File(mpath, "r") as hf:
                sources[m] = h5py.VirtualSource(mpath, "data", shape=hf["data"].shape)
        src = s - int(st["member_start"][m]) + int(st["member_src"][m])
        layout[pos : pos + z] = sources[m][src : src + z]
        pos += z
    with h5py.File(args.output, "w", libver="latest") as out:
        out.create_virtual_dataset("data", layout, fillvalue=None)

    # report: what the curated mix predicts, against the target and the source's own mix
    source = avail / avail.sum()
    print(f"\ntarget: {num_files} engine count file(s), noise KL {noise:.4f}; {len(mapped):,} range(s), {pos:,} rows -> {args.output}")
    print(f"{'member':<32} {'rows':>14} {'taken':>14} {'taken%':>7} {'output%':>8}")
    for m in range(num_members):
        print(
            f"{os.path.basename(str(st['member_path'][m])):<32} {st['member_rows'][m]:>14,} {int(take_m[m]):>14,}"
            f" {100 * take_m[m] / st['member_rows'][m]:>7.1f} {100 * x[m]:>8.2f}"
        )
    for name, t in target.items():
        cur, src = x @ shares[name], source @ shares[name]
        print(f"\n{name}: KL to target {kl({name: t}, shares, source)[0]:.4f} (source) -> {kl({name: t}, shares, x)[0]:.4f}")
        print(f"{name:>6} {'target%':>8} {'source%':>8} {'output%':>8}")
        for b in np.argsort(-np.abs(cur - t))[:SHOW_BUCKETS]:
            print(f"{b:>6} {100 * t[b]:>8.2f} {100 * src[b]:>8.2f} {100 * cur[b]:>8.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inspect", help="sample a VDS per chunk")
    i.add_argument("input", help="training VDS (h5cat.py for a file list)")
    i.add_argument("-o", "--output", required=True, help="stats .npz")
    i.add_argument("--alt-profile-dir", help="as train_torch.py")
    i.add_argument("-F", "--filter", type=int, help="as train_torch.py")
    i.add_argument("--no-capture", action="store_true", help="as train_torch.py")
    i.add_argument("--outcome-scale", type=float, help="as train_torch.py (default: fitted)")
    i.add_argument("--slice-rows", type=int, default=64, help="as train_torch.py")
    i.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    b = sub.add_parser("build", help="curated VDS from inspect stats")
    b.add_argument("stats", help="inspect .npz")
    b.add_argument("-n", "--rows", type=lambda s: int(float(s)), required=True, help="output rows, e.g. 1e9")
    b.add_argument("-t", "--target", nargs="+", required=True, help="nnue-counts-*.json files or their directories")
    b.add_argument("-o", "--output", required=True, help="output VDS .h5")
    args = p.parse_args()
    if args.command == "build":
        build(args)
        sys.exit()
    if args.alt_profile_dir and not os.path.isdir(args.alt_profile_dir):
        p.error(f"--alt-profile-dir: {args.alt_profile_dir} not found")
    inspect(args)
