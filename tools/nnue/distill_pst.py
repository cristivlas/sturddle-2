#!/usr/bin/env python3
#
# Learn piece-square tables from a trained net.
#
# The engine values a piece as its nominal worth plus a bonus for its square. We want bonuses that make
# that simple sum agree with the net as often as possible, so we ask the net about lots of real positions
# and solve for the bonuses by least squares.
#
# One table has to cover the whole game, so it can't be right everywhere. We let positions with roughly
# even material decide where pieces belong, since that's where the bonuses actually swing decisions; how
# much a piece is worth overall is still taken from every position.
#
# Kings need a different approach. A king usually walks up the board because its side is already winning,
# so a plain fit would reward the walk for the win. Instead we take a position, try the king on every
# square it could legally stand on, and see what the net thinks of each. Comparing squares within the same
# position keeps the rest of the board out of it. A few tweaks on top:
#   - a handful of squares are outright blunders (the king walks into mate); we clip those so the typical
#     cost of a square wins out;
#   - far-advanced squares are only legal in thin positions, where an active king is fine; we let each
#     value change with the number of pieces and read it off at a typical count;
#   - the net sees the queenside as a mirror of the kingside, but most games castle short, so each square
#     is averaged with its mirror to give long castling a fair shake;
#   - both kings use the same table, so only differences matter; we shift it so the usual squares are 0.
#
# The queen has the same problem as the king: a queen on g6 mostly shows up when its side is already
# attacking. So we relocate it too, to every empty square where it can't be taken and gives no check, and
# mirror its table across the d/e line like the king's. Its level still comes from the plain fit.
#
# Moving pieces around is expensive, so this part stops as soon as the tables are pinned down well enough.
#
import argparse
import functools
import os
import re
import sys
import time
import types

import h5py
import numpy as np
import torch
from tqdm import tqdm

import train_torch as tt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_engine_constants():
    """Pull WEIGHT/ADJUST/bucket/endgame constants out of chess.h and common.h."""
    with open(os.path.join(REPO_ROOT, "common.h"), encoding="utf-8") as f:
        common = f.read()
    with open(os.path.join(REPO_ROOT, "chess.h"), encoding="utf-8") as f:
        chess = f.read()

    grading = re.search(r"#define EVAL_PIECE_GRADING\s+(\w+)", common).group(1) == "true"
    endgame = int(re.search(r"constexpr int ENDGAME_PIECE_COUNT\s*=\s*(\d+)", common).group(1))
    buckets = int(re.search(r"constexpr int PAWN_BUCKETS\s*=\s*(\d+)", chess).group(1))

    values = re.findall(r"#define PIECE_VALUES\s*\{([^}]*)\}", chess)
    weight = [int(v) for v in values[0 if grading else 1].split(",")]
    assert len(weight) == 7, weight

    if not grading:
        return weight, [[0] * 7 for _ in range(buckets)], buckets, endgame

    macro = chess[chess.index("#define GRADING_ADJUST") :]
    macro = macro[: re.search(r"\n[^\\\n]*\n", macro).start()]  # macro ends with the last '\'-continued line
    macro = re.sub(r"/\*.*?\*/", "", macro, flags=re.S)
    adjust = [[int(v) for v in row.split(",") if v.strip()] for row in re.findall(r"\{([^{}]*)\}", macro)]
    assert len(adjust) == buckets and all(len(row) == 7 for row in adjust), adjust

    return weight, adjust, buckets, endgame


WEIGHT, GRADING_ADJUST, PAWN_BUCKETS, ENDGAME_PIECE_COUNT = _parse_engine_constants()

# H5 column base per piece type (+0 black, +1 white); PST emit order and chess.h PieceType index
H5_BASE = {"PAWN": 2, "KNIGHT": 4, "BISHOP": 6, "ROOK": 8, "QUEEN": 10, "KING": 0}
PST_ORDER = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]
PIECE_TYPE = {"PAWN": 1, "KNIGHT": 2, "BISHOP": 3, "ROOK": 4, "QUEEN": 5, "KING": 6}

PARAMS = 7 * 64  # 6 MG tables + endgame king
WEIGHTED_FIT = ["PAWN", "KNIGHT", "BISHOP", "ROOK"]  # pieces whose square patterns come from the weighted fit

RELOCATION_STAT_SHAPES = ((2, 128, 128), (2, 128), (2, 128, 128), (2, 64), (2, 2))  # gram, rhs, meat, visits, level
TABLE_PRECISION = 1.0  # cp: one rounding step of the emitted tables
CONFIDENCE_Z = 1.96  # 95% two-sided
MIN_PIECES, MAX_PIECES = 2, 32
# middle piece count of each phase
PHASE_REF = ((ENDGAME_PIECE_COUNT + 1 + MAX_PIECES) / 2, (MIN_PIECES + ENDGAME_PIECE_COUNT) / 2)
HUBER_K = 1.345  # 95% efficiency under normal noise
MAD_SCALE = 1.4826  # MAD -> std under normal noise


def pawn_bucket(pawn_count):
    return torch.where(pawn_count <= 4, 0, ((pawn_count - 1) // 4).clamp(max=PAWN_BUCKETS - 1))


def unpack(packed):
    """(B, 13) int64 tensor -> (B, 12, 64) bool, index == chess square (A1=0)."""
    return ((packed[:, :12, None] >> torch.arange(64, device=packed.device)) & 1).bool()


def mirror(bits):
    """Flip ranks: square index sq -> sq ^ 56 (white PST lookup)."""
    return bits.reshape(-1, 8, 8).flip(1).reshape(-1, 64)


def features_and_material(packed):
    """(B, 13) int64 tensor -> (B, PARAMS) float32 signed PST features and (B,) float64 graded material."""
    bits = unpack(packed).float()
    n = packed.shape[0]

    black = {name: bits[:, base] for name, base in H5_BASE.items()}
    white = {name: bits[:, base + 1] for name, base in H5_BASE.items()}

    eg = (bits.sum(dim=(1, 2)) <= ENDGAME_PIECE_COUNT).float()[:, None]

    buckets = pawn_bucket((black["PAWN"].sum(dim=1) + white["PAWN"].sum(dim=1)).long())
    adjust = torch.tensor(GRADING_ADJUST, dtype=torch.float64, device=packed.device)[buckets]  # (B, 7)

    # one column per table entry: +1 for a white piece there, -1 for black (white looks up the rank-flipped
    # square, as the engine does), so x @ tables is the PST part of a white-view eval
    x = torch.empty((n, PARAMS), device=packed.device)
    material = torch.zeros(n, dtype=torch.float64, device=packed.device)
    for i, name in enumerate(PST_ORDER):
        block = mirror(white[name]) - black[name]
        if name == "KING":
            x[:, i * 64 : (i + 1) * 64] = block * (1.0 - eg)
            x[:, (i + 1) * 64 :] = block * eg
        else:
            x[:, i * 64 : (i + 1) * 64] = block
            value = WEIGHT[PIECE_TYPE[name]] + adjust[:, PIECE_TYPE[name]]
            material += value * (white[name].sum(dim=1) - black[name].sum(dim=1))
    return x, material


def _leaper(steps):
    m = np.zeros((64, 64), dtype=np.float32)
    for s in range(64):
        f, r = s % 8, s // 8
        for df, dr in steps:
            if 0 <= f + df < 8 and 0 <= r + dr < 8:
                m[s, (r + dr) * 8 + f + df] = 1.0
    return m


def _rays(steps):
    rays = np.full((len(steps), 64, 7), 64)  # 64: off-board sentinel
    for d, (df, dr) in enumerate(steps):
        for s in range(64):
            f, r = s % 8, s // 8
            for k in range(1, 8):
                if not (0 <= f + df * k < 8 and 0 <= r + dr * k < 8):
                    break
                rays[d, s, k - 1] = (r + dr * k) * 8 + f + df * k
    return rays


ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT_ATTACKS = _leaper([(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)])
KING_ATTACKS = _leaper(ORTHO + DIAG)
PAWN_ATTACKS = [_leaper([(-1, -1), (1, -1)]), _leaper([(-1, 1), (1, 1)])]  # [black, white][from, to]
ORTHO_RAYS, DIAG_RAYS = _rays(ORTHO), _rays(DIAG)


def attacked(occ, side, color):
    """(B, 64) bool tensor: squares attacked by side's pieces (dict of (B, 64) bool), sliders blocked by occ."""
    dev = occ.device
    leapers = ((side["KNIGHT"], KNIGHT_ATTACKS), (side["KING"], KING_ATTACKS), (side["PAWN"], PAWN_ATTACKS[color]))
    att = sum(bits.float() @ torch.as_tensor(m, device=dev) for bits, m in leapers) > 0
    stop = torch.ones((occ.shape[0], 1), dtype=torch.bool, device=dev)
    occ_ext = torch.cat([occ, stop], dim=1)
    squares = torch.arange(64, device=dev)
    for rays, movers in ((ORTHO_RAYS, side["ROOK"] | side["QUEEN"]), (DIAG_RAYS, side["BISHOP"] | side["QUEEN"])):
        movers_ext = torch.cat([movers, ~stop], dim=1)
        for ray in torch.as_tensor(rays, device=dev):
            first = ray[squares, occ_ext[:, ray].byte().argmax(dim=2)]  # (B, 64) first blocker
            att |= torch.gather(movers_ext, 1, first)
    return att


def net_eval(model, packed, chunk):
    """(B, 13) int64 tensor -> (B,) float64 white-view net evals in cp."""
    out = [torch.zeros(0, dtype=torch.float64, device=packed.device)]
    for s in range(0, len(packed), chunk):
        p = packed[s : s + chunk]
        y = model(p)[:, 0].double() * tt.SCALE
        out.append(torch.where(p[:, 12] == 1, y, -y))
    return torch.cat(out)


def king_moves(model, packed, chunk):
    """Net eval of each king relocated to every legal square, everything else fixed.

    Returns endgame mask, piece count and, per color, (legal, evals, king) as (B, 64) in PST index order,
    evals from the king's side."""
    bits = unpack(packed)
    pieces = [{name: bits[:, base + c] for name, base in H5_BASE.items()} for c in (0, 1)]
    occ = bits.any(dim=1)
    count = occ.sum(dim=1)
    eg = count <= ENDGAME_PIECE_COUNT

    out = []
    for c in (0, 1):
        own, enemy = pieces[c], pieces[1 - c]
        occ_c = occ & ~own["KING"]
        legal = ~occ_c & ~attacked(occ_c, enemy, 1 - c)
        own_rest = dict(own, KING=torch.zeros_like(own["KING"]))
        checks = (attacked(occ_c, own_rest, c) & enemy["KING"]).any(dim=1)
        legal &= ~checks[:, None]  # skip positions where moving the king gives check

        b, t = torch.nonzero(legal, as_tuple=True)
        moved = packed[b]
        moved[:, H5_BASE["KING"] + c] = torch.ones_like(t) << t
        evals = torch.zeros(legal.shape, dtype=torch.float64, device=packed.device)
        evals[b, t] = net_eval(model, moved, chunk) * (1 if c else -1)
        king = own["KING"]
        out.append((mirror(legal), mirror(evals), mirror(king)) if c else (legal, evals, king))
    return eg, count, out


def sample_squares(legal, squares):
    """Keep up to `squares` random legal squares per row (0: all)."""
    if not squares:
        return legal
    pick = torch.where(legal, torch.rand(legal.shape, device=legal.device), -1.0)
    top = pick.topk(min(squares, 64), dim=1)
    return torch.zeros_like(legal).scatter_(1, top.indices, top.values >= 0)


def queen_moves(model, packed, chunk, squares):
    """Net eval of each lone queen relocated to up to `squares` random quiet empty squares (0: all of
    them), everything else fixed.

    Same output layout as king_moves. The queen table has no endgame variant, so every position goes
    into the middlegame phase."""
    bits = unpack(packed)
    pieces = [{name: bits[:, base + c] for name, base in H5_BASE.items()} for c in (0, 1)]
    occ = bits.any(dim=1)
    count = occ.sum(dim=1)
    no_endgame = torch.zeros_like(count, dtype=torch.bool)
    none = torch.zeros_like(occ)

    out = []
    for c in (0, 1):
        own, enemy = pieces[c], pieces[1 - c]
        queen = own["QUEEN"]
        occ_c = occ & ~queen

        # A queen that shields its own king, or blocks a check of its own side, can't move freely:
        # relocating it gives an illegal position. Skip such sides, and sides with zero or two queens.
        own_rest = dict(own, QUEEN=none)
        enemy_attacks = attacked(occ_c, enemy, 1 - c)
        exposed = (enemy_attacks & own["KING"]).any(dim=1)
        discovers = (attacked(occ_c, own_rest, c) & enemy["KING"]).any(dim=1)
        usable = (queen.sum(dim=1) == 1) & ~exposed & ~discovers

        # Keep the comparison positional: no squares where the queen can be taken or gives check.
        # A queen looks at the enemy king from exactly the squares a queen on the king's square sees.
        king_lines = attacked(occ_c, dict.fromkeys(H5_BASE, none) | {"QUEEN": enemy["KING"]}, c)
        legal = ~occ_c & ~enemy_attacks & ~king_lines
        legal &= usable[:, None]

        # A queen often has 30+ squares to try. A random handful per position carries nearly the same
        # information (squares of one position move together) for a fraction of the net evals, and the
        # pick does not depend on the evals, so the fit stays unbiased.
        legal = sample_squares(legal, squares)

        b, t = torch.nonzero(legal, as_tuple=True)
        evals = torch.zeros(legal.shape, dtype=torch.float64, device=packed.device)

        # A queen has many squares to try: build the moved positions a chunk at a time to bound memory.
        for s in range(0, len(b), chunk):
            bs, ts = b[s : s + chunk], t[s : s + chunk]
            moved = packed[bs]
            moved[:, H5_BASE["QUEEN"] + c] = torch.ones_like(ts) << ts
            evals[bs, ts] = net_eval(model, moved, chunk) * (1 if c else -1)

        out.append((mirror(legal), mirror(evals), mirror(queen)) if c else (legal, evals, queen))
    return no_endgame, count, out


def read_slices(data, starts, slice_rows):
    # one read for all slices: a VDS walks every mapping per read call
    space = data.id.get_space()
    space.select_none()
    for s in starts:
        space.select_hyperslab((int(s), 0), (slice_rows, data.shape[1]), op=h5py.h5s.SELECT_OR)
    rows = np.empty((len(starts) * slice_rows, data.shape[1]), dtype=data.dtype)
    data.id.read(h5py.h5s.create_simple(rows.shape), space, rows)
    return rows


def fit_outcome_scale(evals, outcomes):
    """S for outcome ~ sigmoid(eval / S), Newton on the convex 1-D logistic loss."""
    beta = 1.0 / tt.OUTCOME_SCALE_START
    while True:
        p = torch.sigmoid(beta * evals)
        hess = (p * (1 - p) * evals**2).sum()
        if not hess > 0:
            raise ValueError("outcome scale fit: no eval signal")
        step = float(((p - outcomes) * evals).sum() / hess)
        beta -= step
        if not beta > 0:
            raise ValueError(f"outcome scale fit failed (beta {beta})")
        if abs(step) <= np.finfo(np.float64).eps ** 0.5 * abs(beta):
            return 1.0 / beta


def balance_weight(material, scale):
    """Win-probability slope at the material balance, scaled to 1 for even material."""
    p = torch.sigmoid(material / scale)
    return 4 * p * (1 - p)


def relocation_stats(eg, count, sides, theta, cluster, weight):
    """Fixed-effects normal equations for one batch of relocated pieces (king or queen), deltas
    Huber-clipped on residuals vs the fit.

    theta (2, 128): per phase, square values at PHASE_REF, then their slopes per piece.
    Returns gram, rhs, meat (per-cluster score outer products, for clustered standard errors), visits
    of the relocated piece and (weighted piece-count sum, weight sum) per phase."""
    dev = eg.device
    stats = [torch.zeros(shape, dtype=torch.float64, device=dev) for shape in RELOCATION_STAT_SHAPES]
    gram, rhs, meat, visits, level = stats
    z = (count - torch.where(eg, PHASE_REF[1], PHASE_REF[0])).double()
    th = torch.where(eg[:, None], theta[1], theta[0])
    fitted = th[:, :64] + th[:, 64:] * z[:, None]  # (B, 64)
    clusters = int(cluster.max()) + 1

    def block(u, w, size):
        return torch.diag(u.sum(dim=0)) - (u / size).T @ w

    for phase, sel in enumerate((~eg, eg)):
        zs = z[sel][:, None]
        scores = torch.zeros((clusters, 128), dtype=torch.float64, device=dev)
        for legal, evals, king in sides:
            m = legal[sel]
            if not m.any():
                continue
            # clip what the fit can't explain, so tactical blowups don't dominate
            resid = evals[sel] - fitted[sel]
            middle = torch.where(m, resid, float("nan")).nanquantile(0.5, dim=1, keepdim=True, interpolation="midpoint")
            resid = resid - middle
            # with an odd count the middle square is 0 by construction: keep it out of the scale
            spread = resid[m].abs()
            spread = spread[spread > 0]
            limit = HUBER_K * MAD_SCALE * spread.median() if len(spread) else 0.0
            clipped = torch.where(m, evals[sel] - resid + resid.clamp(-limit, limit), 0.0)
            w = weight[sel][:, None] * m
            size = w.sum(dim=1, keepdim=True)
            size = torch.where(size > 0, size, 1.0)
            # differences within a position only: its overall eval drops out
            centered = torch.where(m, clipped - (w * clipped).sum(dim=1, keepdim=True) / size, 0.0)
            cross = block(zs * w, w, size)
            gram[phase, :64, :64] += block(w, w, size)
            gram[phase, :64, 64:] += cross
            gram[phase, 64:, :64] += cross.T
            gram[phase, 64:, 64:] += block(zs**2 * w, w, size)
            rhs[phase] += torch.cat([(w * centered).sum(dim=0), (zs * w * centered).sum(dim=0)])
            used = weight[sel] * m.any(dim=1)  # positions with at least one legal square
            visits[phase] += (used[:, None] * king[sel]).sum(dim=0)
            level[phase] += torch.stack([(used * count[sel]).sum(), used.sum()])
            explained = torch.where(m, fitted[sel] - (w * fitted[sel]).sum(dim=1, keepdim=True) / size, 0.0)
            r = w * (centered - explained)
            scores.index_add_(0, cluster[sel], torch.cat([r, zs * r], dim=1))
        meat[phase] += scores.T @ scores
    return stats


def relocation_solve(gram, rhs, ridge):
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + ridge * eye, rhs[..., None])[..., 0]


def relocation_shift(level):
    """Per phase: average piece count minus PHASE_REF."""
    ref = torch.tensor(PHASE_REF, dtype=level.dtype, device=level.device)
    has = level[:, 1] > 0
    return torch.where(has, level[:, 0] / torch.where(has, level[:, 1], 1.0) - ref, 0.0)


def relocation_emit(theta, shift):
    """(2, 64) tables at each phase's average piece count."""
    return theta[:, :64] + theta[:, 64:] * shift[:, None]


def relocation_error(gram, meat, shift, ridge):
    """Clustered standard error of the emitted table, RMS over squares weighted by how often each is tried.

    Weighting by tries rather than by where the piece usually stands makes far squares settle too."""
    tried = np.diag(gram)[:64]
    if not tried.sum() > 0:
        return np.inf
    inv = np.linalg.inv(gram + ridge * np.eye(128))
    emit_map = np.hstack([np.eye(64), shift * np.eye(64)])
    p = tried / tried.sum()
    # relocation only sees differences between squares, so a constant shift is not measured
    proj = np.eye(64) - np.outer(np.ones(64), p)
    cov = proj @ emit_map @ inv @ meat @ inv @ emit_map.T @ proj.T
    return np.sqrt(p @ np.diag(cov))


def relocation_update(stats, eg, count, sides, cluster, weight, ridge, phases):
    """Fold one batch of relocations into the running fit (stats, updated in place).

    Returns the 95% error in cp of the emitted table, worst over the given phases."""
    gram, rhs, _, visits, level = stats

    # Huber by reweighting: refit until the table settles
    table = relocation_solve(gram, rhs, ridge)
    while True:
        batch = relocation_stats(eg, count, sides, table, cluster, weight)
        update = relocation_solve(gram + batch[0], rhs + batch[1], ridge)
        shift = relocation_shift(level + batch[4])

        p = visits + batch[3]
        total_p = p.sum(dim=1, keepdim=True)
        p = p / torch.where(total_p > 0, total_p, 1.0)

        step = relocation_emit(update, shift) - relocation_emit(table, shift)
        step = step - (p * step).sum(dim=1, keepdim=True)  # constant shift is free
        table = update
        if (p * step**2).sum(dim=1).max().sqrt() <= TABLE_PRECISION / CONFIDENCE_Z:
            break

    for acc, part in zip(stats, batch):
        acc += part

    gram, _, meat = [a.cpu().numpy() for a in stats[:3]]
    shift = relocation_shift(stats[4]).cpu().numpy()
    return CONFIDENCE_Z * max(relocation_error(gram[i], meat[i], shift[i], ridge) for i in phases)


def calibrate_squares(eg, count, sides, cluster, weight, ridge, phases, eval_time, other_time):
    """Squares per position that reach a given precision fastest, from a batch where all squares were tried.

    Error**2 shrinks as 1/positions, so error**2 * seconds per batch is the time cost of any target
    precision. Each candidate count is scored by subsampling the batch already evaluated."""
    tried = sum(int(legal.sum()) for legal, _, _ in sides)
    per_eval = eval_time / max(tried, 1)
    most = max(int(legal.sum(dim=1).max()) for legal, _, _ in sides)
    candidates = [2**i for i in range(1, max(most, 2).bit_length())] + [0]
    # residuals against a fit of the whole batch, so real square differences don't count as noise
    full = [torch.zeros(shape, dtype=torch.float64, device=eg.device) for shape in RELOCATION_STAT_SHAPES]
    relocation_update(full, eg, count, sides, cluster, weight, ridge, phases)
    table = relocation_solve(full[0], full[1], ridge)

    best, best_cost = 0, np.inf
    for squares in candidates:
        trial = [(sample_squares(legal, squares), evals, piece) for legal, evals, piece in sides]
        stats = relocation_stats(eg, count, trial, table, cluster, weight)
        gram, _, meat, _, level = [a.cpu().numpy() for a in stats]
        shift = relocation_shift(torch.from_numpy(level)).numpy()
        error = max(relocation_error(gram[i], meat[i], shift[i], ridge) for i in phases)
        cost = error**2 * (other_time + sum(int(legal.sum()) for legal, _, _ in trial) * per_eval)
        if cost < best_cost:
            best, best_cost = squares, cost
    return best


def clock(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def plain_fit_error(gram, meat, occupancy, ridge):
    """95% error in cp of the square patterns that come from the weighted fit (pawn to rook), worst piece.

    Clustered sandwich covariance. Each piece's level is left out: it is one number per piece, set by
    every position, and pinned down long before the 64 squares are."""
    eye = torch.eye(PARAMS, dtype=gram.dtype, device=gram.device)
    inv = torch.linalg.inv(gram + ridge * eye)
    cov = inv @ meat @ inv

    worst = 0.0
    for name in WEIGHTED_FIT:
        i = PST_ORDER.index(name)
        sq = slice(i * 64 + 8, i * 64 + 56) if name == "PAWN" else slice(i * 64, (i + 1) * 64)
        if not occupancy[sq].sum() > 0:
            return np.inf
        p = occupancy[sq] / occupancy[sq].sum()
        proj = eye[sq, sq] - p[None, :]  # the level is set separately, from the plain fit
        worst = max(worst, float(p @ torch.diagonal(proj @ cov[sq, sq] @ proj.T)) ** 0.5)
    return CONFIDENCE_Z * worst


def stacked_forward(self, x, stack, w=None, b=None):
    """Same result as tt.Stacked.forward up to float summation order; that one runs all STACKS layers
    and keeps one per row.

    Inference only: sort rows by stack and run just the layer each group needs. For hidden_2 that is
    most of the net's cost, cut by STACKS times."""
    w = self.weight if w is None else w
    b = self.bias if b is None else b

    order = torch.argsort(stack)
    counts = torch.bincount(stack, minlength=tt.STACKS).tolist()  # one device sync per call
    xs = x[order]

    y = torch.empty((x.shape[0], self.units), dtype=x.dtype, device=x.device)
    start = 0
    for s, n in enumerate(counts):
        if n:
            y[start : start + n] = xs[start : start + n] @ w[s] + b[s]
        start += n

    out = torch.empty_like(y)
    out[order] = y
    return out


def load_model(bin_path, device):
    model = tt.NNUE()
    tt.load_bin(model, bin_path)
    model.eval()

    # The net re-quantizes hidden_1a and hidden_2 on every forward pass, about 25M weights each time,
    # which costs more than the eval itself. The weights never change here: quantize them once and make
    # the per-call pass a no-op. Exact, since the scales are powers of two (quantizing twice changes nothing).
    with torch.no_grad():
        for layer, w_scale, b_scale in ((model.hidden_1a, tt.Q_SCALE, tt.Q_SCALE), (model.hidden_2, tt.Q_W2, tt.Q_B2)):
            layer.weight.copy_(tt.fake_quant(layer.weight, w_scale))
            layer.bias.copy_(tt.fake_quant(layer.bias, b_scale))
    tt.fake_quant = lambda x, scale: x  # this process only runs inference

    for layer in (model.hidden_2, model.hidden_3, model.out):
        layer.forward = types.MethodType(stacked_forward, layer)
    return model.to(device)


def emit(tables, out):
    """Write tables into a copy of the repo's tables.h, replacing only the numbers."""

    def block(values, indent):
        rows = (values[r * 8 : (r + 1) * 8] for r in range(8))
        return "\n".join(indent + " ".join(f"{v:4d}," for v in row) for row in rows)

    pieces = ",".join(
        f"\n    {{ /* {name} */\n{block(tables[i * 64 : (i + 1) * 64], ' ' * 8)}\n    }}"
        for i, name in enumerate(PST_ORDER)
    )
    with open(os.path.join(REPO_ROOT, "tables.h"), encoding="utf-8") as f:
        text = f.read()
    for head, body in (
        (r"int SQUARE_TABLE\[\]\[64\] = \{", "\n    {}/* NONE */," + pieces),
        (r"int ENDGAME_KING_SQUARE_TABLE\[64\] = \{", "\n" + block(tables[6 * 64 :], " " * 4)),
    ):
        pattern = f"({head}).*?(\n}};)"
        text, found = re.subn(pattern, lambda m, body=body: m.group(1) + body + m.group(2), text, count=1, flags=re.S)
        if not found:
            raise ValueError(f"tables.h: no match for {head}")
    out.write(text)


def main():
    p = argparse.ArgumentParser(
        description="Fit piece-square tables to a trained net's evals, minus material; writes tables.h format."
    )
    p.add_argument("data", nargs="+", help="H5 training data file(s)")
    p.add_argument("-m", "--model", default="weights.bin", help="weights.bin")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    p.add_argument("-b", "--batch-size", type=int, default=65536, help="rows per read")
    p.add_argument("--limit", type=int, help="max positions per file, sampled as random slices")
    p.add_argument("--slice-rows", type=int, default=256, help="rows per random slice")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("--ridge", type=float, default=1.0, help="L2 penalty for the least-squares fit")
    p.add_argument("--scale", type=float, default=1.0, help="multiply all table values, e.g. 0.5 for half")
    p.add_argument("--device", help="default: cuda if available, else cpu")
    p.add_argument("--eval-chunk", type=int, default=65536, help="positions per net forward pass")
    p.add_argument("--queen-squares", type=int, help="squares per queen and position (default: calibrated, 0: all)")
    p.add_argument("--early-stop", action="store_true", help=f"stop when all tables reach {TABLE_PRECISION:g}cp (95%%)")
    p.add_argument("--king-only", action="store_true", help="debug: stop once the king tables converge")
    args = p.parse_args()

    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}", file=sys.stderr)

    model = load_model(args.model, args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)  # queen square sampling

    gram = torch.zeros((2, PARAMS, PARAMS), dtype=torch.float64, device=args.device)  # plain, balance-weighted
    rhs = torch.zeros((2, PARAMS), dtype=torch.float64, device=args.device)

    # pieces fitted by relocation: how to move them, and which phases their tables have
    queen_squares = args.queen_squares  # None: calibrate on the second batch, once startup costs are paid
    queen_step = functools.partial(queen_moves, squares=queen_squares)
    relocated = {"KING": (king_moves, (0, 1)), "QUEEN": (queen_step, (0,))}
    if args.king_only:
        del relocated["QUEEN"]
    reloc_stats = {
        name: [torch.zeros(shape, dtype=torch.float64, device=args.device) for shape in RELOCATION_STAT_SHAPES]
        for name in ("KING", "QUEEN")
    }  # each: gram, rhs, meat, visits, level; [middlegame, endgame]
    reloc_error = dict.fromkeys(relocated, np.inf)
    reloc_rows = dict.fromkeys(relocated, 0)

    # --early-stop: clustered covariance of the weighted fit, from residuals against the fit so far
    plain_meat = torch.zeros((PARAMS, PARAMS), dtype=torch.float64, device=args.device)
    plain_error = np.inf
    fitted_rows, scored_rows = 0, 0  # the first batch is fitted but not scored

    def finished():
        if args.king_only:
            return reloc_error["KING"] <= TABLE_PRECISION
        if args.early_stop:
            return plain_error <= TABLE_PRECISION and max(reloc_error.values()) <= TABLE_PRECISION
        return False

    scale = None  # outcome scale of material, fitted on the first batch

    with torch.no_grad():
        for path in args.data:
            with h5py.File(path, "r") as hf:
                data = hf["data"]
                total = data.shape[0]
                # random slices even for a full pass: the relocation steps stop early and must see the whole file
                slices = total // args.slice_rows
                k = min(slices, -(-args.limit // args.slice_rows)) if args.limit else slices
                starts = rng.choice(slices, size=k, replace=False) * args.slice_rows
                # each batch spans the file
                groups = np.array_split(starts, -(-k * args.slice_rows // args.batch_size))
                reads = (read_slices(data, np.sort(g), args.slice_rows) for g in groups)
                n, num_batches = k * args.slice_rows, len(groups)
                print(f"{path}: {n:,} rows", file=sys.stderr)
                bar = tqdm(reads, total=num_batches, unit="batch")
                warmed, mark = False, clock(args.device)
                for rows in bar:
                    keep = (tt.popcount(rows[:, 0]) == 1) & (tt.popcount(rows[:, 1]) == 1)  # one king per side
                    rows = rows[keep]
                    packed = torch.from_numpy(rows[:, :13].astype(np.int64)).to(args.device)
                    x, material = features_and_material(packed)
                    if scale is None:
                        stm = torch.where(packed[:, 12] == 1, material, -material)
                        outcome = torch.from_numpy(rows[:, 14] / 2.0).to(args.device)  # 0, 1/2, 1 for stm
                        scale = fit_outcome_scale(stm, outcome)
                    # weight by material balance, never by net eval (that biases the fit)
                    weight = balance_weight(material, scale)
                    y = net_eval(model, packed, args.eval_chunk) - material

                    # rows of a slice are correlated
                    cluster = torch.from_numpy(np.flatnonzero(keep) // args.slice_rows).to(args.device)

                    if args.early_stop and rhs[1].any():
                        # residuals against the fit from earlier batches, so this batch does not fit itself
                        eye = torch.eye(PARAMS, dtype=gram.dtype, device=gram.device)
                        fit = torch.linalg.solve(gram[1] + args.ridge * eye, rhs[1])
                        resid = y - x.double() @ fit
                        scores = torch.zeros((int(cluster.max()) + 1, PARAMS), dtype=torch.float64, device=args.device)
                        scores.index_add_(0, cluster, (weight * resid)[:, None] * x.double())
                        plain_meat += scores.T @ scores
                        scored_rows += len(rows)

                    xw = x * weight[:, None].float()
                    gram[0] += (x.T @ x).double()
                    rhs[0] += x.double().T @ y
                    gram[1] += (xw.T @ x).double()
                    rhs[1] += xw.double().T @ y
                    fitted_rows += len(rows)

                    if args.early_stop and plain_meat.any():
                        meat = plain_meat * fitted_rows / scored_rows  # scale up to all rows fitted
                        plain_error = plain_fit_error(gram[1], meat, torch.diagonal(gram[0]), args.ridge)

                    # relocated pieces: per position, fit eval differences across the squares tried
                    other_time = clock(args.device) - mark  # this batch without relocation
                    for name, (moves, phases) in relocated.items():
                        if reloc_error[name] <= TABLE_PRECISION:
                            continue
                        if name == "QUEEN" and queen_squares is None:
                            if not warmed:
                                continue
                            start = clock(args.device)
                            eg, count, sides = queen_moves(model, packed, args.eval_chunk, 0)
                            eval_time = clock(args.device) - start
                            queen_squares = calibrate_squares(
                                eg, count, sides, cluster, weight, args.ridge, phases, eval_time, other_time
                            )
                            relocated[name] = (functools.partial(queen_moves, squares=queen_squares), phases)
                            print(f"queen: {queen_squares or 'all'} squares per position", file=sys.stderr)
                        else:
                            eg, count, sides = moves(model, packed, args.eval_chunk)
                        reloc_error[name] = relocation_update(
                            reloc_stats[name], eg, count, sides, cluster, weight, args.ridge, phases
                        )
                        reloc_rows[name] += len(rows)
                        if reloc_error[name] <= TABLE_PRECISION:
                            print(f"{name.lower()} table converged after {reloc_rows[name]:,} rows", file=sys.stderr)

                    postfix = {name.lower(): f"{err:.1f}cp" for name, err in reloc_error.items()}
                    if args.early_stop:
                        postfix["plain"] = f"{plain_error:.1f}cp"
                    bar.set_postfix(postfix)
                    warmed, mark = True, clock(args.device)
                    if finished():
                        break

            if finished():
                if args.early_stop:
                    print("all tables converged, stopping early", file=sys.stderr)
                break

    for name, err in reloc_error.items():
        if err > TABLE_PRECISION:
            print(f"warning: {name.lower()} table did not converge ({err:.1f}cp)", file=sys.stderr)

    # table means stay in: engine adds tables to WEIGHT, so the mean is the net's value delta
    eye = torch.eye(PARAMS, dtype=gram.dtype, device=gram.device)
    plain, theta = torch.linalg.solve(gram + args.ridge * eye, rhs[..., None])[..., 0].cpu().numpy()

    # Queen: square pattern from relocation, which keeps the rest of the board fixed. The plain fit credits
    # a square with whatever advantage usually comes with a queen standing there (an attack on a king
    # castled short, say). Relocation only measures differences; the level is set below, as for the others.
    queen = slice(PST_ORDER.index("QUEEN") * 64, (PST_ORDER.index("QUEEN") + 1) * 64)
    q_gram, q_rhs, _, q_visits, q_level = reloc_stats["QUEEN"]
    if q_visits.sum() > 0:
        q_fit = relocation_solve(q_gram, q_rhs, args.ridge)
        theta[queen] = relocation_emit(q_fit, relocation_shift(q_level))[0].cpu().numpy()
        theta[queen] = (theta[queen] + theta[queen][np.arange(64) ^ 7]) / 2  # net is file-symmetric

    # square patterns from the weighted fit (relocation for the queen), per-piece levels from the plain fit
    occupancy = torch.diagonal(gram[0]).cpu().numpy()  # how often a piece stands on each square
    for i, name in enumerate(PST_ORDER):
        if name == "KING":
            continue
        sq = slice(i * 64 + 8, i * 64 + 56) if name == "PAWN" else slice(i * 64, (i + 1) * 64)
        occ = occupancy[sq] / occupancy[sq].sum()
        theta[sq] += occ @ plain[sq] - occ @ theta[sq]
    theta[:8] = theta[56:64] = 0.0  # no pawns on ranks 1 and 8
    king = PST_ORDER.index("KING") * 64
    k_gram, k_rhs, _, k_visits, k_level = reloc_stats["KING"]
    kings = relocation_emit(relocation_solve(k_gram, k_rhs, args.ridge), relocation_shift(k_level)).cpu().numpy()
    kings = (kings + kings[:, np.arange(64) ^ 7]) / 2  # net is file-symmetric, data mostly castles short
    # the level cancels between the two kings: center on the squares kings usually occupy
    visits = k_visits.cpu().numpy()
    seen = visits.sum(axis=1, keepdims=True)
    kings -= (kings * visits).sum(axis=1, keepdims=True) / np.where(seen > 0, seen, 1.0)
    theta[king : king + 128] = kings.ravel()
    tables = np.rint(theta * args.scale).astype(int)

    if args.output:
        with open(args.output, "w") as out:
            emit(tables, out)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        emit(tables, sys.stdout)


if __name__ == "__main__":
    main()
