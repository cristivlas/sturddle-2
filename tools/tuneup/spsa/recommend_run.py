#!/usr/bin/env python3
"""Plan the *next* SPSA run from an in-progress or finished one.

Reads spsa_state.json (current theta + history) and tuning.json (param
metadata + budget), and prints recommendations for the next run:
per-param bounds (narrow/widen/shift/keep, via recommend.py), new
init values (current theta), and SPSA c/a derived from the chosen R.

Dry-run by default. Pass --write-tuning <path> to emit a ready-to-use
tuning.json (init = current theta, bounds = recommended).

Optionally suggests an iteration-count adjustment based on the
full-run convergence ratio σ(last 25%) / σ(first 25%) over normalized
theta history -- the same metric the charts page shows in solo mode.
"""

import argparse
import json
import math
import os
import re
import sys
from copy import deepcopy

from config import TuningConfig
from recommend import (
    ParamSpec, recommend, format_recommendation, constrain_for,
)


CONV_MIN = 20  # mirror charts.tmpl threshold

# Match piece-square table params (PS_<piece>_<square> or PS_KEG_<square>);
# same regex apply.py uses to skip them from --rebalance.
_PST_RE = re.compile(r'PS_(\d+|KEG)_(\d+)$')

# Headroom over max-observed |θ| when sizing the new PST_RANGE half-width.
# PSTs are tuned as offsets from the static SQUARE_TABLE baseline; SPSA can
# only explore inside the bounds, so leave room for further movement.
PST_HEADROOM = 1.5


def _load_state(path):
    with open(path) as f:
        return json.load(f)


def _build_specs(theta, tuning):
    """Build ParamSpec list from spsa_state theta + tuning.json metadata.

    PST params (PS_<piece>_<square>) are excluded -- they share the
    PST_RANGE macro in config.h, not per-param bounds. _pst_recommendation
    emits a single shared recommendation for them.
    """
    specs = []
    for name, t_val in theta.items():
        if _PST_RE.match(name):
            continue
        param = tuning.parameters.get(name)
        if param is None:
            continue
        if param.is_normalized:
            cur_lo, cur_hi = param.original_lower, param.original_upper
            center = (t_val + 1) * (cur_hi - cur_lo) / 2 + cur_lo
        else:
            cur_lo, cur_hi = param.lower, param.upper
            center = t_val
        is_int = (param.type == 'int') if not param.is_normalized else (
            float(cur_lo).is_integer() and float(cur_hi).is_integer()
        )
        fixed_lo, fixed_hi, floor = constrain_for(cur_lo, cur_hi)
        specs.append(ParamSpec(
            name=name, center=center,
            current_lo=cur_lo, current_hi=cur_hi, is_int=is_int,
            fixed_lo=fixed_lo, fixed_hi=fixed_hi, floor=floor,
        ))
    return specs


def _pst_recommendation(theta, tuning):
    """Compute a single recommended PST_RANGE half-width covering all PST
    params' tuned θ values, with PST_HEADROOM margin.

    Returns dict { 'current_half', 'max_abs', 'driver', 'recommended_half' }
    or None if no PST params.
    """
    pst_engine = []
    current_half = None
    for name, t_val in theta.items():
        if not _PST_RE.match(name):
            continue
        param = tuning.parameters.get(name)
        if param is None:
            continue
        # PSTs are normalized; original_lower/original_upper hold the engine-space
        # ±PST_RANGE (e.g. -35, 35). All PSTs share the same range, so first one
        # found is representative.
        if param.is_normalized:
            engine_val = (t_val + 1) * (param.original_upper - param.original_lower) / 2 + param.original_lower
            if current_half is None:
                current_half = (param.original_upper - param.original_lower) / 2
        else:
            engine_val = t_val
            if current_half is None:
                current_half = (param.upper - param.lower) / 2
        pst_engine.append((name, engine_val))

    if not pst_engine:
        return None

    name_max, val_max = max(pst_engine, key=lambda kv: abs(kv[1]))
    max_abs = abs(val_max)
    recommended = max(1, math.ceil(max_abs * PST_HEADROOM))
    return {
        'current_half': current_half,
        'max_abs': max_abs,
        'driver': name_max,
        'recommended_half': recommended,
        'count': len(pst_engine),
    }


def _conv_ratio(history, name):
    """σ(last 25%) / σ(first 25%) on normalized theta. Returns None if
    too few points or first-quartile σ is zero."""
    pts = [h.get('theta', {}).get(name) for h in history]
    pts = [p for p in pts if p is not None]
    n = len(pts)
    q = n // 4
    if q < CONV_MIN:
        return None
    early = pts[:q]
    late = pts[-q:]

    def sigma(vals):
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    s0 = sigma(early)
    s1 = sigma(late)
    if s0 == 0:
        return None
    return s1 / s0


def _tstat(history, name):
    """t-stat of slope vs zero on normalized theta over the last 25% of run."""
    pts = [(h.get('iteration'), h.get('theta', {}).get(name)) for h in history]
    pts = [(i, y) for (i, y) in pts if i is not None and y is not None]
    n = len(pts)
    q = n // 4
    if q < CONV_MIN:
        return None
    pts = pts[-q:]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sxy / sxx
    sse = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
    # n = q >= CONV_MIN (20) by the early-return above, so n-2 > 0 always.
    se = math.sqrt(sse / ((len(xs) - 2) * sxx))
    if se == 0:
        return None
    return slope / se


def _summarize_run(state, names):
    """Aggregate convergence ratios + slope-significance counts across
    all tunable params. Used to recommend an iteration adjustment."""
    history = state.get('history', [])
    if not history:
        return None

    ratios = []
    sig_count = 0
    total = 0
    for name in names:
        r = _conv_ratio(history, name)
        if r is not None:
            ratios.append(r)
        t = _tstat(history, name)
        if t is not None:
            total += 1
            if abs(t) >= 2.0:
                sig_count += 1

    return {
        'n_points': len(history),
        'ratios': ratios,
        'med_ratio': sorted(ratios)[len(ratios) // 2] if ratios else None,
        'sig_frac': (sig_count / total) if total else None,
        'sig_count': sig_count,
        'total_with_slope': total,
    }


def _iter_recommendation(summary, current_iters):
    """Translate run summary into a verdict + suggested iteration count.

    Heuristic (in priority order):
      1. Conv < 0.7 AND >=30% still drifting -> tightening + still moving:
         1.5x iterations
      2. Conv < 0.7 AND <30% drifting -> tightening done, current sufficient
      3. Conv >= 0.7 AND >=30% still drifting -> not tightening but params
         haven't settled either; 1.5x iterations may let σ shake out, but
         the config likely needs adjusting too
      4. Conv >= 0.7 AND <30% drifting -> stalled; more iters won't help
    """
    if not summary or summary['med_ratio'] is None:
        return current_iters, 'no convergence data; keeping iteration count'
    med = summary['med_ratio']
    sig = summary['sig_frac'] or 0
    drift_str = f'{summary["sig_count"]}/{summary["total_with_slope"]}'
    if med < 0.7 and sig >= 0.30:
        return int(round(current_iters * 1.5)), (
            f'median Conv {med:.2f}x (tighter) and {drift_str} params still drifting -> 1.5x iterations'
        )
    if med < 0.7:
        return current_iters, (
            f'median Conv {med:.2f}x (tighter) and only {drift_str} params still drifting '
            '-> current count was sufficient'
        )
    if sig >= 0.30:
        return int(round(current_iters * 1.5)), (
            f'median Conv {med:.2f}x (holding/wider) but {drift_str} params still drifting '
            '-> 1.5x iterations may help'
        )
    return current_iters, (
        f'median Conv {med:.2f}x (holding/wider) and only {drift_str} drifting -> stalled; '
        'more iterations unlikely to help, widen ranges'
    )


def _write_tuning(out_path, tuning, rec, theta, suggested_iters):
    """Emit a fresh tuning.json reflecting the recommendation.

    init = current theta (engine space), bounds = recommended new_lo/new_hi,
    spsa.c/a from rec, spsa.budget = suggested_iters * games_per_iteration.
    """
    new_cfg = deepcopy(tuning)

    new_cfg.spsa.c = rec.c
    new_cfg.spsa.a = rec.a
    new_cfg.spsa.budget = suggested_iters * new_cfg.games_per_iteration

    by_name = {a.name: a for a in rec.actions}
    for name, p in new_cfg.parameters.items():
        action = by_name.get(name)
        if not action:
            continue
        new_lo = action.new_lo
        new_hi = action.new_hi
        # Init = current theta in engine space.
        t_val = theta.get(name)
        if t_val is None:
            continue
        if p.is_normalized:
            # Re-normalize current engine-space center against the new bounds
            # so the next run starts at the same engine value.
            engine_val = (t_val + 1) * (p.original_upper - p.original_lower) / 2 + p.original_lower
            engine_val = max(new_lo, min(new_hi, engine_val))
            p.original_lower = float(new_lo)
            p.original_upper = float(new_hi)
            if new_hi - new_lo > 0:
                p.init = 2 * (engine_val - new_lo) / (new_hi - new_lo) - 1
            else:
                p.init = 0.0
            p.lower = -1.0
            p.upper = 1.0
        else:
            p.lower = float(new_lo)
            p.upper = float(new_hi)
            p.init = max(new_lo, min(new_hi, float(t_val)))

    with open(out_path, 'w') as f:
        f.write(new_cfg.to_json())
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(
        description='Plan the next SPSA run from a current run\'s state.'
    )
    parser.add_argument('project', help='Path to SPSA project directory or tuning.json file')
    parser.add_argument('--state', default=None, help='Path to spsa_state.json (default: <project>/spsa_state.json)')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Override suggested iteration count for the next run')
    target_grp = parser.add_mutually_exclusive_group()
    target_grp.add_argument('--target-r', type=float, default=None,
                            help='Override R_target for recommendations')
    target_grp.add_argument('--target-c', type=float, default=None,
                            help='Override c; R_target derived as N^gamma / c')
    parser.add_argument('--min-pert-pct', type=float, default=5.0,
                        help='End-of-run perturbation floor as %% of narrowest inlier range '
                             '(allowed [5, 20], default 5). Higher = larger c, keeps '
                             'perturbations bigger for longer. Ignored when --target-c is given.')
    parser.add_argument('--write-tuning', default=None, metavar='PATH',
                        help='Write a fresh tuning.json to PATH (default: dry-run)')
    args = parser.parse_args()

    if not (5.0 <= args.min_pert_pct <= 20.0):
        parser.error(f'--min-pert-pct must be in [5, 20], got {args.min_pert_pct}')

    # Fail fast if --write-tuning would clobber. Better to error before any
    # output than to print a full report and bury the failure at the end.
    if args.write_tuning and os.path.exists(args.write_tuning):
        print(f'Error: refusing to overwrite existing file: {args.write_tuning}', file=sys.stderr)
        sys.exit(1)

    # Resolve project_dir + tuning.json path.
    if os.path.isfile(args.project) and args.project.endswith('.json'):
        tuning_path = args.project
        project_dir = os.path.dirname(os.path.abspath(args.project))
    else:
        project_dir = os.path.abspath(args.project)
        tuning_path = os.path.join(project_dir, 'tuning.json')
    if not os.path.isfile(tuning_path):
        print(f'Error: tuning.json not found at {tuning_path}', file=sys.stderr)
        sys.exit(1)

    state_path = args.state or os.path.join(project_dir, 'spsa_state.json')
    if not os.path.isfile(state_path):
        print(f'Error: spsa_state.json not found at {state_path}', file=sys.stderr)
        sys.exit(1)

    tuning = TuningConfig.from_json(tuning_path)
    state = _load_state(state_path)
    theta = state.get('theta', {})
    if not theta:
        print('Error: no theta in state file', file=sys.stderr)
        sys.exit(1)

    iter_count_seen = state.get('iteration', state.get('history', [{}])[-1].get('iteration', 0))
    planned_iters = tuning.max_iterations()

    # Run summary informs both the printed verdict and the auto iter count.
    # Baseline for the heuristic is iters-actually-run, not iters-planned:
    # "given what this run achieved in N iters, how many for the next run?"
    summary = _summarize_run(state, list(theta.keys()))
    if args.iterations is not None:
        suggested_iters = args.iterations
        reason = f'user override: {suggested_iters} iters'
    else:
        suggested_iters, reason = _iter_recommendation(summary, iter_count_seen)

    specs = _build_specs(theta, tuning)
    pst = _pst_recommendation(theta, tuning)
    if not specs and not pst:
        print('Error: no tunable params found', file=sys.stderr)
        sys.exit(1)

    rec = None
    if specs:
        rec = recommend(
            specs, suggested_iters, tuning.spsa.gamma,
            tuning.spsa.a / tuning.spsa.c,
            target_r=args.target_r, target_c=args.target_c,
            min_pert_pct=args.min_pert_pct,
        )

    print(f'Source run: {project_dir}')
    print(f'  iteration:          {iter_count_seen}/{planned_iters}')
    print(f'  games_per_iter:     {tuning.games_per_iteration}')
    if summary and summary['med_ratio'] is not None:
        print(f'  median Conv ratio:  {summary["med_ratio"]:.2f}x  (lower = tighter)')
    if summary and summary['total_with_slope']:
        print(f'  still drifting:     {summary["sig_count"]}/{summary["total_with_slope"]} params (|t|>=2)')
    print('')
    print(f'Suggested iterations for next run: {suggested_iters}')
    print(f'  reason: {reason}')
    print('')
    if rec:
        format_recommendation(rec, center_label='theta')
    if pst:
        print('')
        print(f'PST_RANGE recommendation ({pst["count"]} PST params, shared bound in config.h):')
        cur = pst['current_half']
        cur_str = f'-{cur:g}, {cur:g}' if cur is not None else 'unknown'
        print(f'  current:           {cur_str}')
        print(f'  max |theta|:       {pst["max_abs"]:.2f}  ({pst["driver"]})')
        rec_h = pst['recommended_half']
        print(f'  recommended:       -{rec_h}, {rec_h}  '
              f'(max-observed * {PST_HEADROOM} headroom, ceil)')
        print('  apply by editing PST_RANGE in config.h (apply.py --rebalance does not handle this)')

    if args.write_tuning:
        if not rec:
            print('Error: --write-tuning needs at least one non-PST tunable param', file=sys.stderr)
            sys.exit(1)
        out = args.write_tuning
        parent = os.path.dirname(os.path.abspath(out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        _write_tuning(out, tuning, rec, theta, suggested_iters)
        print('')
        print(f'Wrote tuning.json -> {out}')


if __name__ == '__main__':
    main()
