"""SPSA range recommendation: shared by genconfig and apply.

Picks an engine-space target range R, derives c/a, and produces per-param
suggested bounds. Two contexts:

  - genconfig --check-ranges: center=init from get_param_info(); cap=engine
    declared bounds; classifies as keep/narrow/REBUILD.
  - apply.py --rebalance:     center=post-tuning theta; cap is None (apply.py
    edits C++ source directly, so REBUILD is moot); classifies as
    keep/narrow/widen/shift.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass
class ParamSpec:
    name: str
    center: float                       # init (genconfig) or post-tuned theta (apply), engine space
    current_lo: float                   # current bounds (engine space) -- used for change labeling
    current_hi: float
    is_int: bool
    cap_lo: Optional[float] = None      # engine source cap; set => REBUILD detection enabled
    cap_hi: Optional[float] = None
    fixed_lo: Optional[float] = None    # pin new lower to this value (e.g., 1 for divisor bookend)
    fixed_hi: Optional[float] = None    # pin new upper to this value (e.g., 1 for probability ceiling)
    floor: Optional[float] = None       # never let new lower go below (e.g., 0 for non-negative)


@dataclass
class ParamAction:
    name: str
    center: float
    current_lo: float
    current_hi: float
    current_w: float
    new_lo: float
    new_hi: float
    action: str                         # 'keep' | 'narrow' | 'widen' | 'shift' | 'rebuild'
    is_int: bool


@dataclass
class Recommendation:
    iterations: int
    gamma: float
    R_target: int
    c: float
    a: float
    target_src: str
    actions: List[ParamAction]


def constrain_for(current_lo, current_hi):
    """Derive bound constraints from a param's current bounds:

      - lower == 1 or upper == 1 marks a hard semantic bookend (divisor floor /
        probability ceiling); the algorithm pins it.
      - lower >= 0 marks a non-negative-only range; the new lower never goes
        below 0.

    Returns (fixed_lo, fixed_hi, floor); each is None when the rule does
    not apply.
    """
    return (
        1 if current_lo == 1 else None,
        1 if current_hi == 1 else None,
        0 if current_lo >= 0 else None,
    )


def recommend(specs: List[ParamSpec], iterations: int, gamma: float,
              a_to_c_ratio: float, target_r: Optional[float] = None,
              target_c: Optional[float] = None,
              outlier_ratio: Optional[float] = 5.0,
              min_pert_pct: float = 5.0) -> Recommendation:
    """Compute SPSA schedule recommendation.

    outlier_ratio: params whose current width exceeds outlier_ratio * median
    width are detected as scale outliers. Their R is rescaled on a log axis
    (R_target * log(current_w) / log(median_inlier_w)) so a wide outlier
    stays slightly wider than inliers without dwarfing them -- e.g. a
    16k-wide param against 120-median inliers comes down to ~250, not 16k.
    Outliers are also excluded from the auto-derived R_target geometric
    mean. Set to None or 0 to disable.

    min_pert_pct: floor for the end-of-run perturbation as a percent of
    R_min (the narrowest param's range). Without this, c is sized so c_k
    hits the integer ±1 clamp at iter N, which drops below the noise floor
    long before that. Default 5% means c_k hits max(1, 5%·R_min) at iter N.
    Set to 0 to recover the legacy "1-unit clamp at end" behavior.
    """
    if not specs:
        raise ValueError('no params')
    if target_r is not None and target_c is not None:
        raise ValueError('target_r and target_c are mutually exclusive')

    # Outlier detection (need >=3 specs for the median to be robust -- with
    # 2 specs the median is whichever sits at index 1, so the smaller one
    # would never be flagged).
    outlier_names: Set[str] = set()
    median_inlier_w: Optional[float] = None
    if outlier_ratio and len(specs) >= 3:
        all_widths = sorted([s.current_hi - s.current_lo for s in specs if s.current_hi > s.current_lo])
        if all_widths:
            median_w = all_widths[len(all_widths) // 2]
            outlier_names = {
                s.name for s in specs
                if (s.current_hi - s.current_lo) > outlier_ratio * median_w
            }
            inlier_widths = [w for w in all_widths if w <= outlier_ratio * median_w]
            if inlier_widths:
                median_inlier_w = inlier_widths[len(inlier_widths) // 2]

    if target_c is not None:
        R_target = max(1, round((iterations ** gamma) / target_c))
        target_src = f'derived from c={target_c}'
    elif target_r is not None:
        R_target = max(1, round(target_r))
        target_src = 'user-specified'
    else:
        widths = [s.current_hi - s.current_lo for s in specs
                  if s.current_hi > s.current_lo and s.name not in outlier_names]
        if not widths:
            raise ValueError('all params have zero/negative width; cannot derive R_target')
        R_target = max(1, round(math.exp(sum(math.log(w) for w in widths) / len(widths))))
        if outlier_names:
            target_src = (f'geometric mean of {len(widths)} inliers '
                          f'(excluded {len(outlier_names)} outlier'
                          f'{"s" if len(outlier_names) != 1 else ""})')
        else:
            target_src = f'geometric mean of {len(widths)} current ranges'

    if target_c is not None:
        c = round(target_c, 4)
    else:
        inlier_widths_for_floor = [s.current_hi - s.current_lo for s in specs if s.current_hi > s.current_lo and s.name not in outlier_names]
        R_min_current = min(inlier_widths_for_floor) if inlier_widths_for_floor else R_target
        pert_floor = max(1.0, R_min_current * min_pert_pct / 100.0) if min_pert_pct else 1.0
        c = round(pert_floor * (iterations ** gamma) / R_target, 4)
    a = round(c * a_to_c_ratio, 4)
    actions = []
    for s in specs:
        if s.name in outlier_names and median_inlier_w and median_inlier_w > 1:
            # Log-scale rescale: outlier_R = R_target * log(cur_w)/log(median_inlier_w).
            # Keeps outliers slightly larger than inliers (ratio = log of the
            # current-width ratio) without preserving their full magnitude.
            cur_w = s.current_hi - s.current_lo
            R_outlier = max(1, round(R_target * math.log(cur_w) / math.log(median_inlier_w)))
            actions.append(_recommend_one(s, R_outlier))
        else:
            actions.append(_recommend_one(s, R_target))
    return Recommendation(iterations, gamma, R_target, c, a, target_src, actions)


def _apply_bookend_constraints(spec, new_lo, new_hi):
    """Apply fixed_lo / fixed_hi / floor; preserve width when possible.

    Order: fixed bookends first (semantic invariants), floor last (fallback
    for non-negative ranges). When constraints conflict (fixed_lo + fixed_hi
    + R_target > current_w, or floor + fixed_hi pinch), prefer bookend pins
    over width preservation -- the resulting range may be narrower than R.
    """
    if spec.fixed_lo is not None and new_lo != spec.fixed_lo:
        new_hi += spec.fixed_lo - new_lo
        new_lo = spec.fixed_lo
    if spec.fixed_hi is not None and new_hi != spec.fixed_hi:
        new_lo -= new_hi - spec.fixed_hi
        new_hi = spec.fixed_hi
        if spec.fixed_lo is not None and new_lo < spec.fixed_lo:
            new_lo = spec.fixed_lo
    if spec.floor is not None and new_lo < spec.floor:
        new_hi += spec.floor - new_lo
        new_lo = spec.floor
        if spec.fixed_hi is not None and new_hi > spec.fixed_hi:
            new_hi = spec.fixed_hi
    return new_lo, new_hi


def _recommend_one(spec: ParamSpec, R_target: int) -> ParamAction:
    cur_w = spec.current_hi - spec.current_lo
    has_cap = spec.cap_lo is not None and spec.cap_hi is not None
    cap_lo = spec.cap_lo if has_cap else float('-inf')
    cap_hi = spec.cap_hi if has_cap else float('+inf')
    cap_w = cap_hi - cap_lo

    if has_cap and R_target > cap_w:
        # One-sided extension: anchor at the side init is farther from.
        # Minimizes engine source diff (one bound changes) while keeping center inside.
        deficit = R_target - cap_w
        if (cap_hi - spec.center) <= (spec.center - cap_lo):
            new_lo, new_hi = cap_lo, cap_hi + deficit
        else:
            new_lo, new_hi = cap_lo - deficit, cap_hi
        new_lo, new_hi = _apply_bookend_constraints(spec, new_lo, new_hi)
        if spec.is_int:
            new_lo, new_hi = int(round(new_lo)), int(round(new_hi))
        return ParamAction(spec.name, spec.center, spec.current_lo, spec.current_hi,
                           cur_w, new_lo, new_hi, 'rebuild', spec.is_int)

    half = R_target / 2
    new_lo = spec.center - half
    new_hi = spec.center + half
    if new_lo < cap_lo:
        new_hi += cap_lo - new_lo
        new_lo = cap_lo
    if new_hi > cap_hi:
        new_lo -= new_hi - cap_hi
        new_hi = cap_hi
        new_lo = max(new_lo, cap_lo)
    new_lo, new_hi = _apply_bookend_constraints(spec, new_lo, new_hi)
    # Bookend shift can push new_hi past cap_hi when the lower side is pinned.
    # Re-clamp upper without shifting the lower (which is now at floor or fixed).
    if new_hi > cap_hi:
        new_hi = cap_hi

    if spec.is_int:
        new_lo, new_hi = int(round(new_lo)), int(round(new_hi))
    new_w = new_hi - new_lo

    if abs(new_w - cur_w) < 1e-9 and abs(new_lo - spec.current_lo) < 1e-9:
        action = 'keep'
    elif new_w < cur_w - 1e-9:
        action = 'narrow'
    elif new_w > cur_w + 1e-9:
        action = 'widen'
    else:
        action = 'shift'

    return ParamAction(spec.name, spec.center, spec.current_lo, spec.current_hi,
                       cur_w, new_lo, new_hi, action, spec.is_int)


def format_recommendation(rec: Recommendation, printer=print, center_label: str = 'init') -> None:
    """Emit the recommendation. `printer` is any callable taking a single string.

    Default is `print` (stdout). Pass `logging.info` (or a lambda wrapping it)
    to interleave correctly with logger output -- otherwise stdout buffering can
    reorder against unbuffered stderr-bound logs.
    """
    def fmt(x):
        if isinstance(x, float) and abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        if isinstance(x, float):
            return f'{x:g}'
        return str(x)

    half_str = f'{rec.R_target / 2:g}'
    printer(f'Recommended schedule for N={rec.iterations} iterations:')
    printer(f'  R_target = {rec.R_target}  ({rec.target_src})')
    printer(f'  c        = {rec.c}')
    printer(f'  a        = {rec.a}')
    printer(f'  search   = +/-{half_str} engine units around {center_label} for each param (R_target / 2)')
    printer('')
    printer('Per-param recommendations:')
    if not rec.actions:
        return

    name_w = max(len(a.name) for a in rec.actions)
    center_strs = {a.name: fmt(int(round(a.center)) if a.is_int else a.center) for a in rec.actions}
    center_w = max(len(s) for s in center_strs.values())

    verbs = {
        'keep':    'keep   ',
        'narrow':  'narrow ',
        'widen':   'widen  ',
        'shift':   'shift  ',
        'rebuild': 'REBUILD',
    }

    for a in sorted(rec.actions, key=lambda a: a.current_w):
        cs = center_strs[a.name]
        verb = verbs.get(a.action, a.action)
        if a.action == 'keep':
            printer(f'  {a.name:<{name_w}}  {center_label}={cs:<{center_w}}  {verb}  [{fmt(a.current_lo)}, {fmt(a.current_hi)}]')
        elif a.action == 'rebuild':
            printer(f'  {a.name:<{name_w}}  {center_label}={cs:<{center_w}}  {verb}  [{fmt(a.current_lo)}, {fmt(a.current_hi)}] -> [{fmt(a.new_lo)}, {fmt(a.new_hi)}]  (engine source change)')
        else:
            printer(f'  {a.name:<{name_w}}  {center_label}={cs:<{center_w}}  {verb}  [{fmt(a.current_lo)}, {fmt(a.current_hi)}] -> [{fmt(a.new_lo)}, {fmt(a.new_hi)}]')
