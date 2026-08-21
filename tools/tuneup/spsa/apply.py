#!/usr/bin/env python3
"""Apply SPSA tuning results to config.h using tuning.json for parameter metadata."""

import argparse
import json
import logging
import os
import re
import sys

from config import TuningConfig
from recommend import ParamSpec, recommend, format_recommendation, constrain_for


# Knob choices for --rebalance. The shared recommend module no longer
# carries defaults; each caller picks its own (see feedback memory).
REBALANCE_OUTLIER_RATIO = 5.0
REBALANCE_MIN_PERT_PCT = 5.0
REBALANCE_SAFETY_PAD = 1.2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Piece weight param names -> index in PIECE_VALUES array {0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}
PIECE_INDEX = {
    'PAWN': 1, 'KNIGHT': 2, 'BISHOP': 3, 'ROOK': 4, 'QUEEN': 5,
}

# Grading adjust param names: ADJUST_<pawn bucket>_<piece> -> (row, column) in GRADING_ADJUST
GRADING_ADJUST_RE = re.compile(r'ADJUST_(\d+)_(PAWN|KNIGHT|BISHOP|ROOK|QUEEN)$')


def denormalize(param, theta_val):
    """Convert a theta value to engine-space integer.

    For normalized parameters (with original_lower/original_upper), maps
    from [-1,1] to the original range.  For non-normalized parameters,
    rounds to int.
    """
    if param.is_normalized:
        lo, hi = param.original_lower, param.original_upper
        engine_val = int(round((theta_val + 1) * (hi - lo) / 2 + lo))
        logging.info(f"  {param.name}: theta={theta_val:+.4f} -> engine={engine_val} (normalized from [{lo}, {hi}])")
        return engine_val

    engine_val = int(round(theta_val))
    logging.info(f"  {param.name}: theta={theta_val} -> engine={engine_val}")
    return engine_val


def average_theta(state, window):
    """Mean theta over the last `window` iterates (history + current theta).

    Skipped iterations are excluded -- they re-record an unchanged theta and
    would double-weight stalled values. Averages per-param over entries where
    the param is present; falls back to the current value for params with no
    history occurrences.
    """
    series = [h.get('theta', {}) for h in state.get('history', []) if not h.get('skipped')]
    current = state.get('theta', {})
    series.append(current)
    series = series[-window:]
    avg = {}
    for name, cur_val in current.items():
        vals = [t[name] for t in series if name in t]
        avg[name] = sum(vals) / len(vals) if vals else cur_val
    return avg


def compute_range(value, range_pct):
    """Compute bounds as value +/- range_pct% of value."""
    half = value * range_pct / 100.0
    return round(value - half), round(value + half)


def _realign(new, old, prev):
    """Preserve column alignment when a numeric token changes width.

    If new is narrower than old, left-pad new with spaces.
    If new is wider, eat the extra characters from prev (the separator/whitespace
    preceding new). Returns (new, prev).
    """
    diff = len(new) - len(old)
    if diff < 0:
        return ' ' * -diff + new, prev
    if diff > 0:
        return new, prev[:-diff]
    return new, prev


# Global toggle. Set by main() from --dry-run; each file-writing site checks it.
DRY_RUN = False


def _to_engine(param, theta_val):
    """Silent version of denormalize for bulk computations (no per-call logging)."""
    if param is not None and getattr(param, 'is_normalized', False):
        lo, hi = param.original_lower, param.original_upper
        return int(round((theta_val + 1) * (hi - lo) / 2 + lo))
    return int(round(theta_val))


def compute_param_drifts(state, tuning):
    """Per-param drift info from full exploration history (engine space).

    Returns list of dicts {name, current, lo, hi, drift_pct} sorted by
    drift_pct descending. Empty list if no history.
    """
    history = state.get('history', [])
    current_theta = state.get('theta', {})
    if not history or not current_theta:
        return []

    drifts = []
    for name, t_current in current_theta.items():
        param = tuning.parameters.get(name)
        current = _to_engine(param, t_current)
        if current == 0:
            continue
        vals = [current]
        for h in history:
            t = h.get('theta', {}).get(name)
            if t is not None:
                vals.append(_to_engine(param, t))
        lo, hi = min(vals), max(vals)
        swing = max(abs(current - lo), abs(current - hi))
        drifts.append({
            'name': name,
            'current': current,
            'lo': lo,
            'hi': hi,
            'drift_pct': 100.0 * swing / abs(current),
        })
    drifts.sort(key=lambda d: d['drift_pct'], reverse=True)
    return drifts


def compute_min_safe_range(state, tuning):
    """Smallest --range %% that covers every explored value of every param.

    Returns (pct, driving_param) or (None, None) if no history.
    """
    drifts = compute_param_drifts(state, tuning)
    if not drifts:
        return None, None
    top = drifts[0]
    return top['drift_pct'], top['name']


def update_config(config_file, engine_values, finalize=False, range_pct=None, bounds_map=None):
    """Patch DECLARE_PARAM/DECLARE_VALUE/DECLARE_NORMAL lines in config.h.

    If finalize is True, also converts DECLARE_PARAM/DECLARE_NORMAL back to
    DECLARE_VALUE for the tuned parameters.

    Bounds adjustment precedence: bounds_map[name] (per-param explicit) >
    range_pct (uniform percentage) > leave unchanged.

    Returns (updated, found) where:
        updated: set of parameter names that were changed
        found: set of parameter names that were found in config.h
    """
    with open(config_file, 'r') as f:
        lines = f.readlines()

    updated = set()
    found = set()
    finalized = set()
    updated_lines = []

    for line in lines:
        original_line = line
        for name, value in engine_values.items():
            for macro in ('DECLARE_VALUE', 'DECLARE_PARAM', 'DECLARE_NORMAL'):
                pattern = re.compile(
                    rf'({macro}\s*\(\s*{re.escape(name)}\s*,\s*)(-?\d+)(\s*,\s*)(-?\d+)(\s*,\s*)(-?\d+)(\s*\))'
                )
                match = pattern.search(line)
                if match:
                    found.add(name)
                    before, old_val, sep1, old_lo, sep2, old_hi, close = match.groups()

                    # Finalize: rename the macro without padding compensation,
                    # so the line looks the same as any pre-existing DECLARE_VALUE.
                    if finalize and macro != 'DECLARE_VALUE':
                        before = before.replace(macro + '(', 'DECLARE_VALUE(', 1)
                        finalized.add(name)

                    new_val = str(value)
                    new_val, before = _realign(new_val, old_val, before)

                    # Adjust bounds: per-param map takes precedence over uniform pct
                    new_lo, new_hi = old_lo, old_hi
                    if bounds_map and name in bounds_map:
                        lo, hi = bounds_map[name]
                        new_lo, sep1 = _realign(str(lo), old_lo, sep1)
                        new_hi, sep2 = _realign(str(hi), old_hi, sep2)
                    elif range_pct is not None:
                        lo, hi = compute_range(value, range_pct)
                        new_lo, sep1 = _realign(str(lo), old_lo, sep1)
                        new_hi, sep2 = _realign(str(hi), old_hi, sep2)

                    line = line[:match.start()] + before + new_val + sep1 + new_lo + sep2 + new_hi + close + line[match.end():]
                    if line != original_line:
                        updated.add(name)
                        logging.info(f"Updated: {original_line.strip()} -> {line.strip()}")

                    break
        updated_lines.append(line)

    if (updated or finalized) and not DRY_RUN:
        with open(config_file, 'w') as f:
            f.writelines(updated_lines)
        if updated:
            logging.info(f"Patched {len(updated)} parameter(s) in {config_file}")
        if finalized:
            logging.info(f"Finalized {len(finalized)} parameter(s) to DECLARE_VALUE")
    else:
        logging.info(f"No changes to {config_file}")

    return updated, found


def _read_eval_piece_grading(header_dir):
    """Read the EVAL_PIECE_GRADING setting from common.h.

    Returns True/False if found, None if the file or define is missing.
    """
    common_h = os.path.join(header_dir, 'common.h')
    if not os.path.exists(common_h):
        return None
    with open(common_h, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            m = re.match(r'#define\s+EVAL_PIECE_GRADING\s+(\w+)', line)
            if m:
                return m.group(1).lower() not in ('0', 'false')
    return None


def update_piece_values(header_file, engine_values):
    """Patch PIECE_VALUES and GRADING_ADJUST array macros in chess.h.

    Recognises piece weight names (PAWN, KNIGHT, ...) and grading adjustment
    names (ADJUST_<bucket>_<piece>) and updates the corresponding element of
    the ``#define PIECE_VALUES { ... }`` line or the ``#define GRADING_ADJUST``
    multi-line macro (one pawn-bucket row per line) matching the active build
    configuration.

    When PIECE_VALUES has conditional definitions guarded by
    ``#if EVAL_PIECE_GRADING``, only the branch matching the current
    setting in common.h is patched.

    Returns (updated, found) sets of parameter names.
    """
    piece_updates = {}   # array index -> (name, value)
    adjust_updates = {}  # (bucket, column) -> (name, value)

    for name, value in engine_values.items():
        if name in PIECE_INDEX:
            piece_updates[PIECE_INDEX[name]] = (name, value)
        else:
            m = GRADING_ADJUST_RE.match(name)
            if m:
                adjust_updates[(int(m.group(1)), PIECE_INDEX[m.group(2)])] = (name, value)

    if not piece_updates and not adjust_updates:
        return set(), set()

    with open(header_file, 'r', encoding='utf-8', newline='') as f:
        lines = f.readlines()

    # Determine which PIECE_VALUES branch to patch
    grading = _read_eval_piece_grading(os.path.dirname(header_file) or '.')
    if grading is not None:
        logging.info(f"EVAL_PIECE_GRADING = {grading}")

    # Weights for the GRADING_ADJUST effective-value comments, post-patch
    adjust_weights = _parse_piece_values(lines, grading)
    if adjust_weights:
        for idx, (name, val) in piece_updates.items():
            if idx < len(adjust_weights):
                adjust_weights[idx] = int(val)

    found = set()
    updated = set()
    define_re = re.compile(
        r'(#define\s+PIECE_VALUES\s*\{\s*)'
        r'([^}]+)'
        r'(\s*\})'
    )
    # Track preprocessor context to identify which branch we're in
    in_grading_if = False   # inside #if EVAL_PIECE_GRADING block
    in_else = False         # inside the #else branch
    adjust_row = -1         # current GRADING_ADJUST row, -1 when outside the macro

    result_lines = []
    for line in lines:
        stripped_line = line.lstrip()

        # Track #if EVAL_PIECE_GRADING / #else / #endif
        if re.match(r'#if\s+EVAL_PIECE_GRADING\b', stripped_line):
            in_grading_if = True
            in_else = False
        elif in_grading_if and stripped_line.startswith('#else'):
            in_else = True
        elif in_grading_if and stripped_line.startswith('#endif'):
            in_grading_if = False
            in_else = False

        # Patch PIECE_VALUES -- skip the wrong branch when we know which is active
        if piece_updates and define_re.search(line):
            skip = False
            if grading is not None and in_grading_if:
                # In the #if block: patch only if grading matches
                if in_else:
                    skip = grading        # skip #else branch when grading is true
                else:
                    skip = not grading    # skip #if branch when grading is false

            if not skip:
                line = _patch_array_line(define_re, line, 'PIECE_VALUES',
                                         piece_updates, found, updated)

        # Patch GRADING_ADJUST (multi-line macro, one pawn-bucket row per line)
        if adjust_updates:
            if re.match(r'#define\s+GRADING_ADJUST\b', stripped_line):
                adjust_row = 0
            elif adjust_row >= 0:
                m = re.search(r'\{([^}]*)\}', line)
                if m:
                    line = _patch_adjust_row(line, m, adjust_row, adjust_updates, found, updated, adjust_weights)
                    adjust_row += 1
                else:
                    adjust_row = -1  # closing brace or unexpected line ends the macro

        result_lines.append(line)

    if updated and not DRY_RUN:
        with open(header_file, 'w', encoding='utf-8', newline='') as f:
            f.writelines(result_lines)
        logging.info(f"Patched {len(updated)} piece value(s) in {header_file}")
    else:
        logging.info(f"No piece-value changes in {header_file}")

    return updated, found


def _parse_piece_values(lines, grading):
    """Graded-branch PIECE_VALUES as a list of 7 ints (for the effective-value comments)."""
    in_if = in_else = False
    fallback = None
    for line in lines:
        s = line.lstrip()
        if re.match(r'#if\s+EVAL_PIECE_GRADING\b', s):
            in_if, in_else = True, False
        elif in_if and s.startswith('#else'):
            in_else = True
        elif in_if and s.startswith('#endif'):
            in_if = in_else = False
        m = re.match(r'#define\s+PIECE_VALUES\s*\{([^}]+)\}', s)
        if m:
            vals = [int(x) for x in m.group(1).split(',')]
            if in_if and not in_else and grading is not False:
                return vals
            if (in_else and grading is False) or not in_if:
                fallback = vals
    return fallback


def _patch_adjust_row(line, m, row, updates, found, updated, weights):
    """Regenerate one ``{ ... }`` row of GRADING_ADJUST: patch updated elements,
    normalize column widths, refresh the effective-piece-value comment (WEIGHT + adjust)."""
    values = [int(tok) for tok in m.group(1).split(',')]
    for (bucket, col), (name, val) in updates.items():
        if bucket != row:
            continue
        found.add(name)
        if col < len(values) and values[col] != int(val):
            logging.info(f"Updated {name}: {values[col]} -> {val} in GRADING_ADJUST row {row}")
            values[col] = int(val)
            updated.add(name)
    indent = line[:len(line) - len(line.lstrip())]
    vals_txt = ', '.join([str(values[0])] + [f"{v:>4}" for v in values[1:6]] + [str(values[6])])
    label = '0-4' if row == 0 else f"{4 * row + 1}-{4 * row + 4}"
    comment = f" /* {label:>5} pawns: " + ', '.join(f"{values[i] + weights[i]:>4}" for i in range(1, 6)) + " */" if weights else ""
    return f"{indent}{{ {vals_txt} }},{comment} \\\n"


def _patch_array_line(pattern, line, macro_name, updates, found, updated):
    """Replace individual elements in a ``#define MACRO { v0, v1, ... }`` line."""
    m = pattern.search(line)
    if not m:
        return line
    prefix, body, suffix = m.group(1), m.group(2), m.group(3)
    values = body.split(',')
    for idx, (name, val) in updates.items():
        found.add(name)
        if idx < len(values):
            old_tok = values[idx]
            stripped = old_tok.strip()
            new_val = str(val)
            if stripped != new_val:
                values[idx] = old_tok.replace(stripped, new_val, 1)
                updated.add(name)
                logging.info(f"Updated {name}: {stripped} -> {new_val} in {macro_name}")
    return line[:m.start()] + prefix + ','.join(values) + suffix + line[m.end():]


def _patch_table_row(line, base_square, updates, found, updated):
    """Patch individual values in a comma-separated row of a PST array.

    base_square: the square index of the first value on this line
    updates: {square: (name, value)}
    """
    parts = line.split(',')
    modified = False
    for i, part in enumerate(parts):
        sq = base_square + i
        if sq not in updates:
            continue
        name, new_val = updates[sq]
        found.add(name)
        m = re.search(r'(-?\d+)', part)
        if not m:
            continue
        old_str = m.group(1)
        new_str = str(new_val)
        if old_str == new_str:
            continue
        # Adjust leading whitespace to preserve column width
        prefix = part[:m.start(1)]
        suffix = part[m.end(1):]
        diff = len(new_str) - len(old_str)
        if diff > 0 and len(prefix.lstrip('\n')) >= diff:
            prefix = prefix[:len(prefix) - diff]
        elif diff < 0:
            prefix = ' ' * (-diff) + prefix
        parts[i] = prefix + new_str + suffix
        updated.add(name)
        logging.info(f"Updated {name}: {old_str} -> {new_str}")
        modified = True
    return ','.join(parts) if modified else line


_PST_NUM_RE = re.compile(r'-?\d+')
# A "PST row" line: indent, comma-separated ints, optional trailing comma, EOL.
_PST_ROW_RE = re.compile(r'^(\s*)(-?\d+(?:\s*,\s*-?\d+)*)(,?)\s*$')


def update_tables(tables_file, pst_values):
    """Patch SQUARE_TABLE and ENDGAME_KING_SQUARE_TABLE arrays in tables.h.

    pst_values: dict of name -> (piece_key, square, value)
        piece_key: '1'-'6' for SQUARE_TABLE, 'KEG' for ENDGAME_KING_SQUARE_TABLE
    Returns (updated, found) sets of parameter names.
    """
    # Group updates by table
    sq_updates = {}   # piece_type_int -> {square: (name, value)}
    keg_updates = {}  # {square: (name, value)}
    for name, (piece_key, square, value) in pst_values.items():
        if piece_key == 'KEG':
            keg_updates[square] = (name, value)
        else:
            sq_updates.setdefault(int(piece_key), {})[square] = (name, value)

    with open(tables_file, 'r', encoding='utf-8', newline='') as f:
        lines = f.readlines()

    found = set()
    updated = set()
    section = None     # 'sq' or 'keg'
    piece_idx = -1     # current sub-array index in SQUARE_TABLE
    square_idx = 0     # current square within piece/table
    in_piece = False   # inside a piece sub-array

    result = []
    for line in lines:
        s = line.strip()

        # Detect section starts
        if section is None:
            if 'SQUARE_TABLE' in line and '[]' in line and '{' in s:
                section = 'sq'
                piece_idx = -1
                in_piece = False
            elif 'ENDGAME_KING_SQUARE_TABLE' in line and '{' in s:
                section = 'keg'
                square_idx = 0
            result.append(line)
            continue

        if section == 'sq':
            if not in_piece:
                if '{' in s:
                    piece_idx += 1
                    square_idx = 0
                    in_piece = True
                    # Self-closing sub-array like {}/* NONE */
                    if '}' in s:
                        in_piece = False
                elif s.startswith('};'):
                    section = None
            else:
                if s.startswith('}'):
                    in_piece = False
                elif re.search(r'-?\d', s):
                    # Data row -- patch if we have updates for this piece
                    piece_updates = sq_updates.get(piece_idx, {})
                    if piece_updates:
                        line = _patch_table_row(line, square_idx, piece_updates, found, updated)
                    square_idx += len(re.findall(r'-?\d+', s))

        elif section == 'keg':
            if '}' in s and ';' in s:
                section = None
            elif re.search(r'-?\d', s):
                if keg_updates:
                    line = _patch_table_row(line, square_idx, keg_updates, found, updated)
                square_idx += len(re.findall(r'-?\d+', s))

        result.append(line)

    if updated:
        result = _reformat_pst_tables(result)

    if updated and not DRY_RUN:
        with open(tables_file, 'w', encoding='utf-8', newline='') as f:
            f.writelines(result)
        logging.info(f"Patched {len(updated)} PST value(s) in {tables_file}")
    elif not updated:
        logging.info(f"No PST changes in {tables_file}")

    return updated, found


def _reformat_pst_tables(lines):
    """Right-align values in each maximal run of numeric-only rows."""
    from itertools import groupby
    out = []
    for is_row, grp in groupby(lines, key=lambda l: bool(_PST_ROW_RE.match(l))):
        rows = list(grp)
        if not is_row:
            out.extend(rows); continue
        ms = [_PST_ROW_RE.match(r) for r in rows]
        w = max(len(t) for m in ms for t in _PST_NUM_RE.findall(m.group(2)))
        indent = min((m.group(1) for m in ms), key=len)
        out.extend(f"{indent}{', '.join(t.rjust(w) for t in _PST_NUM_RE.findall(m.group(2)))}{m.group(3)}\n" for m in ms)
    return out


def adjust_piece_param_bounds(config_file, engine_values, range_pct=None, bounds_map=None):
    """Adjust Config::Param bounds for piece value parameters in config.h.

    Bounds source precedence: bounds_map[name] > range_pct > skip param.
    """
    piece_names = [n for n in engine_values if n in PIECE_INDEX]
    if not piece_names:
        return set()

    with open(config_file, 'r') as f:
        lines = f.readlines()

    adjusted = set()
    result_lines = []
    for line in lines:
        original_line = line
        for name in piece_names:
            if bounds_map and name in bounds_map:
                lo, hi = bounds_map[name]
            elif range_pct is not None:
                lo, hi = compute_range(engine_values[name], range_pct)
            else:
                continue
            pat = re.compile(
                r'("' + re.escape(name) + r'"\s*,\s*Config::Param\{\s*[^,]+,\s*)(\d+)(\s*,\s*)(\d+)(\s*,)'
            )
            match = pat.search(line)
            if match:
                prefix, old_lo, sep, old_hi, trailing = match.groups()
                new_lo, prefix = _realign(str(lo), old_lo, prefix)
                new_hi, sep = _realign(str(hi), old_hi, sep)
                line = line[:match.start()] + prefix + new_lo + sep + new_hi + trailing + line[match.end():]
                if line != original_line:
                    adjusted.add(name)
                    logging.info(f"Adjusted bounds: {original_line.strip()} -> {line.strip()}")
                break
        result_lines.append(line)

    if adjusted and not DRY_RUN:
        with open(config_file, 'w') as f:
            f.writelines(result_lines)
        logging.info(f"Adjusted {len(adjusted)} piece param bound(s) in {config_file}")

    return adjusted


_PST_RE = re.compile(r'PS_(\d+|KEG)_(\d+)$')


def _build_rebalance_specs(theta, tuning):
    """Build ParamSpec list from spsa_state theta + tuning.json metadata.

    Returns (specs, skipped_pst). PST params are skipped because their bounds
    live in a single shared PST_RANGE macro (config.h), not per-param.
    """
    specs = []
    skipped_pst = []
    for name, t_val in theta.items():
        if _PST_RE.match(name):
            skipped_pst.append(name)
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
        specs.append(ParamSpec(name=name, center=center,
                               current_lo=cur_lo, current_hi=cur_hi, is_int=is_int,
                               fixed_lo=fixed_lo, fixed_hi=fixed_hi, floor=floor))
    return specs, skipped_pst


def main():
    parser = argparse.ArgumentParser(
        description='Apply SPSA tuning results to config.h (using tuning.json for parameter metadata).'
    )
    parser.add_argument('project', help='Path to SPSA project directory or tuning.json file')
    parser.add_argument('--state', default=None, help='Path to spsa_state.json (default: <project>/spsa_state.json)')
    parser.add_argument('--config', default='config.h', help='Path to config.h (default: config.h)')
    parser.add_argument('--header', default=None, help='Path to chess.h for piece values (default: chess.h next to config.h)')
    parser.add_argument('--tables', default=None, help='Path to tables.h for piece-square tables (default: tables.h next to config.h)')
    parser.add_argument('--finalize', action='store_true', help='Convert DECLARE_PARAM/DECLARE_NORMAL to DECLARE_VALUE')
    parser.add_argument('--range', type=float, default=None, metavar='PCT', dest='range_pct',
                        help='Adjust parameter bounds to value +/- PCT%% of value (e.g. 20 to narrow, 150 to widen)')
    parser.add_argument('--auto-range', action='store_true',
                        help='Use the minimum safe --range computed from exploration history')
    parser.add_argument('--rebalance', action='store_true',
                        help='Compute per-param range recommendations from tuned theta and apply to config.h')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Iterations for --rebalance schedule (default: tuning.json budget / games_per_iteration)')
    target_grp = parser.add_mutually_exclusive_group()
    target_grp.add_argument('--target-r', type=float, default=None,
                            help='Override R_target for --rebalance')
    target_grp.add_argument('--target-c', type=float, default=None,
                            help='Override c for --rebalance; R_target derived as N^gamma / c')
    parser.add_argument('--window', type=int, default=None, metavar='N',
                        help='Average theta over the last N non-skipped iterates, current theta included '
                             '(N=1 is the final value; omit to use the final value without averaging)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and report changes without writing any files')
    args = parser.parse_args()

    if args.range_pct is not None and args.range_pct <= 0:
        parser.error('--range must be > 0')
    if args.window is not None and args.window < 1:
        parser.error('--window must be >= 1')
    if args.auto_range and args.range_pct is not None:
        parser.error('--range and --auto-range are mutually exclusive')
    if args.rebalance and (args.range_pct is not None or args.auto_range):
        parser.error('--rebalance is mutually exclusive with --range and --auto-range')
    if (args.target_r is not None or args.target_c is not None) and not args.rebalance:
        parser.error('--target-r / --target-c require --rebalance')

    global DRY_RUN
    DRY_RUN = args.dry_run

    # Accept either a project directory or a tuning.json file directly
    if os.path.isfile(args.project) and args.project.endswith('.json'):
        tuning_path = args.project
        project_dir = os.path.dirname(tuning_path) or '.'
    elif os.path.isdir(args.project):
        project_dir = args.project
        tuning_path = os.path.join(project_dir, 'tuning.json')
    else:
        logging.error(f"Not a valid project directory or tuning.json: {args.project}")
        sys.exit(1)

    # Load tuning config for parameter metadata
    if not os.path.exists(tuning_path):
        logging.error(f"Tuning config not found: {tuning_path}")
        sys.exit(1)

    tuning = TuningConfig.from_json(tuning_path)
    logging.info(f"Loaded tuning config: {len(tuning.parameters)} parameter(s)")

    # Load SPSA state
    state_path = args.state or os.path.join(project_dir, 'spsa_state.json')
    if not os.path.exists(state_path):
        logging.error(f"State file not found: {state_path}")
        sys.exit(1)

    logging.info(f"Loading state from {state_path}")
    with open(state_path) as f:
        state = json.load(f)

    theta = state.get('theta', {})
    if not theta:
        logging.error("No theta found in state file")
        sys.exit(1)

    iteration = state.get('iteration', '?')
    logging.info(f"State at iteration {iteration}, {len(theta)} parameter(s)")

    if args.window is not None:
        n_avail = sum(1 for h in state.get('history', []) if not h.get('skipped')) + 1
        if args.window > n_avail:
            logging.warning(f"--window {args.window} exceeds {n_avail} available iterate(s); using all")
        theta = average_theta(state, args.window)
        # Recenter state's theta so drift / --auto-range computations are
        # consistent with the averaged values applied below.
        state['theta'] = theta
        logging.info(f"Averaged theta over last {min(args.window, n_avail)} non-skipped iterate(s)")

    # Report the minimum safe --range based on full exploration history,
    # along with per-param drift sorted by swing percentage.
    drifts = compute_param_drifts(state, tuning)
    if drifts:
        logging.info("Per-param drift (sorted by swing %%):")
        for d in drifts:
            logging.info(f"  {d['name']}: current={d['current']}, range=[{d['lo']}, {d['hi']}], drift={d['drift_pct']:.1f}%")
        top = drifts[0]
        logging.info(f"Minimum safe --range: {top['drift_pct']:.1f}% (driven by {top['name']})")
    if args.auto_range:
        if not drifts:
            parser.error('--auto-range requested but no exploration history available')
        args.range_pct = drifts[0]['drift_pct']
        logging.info(f"Using --auto-range = {args.range_pct:.1f}%")

    # Compute --rebalance recommendations (per-param bounds)
    bounds_map = None
    if args.rebalance:
        specs, skipped_pst = _build_rebalance_specs(theta, tuning)
        if skipped_pst:
            logging.warning(f"--rebalance skipped {len(skipped_pst)} PST param(s); "
                            f"PST bounds use shared PST_RANGE macro -- adjust manually in config.h")
        if not specs:
            logging.error("No params eligible for --rebalance")
            sys.exit(1)
        iterations = args.iterations if args.iterations is not None else tuning.max_iterations()
        rec = recommend(specs, iterations, tuning.spsa.gamma,
                        tuning.spsa.a / tuning.spsa.c,
                        target_r=args.target_r, target_c=args.target_c,
                        outlier_ratio=REBALANCE_OUTLIER_RATIO,
                        min_pert_pct=REBALANCE_MIN_PERT_PCT,
                        safety_pad=REBALANCE_SAFETY_PAD)
        format_recommendation(rec, center_label='value')
        sys.stdout.flush()  # ensure stdout table lands before subsequent stderr-bound logs
        # 'keep' actions don't enter the bounds_map -- current bounds are already correct.
        bounds_map = {a.name: (a.new_lo, a.new_hi) for a in rec.actions if a.action != 'keep'}

    # Convert all theta values to engine-space integers
    logging.info(f"Denormalizing {len(theta)} parameter(s):")
    engine_values = {}
    for name, val in theta.items():
        param = tuning.parameters.get(name)
        if param:
            engine_values[name] = denormalize(param, val)
        else:
            engine_val = int(round(val))
            logging.info(f"  {name}: theta={val} -> engine={engine_val} (not in tuning config)")
            engine_values[name] = engine_val

    # Separate PST params (PS_<piece>_<square>, PS_KEG_<square>) from config.h params
    pst_values = {}   # name -> (piece_key, square, value)
    config_values = {}
    for name, val in engine_values.items():
        m = _PST_RE.match(name)
        if m:
            pst_values[name] = (m.group(1), int(m.group(2)), val)
        else:
            config_values[name] = val

    if pst_values:
        logging.info(f"{len(pst_values)} PST parameter(s), {len(config_values)} config parameter(s)")

    # Patch config.h (DECLARE_PARAM / DECLARE_VALUE / DECLARE_NORMAL)
    updated, found = update_config(args.config, config_values, finalize=args.finalize,
                                    range_pct=args.range_pct, bounds_map=bounds_map)

    # Patch piece values in chess.h (PIECE_VALUES / ENDGAME_ADJUST macros)
    not_in_config = {n: v for n, v in engine_values.items() if n not in found}
    piece_candidates = {n: v for n, v in not_in_config.items() if n in PIECE_INDEX or GRADING_ADJUST_RE.match(n)}

    if piece_candidates:
        header_path = args.header or os.path.join(os.path.dirname(args.config) or '.', 'chess.h')
        if os.path.exists(header_path):
            pv_updated, pv_found = update_piece_values(header_path, piece_candidates)
            updated |= pv_updated
            found |= pv_found
        else:
            logging.warning(f"Header file not found for piece values: {header_path}")

    # Adjust Config::Param bounds for piece values in config.h
    if (args.range_pct is not None or bounds_map) and piece_candidates:
        adjust_piece_param_bounds(args.config, piece_candidates,
                                   range_pct=args.range_pct, bounds_map=bounds_map)

    # Patch piece-square tables in tables.h
    if pst_values:
        tables_path = args.tables or os.path.join(os.path.dirname(args.config) or '.', 'tables.h')
        if os.path.exists(tables_path):
            pst_updated, pst_found = update_tables(tables_path, pst_values)
            updated |= pst_updated
            found |= pst_found
        else:
            logging.warning(f"Tables file not found for PST values: {tables_path}")

    # Report params that match current values (no change needed)
    unchanged = {name: engine_values[name] for name in found if name not in updated}
    if unchanged:
        logging.info(f"{len(unchanged)} parameter(s) already at target value:")
        for name, val in sorted(unchanged.items()):
            print(f"  {name} = {val}")

    # Report params not found in any file
    not_found = {name: engine_values[name] for name in engine_values if name not in found}
    if not_found:
        logging.info(f"{len(not_found)} parameter(s) not found:")
        for name, val in sorted(not_found.items()):
            print(f"  {name} = {val}")


if __name__ == '__main__':
    main()
