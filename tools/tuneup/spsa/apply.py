#!/usr/bin/env python3
"""Apply SPSA tuning results to config.h using tuning.json for parameter metadata."""

import argparse
import json
import logging
import os
import re
import sys

from config import TuningConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Piece weight param names -> index in PIECE_VALUES array {0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}
PIECE_INDEX = {
    'PAWN': 1, 'KNIGHT': 2, 'BISHOP': 3, 'ROOK': 4, 'QUEEN': 5,
}

# Endgame adjust param names -> index in ENDGAME_ADJUST array
ENDGAME_ADJUST_INDEX = {
    'ENDGAME_PAWN_ADJUST': 1, 'ENDGAME_KNIGHT_ADJUST': 2, 'ENDGAME_BISHOP_ADJUST': 3,
    'ENDGAME_ROOK_ADJUST': 4, 'ENDGAME_QUEEN_ADJUST': 5,
}


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


def compute_range(value, range_pct):
    """Compute bounds as value +/- range_pct% of value."""
    half = value * range_pct / 100.0
    return round(value - half), round(value + half)


def update_config(config_file, engine_values, finalize=False, range_pct=None):
    """Patch DECLARE_PARAM/DECLARE_VALUE/DECLARE_NORMAL lines in config.h.

    If finalize is True, also converts DECLARE_PARAM/DECLARE_NORMAL back to
    DECLARE_VALUE for the tuned parameters.

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

                    # Finalize macro first so alignment accounts for name change
                    if finalize and macro != 'DECLARE_VALUE':
                        pad = ' ' * (len(macro) - len('DECLARE_VALUE'))
                        before = before.replace(macro + '(', 'DECLARE_VALUE(' + pad, 1)
                        finalized.add(name)

                    new_val = str(value)

                    # Preserve column alignment for value
                    if len(new_val) < len(old_val):
                        new_val = ' ' * (len(old_val) - len(new_val)) + new_val
                    elif len(new_val) > len(old_val):
                        before = before[:-(len(new_val) - len(old_val))]

                    # Adjust bounds if requested
                    new_lo, new_hi = old_lo, old_hi
                    if range_pct is not None:
                        lo, hi = compute_range(value, range_pct)
                        new_lo, new_hi = str(lo), str(hi)
                        if len(new_lo) < len(old_lo):
                            new_lo = ' ' * (len(old_lo) - len(new_lo)) + new_lo
                        elif len(new_lo) > len(old_lo):
                            sep1 = sep1[:-(len(new_lo) - len(old_lo))]
                        if len(new_hi) < len(old_hi):
                            new_hi = ' ' * (len(old_hi) - len(new_hi)) + new_hi
                        elif len(new_hi) > len(old_hi):
                            sep2 = sep2[:-(len(new_hi) - len(old_hi))]

                    line = line[:match.start()] + before + new_val + sep1 + new_lo + sep2 + new_hi + close + line[match.end():]
                    if line != original_line:
                        updated.add(name)
                        logging.info(f"Updated: {original_line.strip()} -> {line.strip()}")

                    break
        updated_lines.append(line)

    if updated or finalized:
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
    """Patch PIECE_VALUES and ENDGAME_ADJUST array macros in chess.h.

    Recognises piece weight names (PAWN, KNIGHT, ...) and endgame adjustment
    names (ENDGAME_PAWN_ADJUST, ...) and updates the corresponding element
    inside the ``#define PIECE_VALUES { ... }`` or ``#define ENDGAME_ADJUST { ... }``
    line that matches the active build configuration.

    When PIECE_VALUES has conditional definitions guarded by
    ``#if EVAL_PIECE_GRADING``, only the branch matching the current
    setting in common.h is patched.

    Returns (updated, found) sets of parameter names.
    """
    piece_updates = {}   # array index -> (name, value)
    adjust_updates = {}

    for name, value in engine_values.items():
        if name in PIECE_INDEX:
            piece_updates[PIECE_INDEX[name]] = (name, value)
        elif name in ENDGAME_ADJUST_INDEX:
            adjust_updates[ENDGAME_ADJUST_INDEX[name]] = (name, value)

    if not piece_updates and not adjust_updates:
        return set(), set()

    with open(header_file, 'r', encoding='utf-8', newline='') as f:
        lines = f.readlines()

    # Determine which PIECE_VALUES branch to patch
    grading = _read_eval_piece_grading(os.path.dirname(header_file) or '.')
    if grading is not None:
        logging.info(f"EVAL_PIECE_GRADING = {grading}")

    found = set()
    updated = set()
    define_re = re.compile(
        r'(#define\s+PIECE_VALUES\s*\{\s*)'
        r'([^}]+)'
        r'(\s*\})'
    )
    adjust_re = re.compile(
        r'(#define\s+ENDGAME_ADJUST\s*\{\s*)'
        r'([^}]+)'
        r'(\s*\})'
    )

    # Track preprocessor context to identify which branch we're in
    in_grading_if = False   # inside #if EVAL_PIECE_GRADING block
    in_else = False         # inside the #else branch

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

        # Patch ENDGAME_ADJUST (unconditional -- only one definition)
        if adjust_updates and adjust_re.search(line):
            line = _patch_array_line(adjust_re, line, 'ENDGAME_ADJUST',
                                     adjust_updates, found, updated)

        result_lines.append(line)

    if updated:
        with open(header_file, 'w', encoding='utf-8', newline='') as f:
            f.writelines(result_lines)
        logging.info(f"Patched {len(updated)} piece value(s) in {header_file}")
    else:
        logging.info(f"No piece-value changes in {header_file}")

    return updated, found


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
        with open(tables_file, 'w', encoding='utf-8', newline='') as f:
            f.writelines(result)
        logging.info(f"Patched {len(updated)} PST value(s) in {tables_file}")
    else:
        logging.info(f"No PST changes in {tables_file}")

    return updated, found


def adjust_piece_param_bounds(config_file, engine_values, range_pct):
    """Adjust Config::Param bounds for piece value parameters in config.h."""
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
            pat = re.compile(
                r'("' + re.escape(name) + r'"\s*,\s*Config::Param\{\s*[^,]+,\s*)(\d+)(\s*,\s*)(\d+)(\s*,)'
            )
            match = pat.search(line)
            if match:
                prefix, old_lo, sep, old_hi, trailing = match.groups()
                lo, hi = compute_range(engine_values[name], range_pct)
                new_lo, new_hi = str(lo), str(hi)
                if len(new_lo) < len(old_lo):
                    new_lo = ' ' * (len(old_lo) - len(new_lo)) + new_lo
                elif len(new_lo) > len(old_lo):
                    prefix = prefix[:-(len(new_lo) - len(old_lo))]
                if len(new_hi) < len(old_hi):
                    new_hi = ' ' * (len(old_hi) - len(new_hi)) + new_hi
                elif len(new_hi) > len(old_hi):
                    sep = sep[:-(len(new_hi) - len(old_hi))]
                line = line[:match.start()] + prefix + new_lo + sep + new_hi + trailing + line[match.end():]
                if line != original_line:
                    adjusted.add(name)
                    logging.info(f"Adjusted bounds: {original_line.strip()} -> {line.strip()}")
                break
        result_lines.append(line)

    if adjusted:
        with open(config_file, 'w') as f:
            f.writelines(result_lines)
        logging.info(f"Adjusted {len(adjusted)} piece param bound(s) in {config_file}")

    return adjusted


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
    args = parser.parse_args()

    if args.range_pct is not None and args.range_pct <= 0:
        parser.error('--range must be > 0')

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
    pst_re = re.compile(r'PS_(\d+|KEG)_(\d+)$')
    pst_values = {}   # name -> (piece_key, square, value)
    config_values = {}
    for name, val in engine_values.items():
        m = pst_re.match(name)
        if m:
            pst_values[name] = (m.group(1), int(m.group(2)), val)
        else:
            config_values[name] = val

    if pst_values:
        logging.info(f"{len(pst_values)} PST parameter(s), {len(config_values)} config parameter(s)")

    # Patch config.h (DECLARE_PARAM / DECLARE_VALUE / DECLARE_NORMAL)
    updated, found = update_config(args.config, config_values, finalize=args.finalize, range_pct=args.range_pct)

    # Patch piece values in chess.h (PIECE_VALUES / ENDGAME_ADJUST macros)
    not_in_config = {n: v for n, v in engine_values.items() if n not in found}
    piece_candidates = {n: v for n, v in not_in_config.items()
                        if n in PIECE_INDEX or n in ENDGAME_ADJUST_INDEX}

    if piece_candidates:
        header_path = args.header or os.path.join(os.path.dirname(args.config) or '.', 'chess.h')
        if os.path.exists(header_path):
            pv_updated, pv_found = update_piece_values(header_path, piece_candidates)
            updated |= pv_updated
            found |= pv_found
        else:
            logging.warning(f"Header file not found for piece values: {header_path}")

    # Adjust Config::Param bounds for piece values in config.h
    if args.range_pct is not None and piece_candidates:
        adjust_piece_param_bounds(args.config, piece_candidates, args.range_pct)

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
