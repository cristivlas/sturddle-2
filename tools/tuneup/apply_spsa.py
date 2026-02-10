#!/usr/bin/env python3
"""
Apply SPSA tuning results to config.h.

Reads spsa_state.json (produced by the SPSA coordinator) and updates
DECLARE_PARAM / DECLARE_VALUE / DECLARE_NORMAL lines in config.h.

Handles both normalized (DECLARE_NORMAL, values in [-1,1]) and
non-normalized parameters, converting all to engine-space integers.

Parameters not found in config.h (e.g. piece weights, piece-square
tables) are listed with their engine-space values for manual review.

Usage:
    python apply_spsa.py <spsa_state.json | project_dir> [--config config.h] [--finalize]
"""

import argparse
import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def root_path():
    return os.path.abspath(os.path.join(os.path.split(sys.argv[0])[0], '../..'))


sys.path.append(root_path())
from chess_engine import get_param_info

params = get_param_info()


def denormalize(name, theta_val):
    """Convert a theta value to engine-space integer.

    For normalized parameters (DECLARE_NORMAL), maps from [-1,1] to [lo,hi].
    For non-normalized parameters, rounds to int.
    """
    p = params.get(name)
    if p:
        _default_val, lo, hi, _grp, normal = p
        if normal:
            return int(round((theta_val + 1) * (hi - lo) / 2 + lo))

    return int(round(theta_val))


def update_config(config_file, engine_values, finalize=False):
    """Patch DECLARE_PARAM/DECLARE_VALUE/DECLARE_NORMAL lines in config.h.

    If finalize is True, also converts DECLARE_PARAM/DECLARE_NORMAL back to
    DECLARE_VALUE for the tuned parameters.

    Returns set of parameter names that were successfully updated.
    """
    with open(config_file, 'r') as f:
        lines = f.readlines()

    updated = set()
    finalized = set()
    updated_lines = []

    for line in lines:
        original_line = line
        for name, value in engine_values.items():
            for macro in ('DECLARE_VALUE', 'DECLARE_PARAM', 'DECLARE_NORMAL'):
                pattern = re.compile(
                    rf'({macro}\s*\(\s*{re.escape(name)}\s*,\s*)(-?\d+)(\s*,\s*-?\d+\s*,\s*-?\d+\s*\))'
                )
                match = pattern.search(line)
                if match:
                    before = match.group(1)
                    old_val = match.group(2)
                    after = match.group(3)

                    new_val = str(value)

                    # Preserve column alignment
                    if len(new_val) < len(old_val):
                        new_val = ' ' * (len(old_val) - len(new_val)) + new_val
                    elif len(new_val) > len(old_val):
                        before = before[:-(len(new_val) - len(old_val))]

                    line = pattern.sub(f'{before}{new_val}{after}', line)
                    if line != original_line:
                        updated.add(name)
                        logging.info(f"Updated: {original_line.strip()} -> {line.strip()}")

                    if finalize and macro != 'DECLARE_VALUE':
                        line = line.replace(macro, 'DECLARE_VALUE', 1)
                        finalized.add(name)

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

    return updated


def main():
    parser = argparse.ArgumentParser(
        description='Apply SPSA tuning results to config.h.'
    )
    parser.add_argument('state', help='Path to spsa_state.json or SPSA project directory')
    parser.add_argument('--config', default=os.path.join(root_path(), 'config.h'),
                        help='Path to config.h (default: <project_root>/config.h)')
    parser.add_argument('--finalize', action='store_true',
                        help='Convert DECLARE_PARAM/DECLARE_NORMAL back to DECLARE_VALUE for tuned parameters')
    args = parser.parse_args()

    # Resolve state file path
    state_path = args.state
    if os.path.isdir(state_path):
        state_path = os.path.join(state_path, 'spsa_state.json')

    if not os.path.exists(state_path):
        logging.error(f"State file not found: {state_path}")
        sys.exit(1)

    # Load SPSA state
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
    engine_values = {}
    for name, val in theta.items():
        engine_values[name] = denormalize(name, val)

    # Patch config.h
    updated = update_config(args.config, engine_values, finalize=args.finalize)

    # Report params not found in config.h
    not_found = {name: engine_values[name] for name in engine_values if name not in updated}
    if not_found:
        logging.info(f"{len(not_found)} parameter(s) not found in {args.config}:")
        for name, val in sorted(not_found.items()):
            print(f"  {name} = {val}")


if __name__ == '__main__':
    main()
