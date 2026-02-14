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


def update_config(config_file, engine_values, finalize=False):
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
                    rf'({macro}\s*\(\s*{re.escape(name)}\s*,\s*)(-?\d+)(\s*,\s*-?\d+\s*,\s*-?\d+\s*\))'
                )
                match = pattern.search(line)
                if match:
                    found.add(name)
                    before = match.group(1)
                    old_val = match.group(2)
                    after = match.group(3)

                    # Finalize macro first so alignment accounts for name change
                    if finalize and macro != 'DECLARE_VALUE':
                        pad = ' ' * (len(macro) - len('DECLARE_VALUE'))
                        before = before.replace(macro + '(', 'DECLARE_VALUE(' + pad, 1)
                        finalized.add(name)

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


def main():
    parser = argparse.ArgumentParser(
        description='Apply SPSA tuning results to config.h (using tuning.json for parameter metadata).'
    )
    parser.add_argument('project', help='Path to SPSA project directory or tuning.json file')
    parser.add_argument('--state', default=None, help='Path to spsa_state.json (default: <project>/spsa_state.json)')
    parser.add_argument('--config', default='config.h', help='Path to config.h (default: config.h)')
    parser.add_argument('--finalize', action='store_true', help='Convert DECLARE_PARAM/DECLARE_NORMAL to DECLARE_VALUE')
    args = parser.parse_args()

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

    # Patch config.h
    updated, found = update_config(args.config, engine_values, finalize=args.finalize)

    # Report params that match current values (no change needed)
    unchanged = {name: engine_values[name] for name in found if name not in updated}
    if unchanged:
        logging.info(f"{len(unchanged)} parameter(s) already at target value:")
        for name, val in sorted(unchanged.items()):
            print(f"  {name} = {val}")

    # Report params not found in config.h
    not_found = {name: engine_values[name] for name in engine_values if name not in found}
    if not_found:
        logging.info(f"{len(not_found)} parameter(s) not found in {args.config}:")
        for name, val in sorted(not_found.items()):
            print(f"  {name} = {val}")


if __name__ == '__main__':
    main()
