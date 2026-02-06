#!/usr/bin/env python3
"""
Generate SPSA tuning project from engine parameters.

Uses get_param_info() from chess_engine (same pattern as genlakas.py/gentune.py)
to create a project directory with tuning.json, worker.json, and (on Windows)
an engine wrapper batch file, ready for hand-editing.

Usage:
    python genconfig.py <project_name> [-D 8] [-H 256] [-T 1]
                        [-b 10000] [-g 200] [param_names... | all]

Creates:
    tuneup/<project_name>/
        tuning.json     # session config (edit before running)
        worker.json     # local worker config (edit per machine)
        engine.bat      # engine wrapper (Windows only)
"""

import argparse
import json
import os
import sys
import sysconfig
import warnings
from pathlib import PurePosixPath


def root_path():
    return os.path.abspath(os.path.join(os.path.split(sys.argv[0])[0], '..', '..', '..'))


def tuneup_path():
    return os.path.join(root_path(), 'tuneup')


def to_forward_slash(path):
    """Convert path to use forward slashes (cross-platform consistency)."""
    normalized = os.path.normpath(path).replace('\\', '/')
    # On Unix/Linux, ensure absolute paths start with /
    # On Windows, drive letter paths like C:/path are already correct
    if os.path.isabs(path) and not normalized.startswith('/') and ':' not in normalized[:2]:
        normalized = '/' + normalized
    return normalized


def abspath(path):
    """Absolute path with forward slashes."""
    return to_forward_slash(os.path.abspath(path))


sys.path.append(root_path())
from chess_engine import get_param_info


def get_engine_cmd(project_dir):
    """
    Get engine command for worker config.

    On Windows: creates engine.bat wrapper and returns absolute path to it.
    On Linux: returns absolute path to main.py directly.
    """
    engine_py = abspath(os.path.join(root_path(), 'main.py'))
    windows = sysconfig.get_platform().startswith('win')

    if windows:
        bat_path = os.path.join(project_dir, 'engine.bat')
        with open(bat_path, 'w') as f:
            f.write(f'@"{sys.executable}" "{engine_py}" %*\n')
        return abspath(bat_path)
    else:
        return engine_py


def main():
    # Enumerate available engine parameters
    params = {}
    groups = set()

    for name, (val, lo, hi, grp, normal) in get_param_info().items():
        if grp == 'Settings':
            continue
        groups.add(grp)
        if normal:
            unscaled_val = val
            val = 2 * (val - lo) / (hi - lo) - 1
            if val < -1 or val > 1:
                raise ValueError(f'{name}: {val} (unscaled: {unscaled_val}) is out of range')
            params[name] = (val, -1.0, 1.0, grp, 'float')
        else:
            ptype = 'float' if isinstance(val, float) else 'int'
            params[name] = (val, lo, hi, grp, ptype)

    tunable = ['all'] + sorted(params.keys())

    parser = argparse.ArgumentParser(
        description='Generate SPSA tuning project from engine parameters.'
    )
    parser.add_argument('project', help='Project name (creates tuneup/<project>/)')
    parser.add_argument('tune', choices=tunable, nargs='*', default='all',
                        help='Parameter names to tune (or "all")')
    parser.add_argument('-t', '--time-control', default='1+0.1',
                        help='Time control (default: 1+0.1)')
    parser.add_argument('-D', '--depth', type=int, default=None,
                        help='Fixed search depth (overrides time control)')
    parser.add_argument('-H', '--hash', type=int, default=256,
                        help='Engine hash table size in MB (default: 256)')
    parser.add_argument('-T', '--threads', type=int, default=1,
                        help='Engine threads (default: 1)')
    parser.add_argument('-b', '--budget', type=int, default=10000,
                        help='Total games budget (default: 10000)')
    parser.add_argument('-g', '--games-per-iteration', type=int, default=200,
                        help='Games per SPSA iteration (default: 200)')
    parser.add_argument('-c', '--spsa-c', type=float, default=0.05,
                        help='SPSA perturbation size as fraction of range (default: 0.05)')
    parser.add_argument('-a', '--spsa-a', type=float, default=0.5,
                        help='SPSA learning rate (default: 0.5)')

    args = parser.parse_args()

    # Resolve 'all' and build tune list
    if not isinstance(args.tune, list):
        args.tune = [args.tune]

    all_names = sorted(params.keys())
    tune_names = set()
    for p in args.tune:
        if p == 'all':
            tune_names.update(all_names)
        else:
            tune_names.add(p)

    # Create project directory
    project_dir = os.path.join(tuneup_path(), args.project)
    if os.path.exists(project_dir):
        print(f'Error: Project directory already exists: {project_dir}', file=sys.stderr)
        sys.exit(1)
    os.makedirs(project_dir, exist_ok=False)
    os.makedirs(os.path.join(project_dir, 'logs'), exist_ok=True)
    project_dir_abs = abspath(project_dir)

    # Engine command (creates wrapper on Windows)
    engine_cmd = get_engine_cmd(project_dir)

    # Default book path (absolute, forward slashes)
    default_book = abspath(os.path.join(tuneup_path(), 'books', '8moves_v3.pgn'))

    # Build tunable params
    tune_params = {}
    for name in sorted(tune_names):
        val, lo, hi, grp, ptype = params[name]
        tune_params[name] = {
            'init': val,
            'lower': lo,
            'upper': hi,
            'type': ptype,
        }

    # --- tuning.json (session-level, shared) ---
    tuning_config = {
        'engine': {
            'protocol': 'uci',
            'fixed_options': {
                'Hash': args.hash,
                'Threads': args.threads,
                'OwnBook': False,
            },
        },
        'games_per_iteration': args.games_per_iteration,
        'output_dir': project_dir_abs,
        'spsa': {
            'budget': args.budget,
            'a': args.spsa_a,
            'c': args.spsa_c,
            'A_ratio': 0.1,
            'alpha': 0.602,
            'gamma': 0.101,
        },
        'parameters': tune_params,
    }

    if args.depth is not None:
        tuning_config['depth'] = args.depth
    else:
        tuning_config['time_control'] = args.time_control

    tuning_path = os.path.join(project_dir, 'tuning.json')
    with open(tuning_path, 'w') as f:
        json.dump(tuning_config, f, indent=2)
        f.write('\n')

    # --- worker.json (per-machine, local) ---
    games_dir = abspath(os.path.join(project_dir, 'games'))
    log_file = abspath(os.path.join(project_dir, 'logs', 'worker.log'))

    worker_config = {
        'coordinator': 'http://localhost:8080',
        'engine': engine_cmd,
        'cutechess_cli': 'cutechess-cli',
        'concurrency': os.cpu_count() or 1,
        'opening_book': default_book,
        'book_format': 'pgn',
        'book_depth': 8,
        'games_dir': games_dir,
        'log_file': log_file,
        'parameter_overrides': {
            '_comment': 'per-machine parameter overrides (e.g., SyzygyPath)',
        },
    }

    worker_path = os.path.join(project_dir, 'worker.json')
    with open(worker_path, 'w') as f:
        json.dump(worker_config, f, indent=2)
        f.write('\n')

    # Summary
    iterations = args.budget // args.games_per_iteration
    print(f'Project created: {project_dir_abs}/')
    print(f'  tuning.json   - {len(tune_params)} parameters, {args.budget} games, {iterations} iterations')
    print(f'  worker.json   - concurrency={worker_config["concurrency"]}, engine={engine_cmd}')
    print()
    print('Next steps:')
    print(f'  1. Review and edit tuning.json and worker.json')
    print(f'  2. cd {project_dir_abs}')

    coordinator_py = abspath(os.path.join(root_path(), 'tools', 'tuneup', 'spsa', 'coordinator.py'))
    worker_py = abspath(os.path.join(root_path(), 'tools', 'tuneup', 'spsa', 'worker.py'))
    print(f'  3. python {coordinator_py} -c tuning.json')
    print(f'  4. python {worker_py} -c worker.json')

    if not tune_params:
        warnings.warn('No tunable parameters selected!')


if __name__ == '__main__':
    main()
