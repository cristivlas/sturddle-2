#!/usr/bin/env python3
"""
Generate SPSA tuning project from engine parameters.

Uses get_param_info() from chess_engine (same pattern as genlakas.py/gentune.py)
to create a project directory with tuning.json and worker.json, ready for hand-editing.

Usage:
    python genconfig.py <project_name> [-D 8] [-H 256] [-T 1]
                        [-i 100] [-g 100] [param_names... | all]
    python genconfig.py <project_name> -w   # worker.json only (no engine needed)
    python genconfig.py <project_name> -s   # tuning.json (coordinator) only

Creates:
    tuneup/<project_name>/
        tuning.json     # session config (edit before running)
        worker.json     # local worker config (edit per machine)
"""

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import warnings

from config import (
    EngineConfig, SPSAConfig, TuningConfig, Parameter, WorkerConfig,
)
from recommend import ParamSpec, recommend, format_recommendation, constrain_for


# Knob choices for --check-ranges. The shared recommend module no longer
# carries defaults; each caller picks its own (see feedback memory).
CHECK_OUTLIER_RATIO = 5.0
CHECK_MIN_PERT_PCT = 5.0
CHECK_SAFETY_PAD = 1.2


def physical_cpu_count():
    """Return the number of physical CPU cores (not logical/hyperthreaded)."""
    try:
        if sys.platform == 'win32':
            out = subprocess.check_output(['wmic', 'cpu', 'get', 'NumberOfCores', '/value'], text=True)
            cores = sum(int(m.group(1)) for m in re.finditer(r'NumberOfCores=(\d+)', out))
            if cores > 0:
                return cores
        else:
            with open('/proc/cpuinfo') as f:
                ids = set(re.findall(r'^core id\s*:\s*(\d+)', f.read(), re.MULTILINE))
            if ids:
                return len(ids)
    except Exception:
        pass
    return os.cpu_count() or 1


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


def find_game_runner():
    """Auto-detect fastchess or cutechess-cli. Fastchess preferred over cutechess-cli."""
    windows = sysconfig.get_platform().startswith('win')
    binary = 'fastchess.exe' if windows else 'fastchess'
    # 1) fastchess in PATH
    path = shutil.which('fastchess')
    if path:
        print(f'  Found fastchess in PATH: {path}')
        return abspath(path)
    # 2) fastchess in sibling directories of root project
    parent = os.path.dirname(root_path())
    pattern = os.path.join(parent, '*fastchess*')
    print(f'  Scanning siblings: {pattern}')
    for d in glob.glob(pattern):
        candidate = os.path.join(d, binary)
        print(f'  Checking: {candidate} (exists={os.path.isfile(candidate)})')
        if os.path.isfile(candidate):
            return abspath(candidate)
    # 3) cutechess-cli in PATH as fallback
    path = shutil.which('cutechess-cli')
    if path:
        print(f'  Found cutechess-cli in PATH: {path}')
        return abspath(path)
    warnings.warn('Neither fastchess nor cutechess-cli found in PATH or sibling directories')
    return 'cutechess-cli'


def parse_version():
    path = os.path.join(root_path(), 'version.h')
    with open(path) as f:
        text = f.read()
    major = re.search(r'^#define\s+STURDDLE_VERSION_MAJOR\s+(\d+)', text, re.M).group(1)
    minor = re.search(r'^#define\s+STURDDLE_VERSION_MINOR\s+(\d+)', text, re.M).group(1)
    patch = re.search(r'^#define\s+STURDDLE_VERSION_PATCH\s+"([^"]+)"', text, re.M).group(1)
    return f'{major}.{minor}.{patch}'


def resolve_native_engine():
    version = parse_version()
    windows = sysconfig.get_platform().startswith('win')
    name = f'sturddle-{version}.exe' if windows else f'sturddle-{version}'
    path = os.path.join(root_path(), 'dist', 'native', name)
    if not os.path.isfile(path):
        print(f'Error: Native engine not found: {path}', file=sys.stderr)
        print(f'Build it with: python tools/make-native.py', file=sys.stderr)
        sys.exit(1)
    return abspath(path)


def resolve_engine(value):
    """Resolve engine: if value is a path, canonicalize it; otherwise look up version in dist/. Exits on failure."""
    if os.path.exists(value):
        path = os.path.realpath(value)
        if not os.path.isfile(path):
            print(f'Error: Not a regular file: {path}', file=sys.stderr)
            sys.exit(1)
        return abspath(path)
    dist_dir = os.path.join(root_path(), 'dist')
    windows = sysconfig.get_platform().startswith('win')
    if windows:
        name = f'sturddle-{value}.exe'
    else:
        name = f'sturddle-{value}-Linux-{platform.machine()}'
    path = os.path.join(dist_dir, name)
    if not os.path.isfile(path):
        print(f'Error: Engine not found: {path}', file=sys.stderr)
        sys.exit(1)
    return abspath(path)


_PYI_MAGIC = b'MEI\x0c\x0b\x0a\x0b\x0e'


def is_pyinstaller_onefile(path: str) -> bool:
    """Detect PyInstaller --onefile binaries by scanning the tail for the CArchive cookie magic."""
    if not path or not os.path.isfile(path):
        return False
    if path.lower().endswith(('.py', '.bat', '.cmd', '.sh')):
        return False
    try:
        size = os.path.getsize(path)
        chunk = min(size, 4 * 1024 * 1024)
        with open(path, 'rb') as f:
            f.seek(size - chunk)
            return _PYI_MAGIC in f.read(chunk)
    except OSError:
        return False


def _build_tune_params(tune_arg):
    """Load chess_engine and build tune_params from get_param_info(). Exits on unknown name."""
    sys.path.append(root_path())
    from chess_engine import get_param_info

    params = {}
    for name, (val, lo, hi, grp, normal) in get_param_info().items():
        if grp == 'Settings':
            continue
        if normal:
            unscaled_val = val
            v = 2 * (val - lo) / (hi - lo) - 1
            if v < -1 or v > 1:
                raise ValueError(f'{name}: {v} (unscaled: {unscaled_val}) is out of range')
            params[name] = (v, -1.0, 1.0, grp, 'float', lo, hi)
        else:
            ptype = 'float' if isinstance(val, float) else 'int'
            params[name] = (val, lo, hi, grp, ptype, None, None)

    if not isinstance(tune_arg, list):
        tune_arg = [tune_arg]
    all_names = sorted(params.keys())
    tune_names = set()
    for p in tune_arg:
        if p == 'all':
            tune_names.update(all_names)
        elif p in params:
            tune_names.add(p)
        else:
            print(f'Error: Unknown parameter: {p}', file=sys.stderr)
            print(f'Available: {", ".join(all_names)}', file=sys.stderr)
            sys.exit(1)

    tune_params = {}
    for name in sorted(tune_names):
        val, lo, hi, grp, ptype, orig_lo, orig_hi = params[name]
        p = {
            'init': val,
            'lower': lo,
            'upper': hi,
            'type': ptype,
        }
        if orig_lo is not None:
            p['original_lower'] = orig_lo
            p['original_upper'] = orig_hi
        tune_params[name] = p

    return tune_params


def _print_recommendations(tune_params, iterations, gamma, a_to_c_ratio, target_r=None, target_c=None):
    """Dry-run: build ParamSpec list from tune_params and delegate to recommend module."""
    if not tune_params:
        print('No tunable parameters to analyze.')
        return

    def is_int_like(x):
        return isinstance(x, int) or (isinstance(x, float) and x.is_integer())

    specs = []
    for name, p in tune_params.items():
        if 'original_lower' in p:
            lo, hi = p['original_lower'], p['original_upper']
            center = (p['init'] + 1) * (hi - lo) / 2 + lo
        else:
            lo, hi = p['lower'], p['upper']
            center = p['init']
        is_int = is_int_like(lo) and is_int_like(hi)
        fixed_lo, fixed_hi, floor = constrain_for(lo, hi)
        # Genconfig: current bounds == engine cap (fresh from get_param_info), so pass as both.
        specs.append(ParamSpec(name=name, center=center, current_lo=lo, current_hi=hi,
                               is_int=is_int, cap_lo=lo, cap_hi=hi,
                               fixed_lo=fixed_lo, fixed_hi=fixed_hi, floor=floor))

    rec = recommend(specs, iterations, gamma, a_to_c_ratio,
                    target_r=target_r, target_c=target_c,
                    outlier_ratio=CHECK_OUTLIER_RATIO,
                    min_pert_pct=CHECK_MIN_PERT_PCT,
                    safety_pad=CHECK_SAFETY_PAD)
    format_recommendation(rec, center_label='init')


def main():
    parser = argparse.ArgumentParser(description='Generate SPSA tuning project from engine parameters.')
    parser.add_argument('project', nargs='?', default=None, help='Project name (creates tuneup/<project>/) -- omit with --dry-run')
    parser.add_argument('tune', nargs='*', default='all', help='Parameter names to tune (or "all")')
    parser.add_argument('-w', '--worker-only', action='store_true', help='Generate worker.json only (no engine needed)')
    parser.add_argument('-s', '--server-only', action='store_true', help='Generate tuning.json (coordinator) only, skip worker.json')
    parser.add_argument('--dry-run', '--check-ranges', dest='dry_run', action='store_true', help='Report engine-space range distribution and SPSA schedule implications; write nothing')
    target_grp = parser.add_mutually_exclusive_group()
    target_grp.add_argument('--target-r', type=float, default=None, help='Override R_target (engine-space range) for --check-ranges; default is geometric mean of current ranges')
    target_grp.add_argument('--target-c', type=float, default=None, help='Override c (perturbation fraction); R_target is derived as N^gamma / c')
    parser.add_argument('-e', '--engine', metavar='VERSION_OR_PATH', help='Engine version from dist/ (e.g., 2.5.1-pieces) or path to binary; defaults to dist/native/ build')
    parser.add_argument('-r', '--ref', metavar='VERSION_OR_PATH', help='Reference engine version from dist/ (e.g., 2.5.0) or path to binary')
    _tc = TuningConfig()
    _spsa = SPSAConfig()
    parser.add_argument('-t', '--time-control', default=_tc.time_control, help=f'Time control (default: {_tc.time_control})')
    parser.add_argument('-D', '--depth', type=int, default=_tc.depth, help='Fixed search depth (overrides time control)')
    parser.add_argument('-H', '--hash', type=int, default=256, help='Engine hash size in MB (default: 256)')
    parser.add_argument('-T', '--threads', type=int, default=1, help='Engine threads (default: 1)')
    parser.add_argument('-i', '--iterations', type=int, default=10000, help='SPSA iterations (default: 10000)')
    parser.add_argument('-g', '--games-per-iteration', type=int, default=_tc.games_per_iteration, help=f'Games per iteration (default: {_tc.games_per_iteration})')
    parser.add_argument('-c', '--spsa-c', type=float, default=None, help=f'SPSA perturbation size (default: auto from param ranges)')
    parser.add_argument('-a', '--spsa-a', type=float, default=None, help=f'SPSA learning rate (default: auto, scaled from c)')
    args = parser.parse_args()

    if args.dry_run:
        if args.worker_only:
            parser.error('--dry-run is incompatible with --worker-only (no params to analyze)')
        # In dry-run there is no project; the first positional is also a tune name.
        if args.project is None:
            tune_arg = args.tune
        else:
            rest = args.tune if isinstance(args.tune, list) else []
            tune_arg = [args.project] + rest
        tune_params = _build_tune_params(tune_arg)
        _print_recommendations(tune_params, args.iterations, _spsa.gamma, _spsa.a / _spsa.c, target_r=args.target_r, target_c=args.target_c)
        return

    if not args.project:
        parser.error('project name required (or pass --dry-run)')

    # Create project directory
    if os.path.isabs(args.project):
        project_dir = args.project
    else:
        project_dir = os.path.join(tuneup_path(), args.project)
    if os.path.exists(project_dir):
        print(f'Error: Project directory already exists: {project_dir}', file=sys.stderr)
        sys.exit(1)
    os.makedirs(project_dir, exist_ok=False)
    os.makedirs(os.path.join(project_dir, 'logs'), exist_ok=True)
    project_dir_abs = abspath(project_dir)

    # Engine command: explicit path/version via -e, otherwise the native build from dist/native/.
    if not args.server_only:
        engine_cmd = resolve_engine(args.engine) if args.engine else resolve_native_engine()

    # Default book path (absolute, forward slashes)
    default_book = abspath(os.path.join(tuneup_path(), 'books', 'UHO_2024_6mvs_+085_+094.pgn'))

    # --- tuning.json (session-level, shared) ---
    tune_params = {}

    if not args.worker_only:
        tune_params = _build_tune_params(args.tune)

        # Auto-calculate c so the tightest param hits the min-perturbation clamp
        # at ~100% of the budget: c = N^gamma / min_engine_range
        if args.spsa_c is not None:
            spsa_c = args.spsa_c
        else:
            min_engine_range = min((p.get('original_upper', p['upper']) - p.get('original_lower', p['lower'])) for p in tune_params.values()) if tune_params else 1.0
            spsa_c = round(args.iterations ** _spsa.gamma / min_engine_range, 4)
            print(f'  Auto c={spsa_c} (min engine range={min_engine_range:.0f}, clamp target={args.iterations} iters)')
        spsa_a = args.spsa_a if args.spsa_a is not None else round(spsa_c * (_spsa.a / _spsa.c), 4)
        if args.spsa_a is None:
            print(f'  Auto a={spsa_a} (c={spsa_c} * ratio {_spsa.a / _spsa.c:.0f})')

        # SSE heartbeat interval (real updates push immediately)
        dashboard_refresh = 60

        parameters = {
            name: Parameter(name=name, **p) for name, p in tune_params.items()
        }

        tuning_config = TuningConfig(
            engine=EngineConfig(
                protocol='uci',
                fixed_options={
                    'Hash': args.hash,
                    'Threads': args.threads,
                    'OwnBook': False,
                },
            ),
            time_control=args.time_control,
            depth=args.depth,
            games_per_iteration=args.games_per_iteration,
            output_dir=project_dir_abs,
            static_dir=abspath(root_path()),
            dashboard_refresh=dashboard_refresh,
            spsa=SPSAConfig(
                budget=args.iterations * args.games_per_iteration,
                a=spsa_a,
                c=spsa_c,
            ),
            parameters=parameters,
        )

        tuning_path = os.path.join(project_dir, 'tuning.json')
        with open(tuning_path, 'w') as f:
            f.write(tuning_config.to_json())
            f.write('\n')

    # --- worker.json (per-machine, local) ---
    if not args.server_only:
        games_dir = abspath(os.path.join(project_dir, 'games'))
        log_file = abspath(os.path.join(project_dir, 'logs', 'worker.log'))

        _wd = WorkerConfig  # shorthand for accessing dataclass defaults
        worker_config = {
            'name': platform.node(),
            'coordinator': _wd.coordinator,
            'engine': engine_cmd,
            'cutechess_cli': find_game_runner(),
            'concurrency': physical_cpu_count(),
            'opening_book': default_book,
            'book_format': 'pgn',
            'book_depth': _wd.book_depth,
            'games_dir': games_dir,
            'log_file': log_file,
            'max_forfeit_pct': _wd.max_forfeit_pct,
            'parameter_overrides': {
                '_comment': 'per-machine parameter overrides (e.g., SyzygyPath)',
            },
        }
        if args.ref:
            worker_config['reference_engine'] = resolve_engine(args.ref)

        # Enable ramdisk only if an engine is a PyInstaller --onefile binary.
        if is_pyinstaller_onefile(engine_cmd) or is_pyinstaller_onefile(worker_config.get('reference_engine', '')):
            worker_config['ramdisk'] = True

        worker_path = os.path.join(project_dir, 'worker.json')
        with open(worker_path, 'w') as f:
            json.dump(worker_config, f, indent=2)
            f.write('\n')

    # Summary
    indent = 2 * ' '
    print(f'Project created: {project_dir_abs}/')
    if not args.worker_only:
        budget = args.iterations * args.games_per_iteration
        print(f'{indent}tuning.json   - {len(tune_params)} parameters, {args.iterations} iterations, {budget} games')
    if not args.server_only:
        print(f'{indent}worker.json   - concurrency={worker_config["concurrency"]}, engine={engine_cmd}, runner={worker_config["cutechess_cli"]}')
        if args.ref:
            print(f'{indent}{14 * " "}  reference_engine={worker_config["reference_engine"]}')
    print()
    print('Next steps:')
    step = 1
    if args.worker_only:
        edit_files = 'worker.json'
    elif args.server_only:
        edit_files = 'tuning.json'
    else:
        edit_files = 'tuning.json and worker.json'
    print(f'{indent}{step}. Review and edit {edit_files}')
    step += 1
    print(f'{indent}{step}. cd {project_dir_abs}')
    step += 1
    coordinator_py = abspath(os.path.join(root_path(), 'tools', 'tuneup', 'spsa', 'coordinator.py'))
    worker_py = abspath(os.path.join(root_path(), 'tools', 'tuneup', 'spsa', 'worker.py'))
    if not args.worker_only:
        print(f'{indent}{step}. python {coordinator_py} -c tuning.json')
        step += 1
    if not args.server_only:
        print(f'{indent}{step}. python {worker_py} -c worker.json')
        _print_numa_hint(worker_config['concurrency'], indent)

    if not args.worker_only and not tune_params:
        warnings.warn('No tunable parameters selected!')


def _print_numa_hint(concurrency: int, indent: str):
    """If this machine is NUMA with 2+ CPU-bearing nodes, print a hint
    pointing the user at the split/launch helpers. No-op otherwise."""
    try:
        from split_numa_config import numa_available, cpu_bearing_nodes
    except ImportError:
        return
    if not numa_available():
        return
    nodes = cpu_bearing_nodes()
    n = len(nodes)
    if n < 2:
        return
    spsa_dir = os.path.dirname(os.path.abspath(__file__))
    split_py = os.path.join(spsa_dir, 'split_numa_config.py')
    launch_py = os.path.join(spsa_dir, 'launch_numa_workers.py')
    print()
    print(f'{indent}NUMA detected: {n} CPU-bearing nodes. For pinned per-node workers:')
    if concurrency % n == 0:
        print(f'{indent}  python {split_py} worker.json')
        print(f'{indent}  python {launch_py} worker.json')
    else:
        lower = (concurrency // n) * n
        upper = lower + n
        suggestions = ' or '.join(str(v) for v in (lower, upper) if v > 0)
        print(f'{indent}  (adjust concurrency to {suggestions} first -- must divide {n} evenly)')


if __name__ == '__main__':
    main()
