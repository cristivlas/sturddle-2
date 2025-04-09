#! /usr/bin/env python3
#
# Generate script for Lakas
# https://github.com/fsmosca/Lakas/tree/main
#
import argparse
import os
import shutil
import sys
import sysconfig
import warnings


def root_path():
    return os.path.abspath(os.path.join(os.path.split(sys.argv[0])[0], '..'))

def make_path(*args):
    return os.path.abspath(os.path.join(root_path(), *args))

sys.path.append(root_path())
from chess_engine import *


def get_engine_path(args, windows=False):
    try:
        import uci
        engine = make_path('sturddle.py')
    except:
        # use native (built-in) uci version
        engine = make_path('main.py')
    if windows:
        with open('engine.bat', 'w') as out:
            out.write(sys.executable + ' ' + engine)
        engine = 'engine.bat'
    return engine

optimizers = ('oneplusone', 'tbpsa', 'bayesopt', 'spsa', 'cmaes', 'ngopt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Lakas tuning script.')
    parser.add_argument('-b', '--budget', type=int, default=100)
    parser.add_argument('-c', '--concurrency', type=int, default=os.cpu_count())
    parser.add_argument('-d', '--data-file', default='checkpoint.dat')
    parser.add_argument('-g', '--games_per_budget', type=int, default=200)
    parser.add_argument('-l', '--log-file', default='log.txt')
    parser.add_argument('-o', '--output')
    parser.add_argument('-p', '--lakas-path', default='')
    parser.add_argument('-s', '--strategy', choices=optimizers, default='spsa')
    parser.add_argument('-t', '--time-control', default='5+0.05')

    # Enumerate available engine settings
    params = {}
    groups = set()

    for name, (val, lo, hi, grp, normal) in  get_param_info().items():
        if grp == 'Settings':
            continue
        groups.add(grp)
        if normal:
            val = 2 * (val - lo) / (hi - lo) - 1
            params[name] = val, -1.0, 1.0, grp
        else:
            params[name] = val, lo, hi, grp

    tunable = tuple(['all'] + list(params.keys()))
    parser.add_argument('tune', choices=tunable, nargs='*', default='all')

    args = parser.parse_args()

    # strip 'all'
    _, *tunable = tunable

    # deal with default 'all'
    if type(args.tune) is not list:
        args.tune = [args.tune]

    # substitute 'all'
    args.tune = set(p if p != 'all' else q for q in tunable for p in args.tune)

    # construct list of tunable params
    tune_params = []
    for name in sorted(list(args.tune)):
        val, lo, hi, _ = params[name]
        tune_params.append(f"\\\n'{name}':{{'init':{val},'lower':{lo},'upper':{hi}}},")

    # construct time control arguments
    tc = args.time_control.split('+')
    time_control = f'--base-time-sec {tc[0]} --inc-time-sec {tc[1] if len(tc) > 1 else 0}'

    # detect cutechess-cli location
    cutechess = shutil.which('cutechess-cli')
    if cutechess is None:
        raise RuntimeError('Could not locate cutechess-cli')

    windows = sysconfig.get_platform().startswith('win')

    # fill out the script template
    if windows:
        script = f'''
python {os.path.join(args.lakas_path, 'lakas.py')} ^
    --budget {args.budget} --games-per-budget {args.games_per_budget} ^
    --concurrency {args.concurrency} ^
    --engine {get_engine_path(args, True)} ^
    --input-data-file {args.data_file} ^
    --opening-file {os.path.join(args.lakas_path, 'start_opening/ogpt_chess_startpos.epd')} ^
    --optimizer {args.strategy} ^
    --optimizer-log-file {args.log_file} ^
    --output-data-file {args.data_file} ^
    --match-manager-path "{cutechess}" ^
    {time_control} ^
    --input-param="{{{"".join(tune_params).replace('\\\n', ' ')} }}"
'''
    else:
        script = f'''#!/usr/bin/env bash

python3 {os.path.join(args.lakas_path, 'lakas.py')} \\
    --budget {args.budget} --games-per-budget {args.games_per_budget} \\
    --concurrency {args.concurrency} \\
    --engine {get_engine_path(args)} \\
    --input-data-file {args.data_file} \\
    --opening-file {os.path.join(args.lakas_path, 'start_opening/ogpt_chess_startpos.epd')} \\
    --optimizer {args.strategy} \\
    --optimizer-log-file {args.log_file} \\
    --output-data-file {args.data_file} \\
    --match-manager-path {cutechess} \\
    {time_control} \\
    --input-param="{{{"".join(tune_params)}\\
}}"
'''
    # write out the script
    if not args.output:
        print(script)
    else:
        file_path = args.output
        with open(file_path, 'w') as out:
            out.write(script)
        if not windows:
            # make it executable
            os.chmod(file_path, os.stat(file_path).st_mode | 0o111)

    if not params:
        warnings.warn('No tunable parameters detected!')
