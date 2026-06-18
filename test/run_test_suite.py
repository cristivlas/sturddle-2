#! /usr/bin/env python3
"""
Tests Runner for the Sturddle Chess Engine (c) 2022 - 2025 Cristian Vlasceanu.
-------------------------------------------------------------------------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------------

Any third-party files include in this project are subject to copyright
and licensed as stated in their respective header notes.
"""
import argparse
import sys
from datetime import datetime
from functools import partial
from os import path

def root_path():
    return path.abspath(path.join(path.split(sys.argv[0])[0], '..'))

sys.path.append(root_path())

from chess_engine import *

ALGORITHM = { 'mtdf': MTDf_i, 'negamax': Negamax_i, 'negascout': Negascout_i }


class Timer:
    def __init__(self):
        self.start = datetime.now()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        nsec = int(self.seconds_elapsed())
        self.info = f'{nsec // 60:02d}:{nsec % 60:02d}'

    def seconds_elapsed(self):
        return (datetime.now() - self.start).total_seconds()


def print_header():
    print (f'Algorithm | {"Test":34s} | Time  | Result |    Depth    |  Evals   |   Nodes   |  Hash  |  TT hit   |\
     Speed     |   Solved')
    print (f"{'-'*10}+{'-'*36}+{'-'*7}+{'-'*8}+{'-'*13}+{'-'*10}+{'-'*11}+{'-'*8}+{'-'*11}+{'-'*15}+{'-'*15}")


"""
threads_report callback
"""
def collect_stats(stats, algo, ctxts):
    main_context = algo.context
    stats[main_context.task_id] = main_context.stats()

    for secondary_task_context in ctxts:
        if secondary_task_context:
            stats[secondary_task_context.task_id] = secondary_task_context.stats()


have_header = False


"""
Write move stats to file
"""
def write_stats(filename, name, result, algo, stats):
    global have_header

    agg = {}
    for k in stats[0]:
        if k in ['nps', 'tt-usage', 'tt-hit-rate', 'tt-l1-rate']:
            agg[k] = stats[0][k]  # Do not aggregate.
        else:
            agg[k] = sum([s[k] for s in stats.values()])

    agg['test'] = name
    #agg['result'] = result
    #agg['depth'] = algo.current_depth
    #agg['stm'] = chess.COLOR_NAMES[algo.context.board().turn]

    with open(filename, 'a+') as out:
        if not have_header:
            out.write(','.join(agg.keys()) + '\n')
            have_header = True
        out.write(','.join([str(x) for x in agg.values()]) + '\n')


def search(algo_class, name, board, expected, depths, **kwargs):
    logging.info(f'testing: {algo_class.__name__} {name}')
    task_stats = {}

    # Construct a search algorithm instance
    algo = algo_class(board, threads_report=partial(collect_stats, task_stats), **kwargs)

    info = f'{algo_class.__name__[:-2][:10]:10s}| {name[:34]:34s}'
    print (info, end=' | ', flush=True)

    with Timer() as timer:
        move, score = algo.search()

    print (f'{timer.info}', end=' | ')
    uci = move.uci() if move else None
    new_board = board.copy()
    san = new_board.san_and_push(move) if move else None
    assert new_board.is_valid(), (board, move)

    depths[0] += algo.current_depth

    a = depths[0] / depths[1]

    t = len(algo.context.get_pv())
    d = algo.current_depth

    # task_stats is empty when not running in multi-thread mode
    if not task_stats:
        task_stats = { 0: algo.stats }
    try:
        e = sum([stat['eval-count'] for stat in task_stats.values()])
        n = sum([stat['nodes'] for stat in task_stats.values()])
        h = sum([stat['tt-hits'] for stat in task_stats.values()])
        pr = sum([stat['tt-probes'] for stat in task_stats.values()])
        l1 = sum([stat['tt-l1-hits'] for stat in task_stats.values()])
    except:
        e = n = h = pr = l1 = 0

    tt_total[0] += h
    tt_total[1] += pr
    node_total[0] += n
    l1_total[0] += l1
    hr = 100.0 * h / pr if pr else 0.0

    secs = timer.seconds_elapsed()
    time_total[0] += secs
    nps = n / secs if secs else n
    u = algo.tt_usage
    stats = f'|{d:3d} {t:2d} ({a:4.1f})| {e:8d} | {n:9d} |{u:6.2f}% |{hr:6.2f}% hit| {nps/1000:8.1f} knps |'
    if uci in expected or san in expected:
        print (f' \u001b[32mOK.\u001b[0m   {stats}', end='')
        result = True
    else:
        print (f' \u001b[31mFail.\u001b[0m {stats}', end='')
        fail = f'result={move} ({san}) (score={score}), expected={expected}, pv={algo.context.get_pv()}'
        logging.error(fail)
        result = False

    if stats_filename := kwargs.get('stats', None):
        write_stats(stats_filename, name.strip('"'), result, algo, task_stats)

    return result, task_stats


perft_total = [0, 0]
tt_total = [0, 0]  # [hits, probes] accumulated over the suite
node_total = [0]   # total nodes accumulated over the suite
time_total = [0.0] # total search seconds accumulated over the suite
l1_total = [0]     # total L1 hits accumulated over the suite

"""
Run search over a collection of position (puzzles) read from an epd file
"""
def test_epd(args, filename, tests, algo_class, **kwargs):
    succeeded, total = 0, 0
    depths = [0, 0]
    tt_total[0] = tt_total[1] = 0
    node_total[0] = 0
    time_total[0] = 0.0
    l1_total[0] = 0
    i = 0
    tests = [line for line in tests.split('\n') if line]
    count = len(tests)

    if args.perft:
        print(f'Running {count} perf tests...')
    else:
        print()
        print(f'Running {count} tests...')
        print_header()

    for test in tests:
        fields = test.split(';')
        test = fields[0]
        if ' bm ' in test:
            test = test.split(' bm ')
        elif ' am ' in test:
            test = test.split(' am ')
        elif ' pv ' in test:
            test = test.split(' pv ')

        fen = test[0]
        if args.perft:
            perft_result = perft(fen, 1000000)
            # perft_result = perft2(fen, 1000000)
            # perft_result = perft3(fen, 100000)

            perft_total[0] += perft_result[0]
            perft_total[1] += perft_result[1]
            return

        expected = test[1].strip().split(' ')

        i += 1
        id = f'{path.splitext(path.basename(filename))[0]} {i}/{count}'
        for f in fields:
            f = f.strip()
            if f.startswith('id '):
                id = f.split('id ')[1]
                if len(id) > 33:
                    id = id[:15] + '...' + id[-15:]
                break

        total += 1
        depths[1] = total

        results = search(algo_class, id, chess.Board(fen=fen), expected, depths, **kwargs)
        if results[0]:
            succeeded += 1

        print(f' {succeeded:3d} ({succeeded * 100 / total:6.2f}%)')
        if args.verbose:
            for n in sorted(results[1].items()):
                print (f'{" "*77} | {n[1]["eval-count"]:8d} | {n[1]["nodes"]:9d} |')


    print(f'Succeeded: {succeeded} / {total}')
    print(f'Average depth: {depths[0] / total:.2f}')
    if tt_total[1]:
        print(f'TT hit rate: {100.0 * tt_total[0] / tt_total[1]:.2f}% ({tt_total[0]} / {tt_total[1]})')
    print(f'Total nodes: {node_total[0]}')
    if time_total[0]:
        print(f'Aggregate NPS: {node_total[0] / time_total[0] / 1000:.1f} knps ({time_total[0]:.1f}s search)')
    if l1_total[0]:
        print(f'L1 hit rate: {100.0 * l1_total[0] / (l1_total[0] + tt_total[1]):.2f}% ({l1_total[0]} L1 / {tt_total[1]} big-TT probes)')


def configure_logger(args):
    log = logging.getLogger()
    for h in log.handlers[:]:
        log.removeHandler(h)
    format = '%(asctime)s %(levelname)-8s %(message)s'
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, filename=args.logfile, format=format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_suites', nargs='+')
    parser.add_argument('-a', '--algo', choices=ALGORITHM.keys(), default='mtdf')
    parser.add_argument('-c', '--config', default='sturddle.cfg')
    parser.add_argument('-l', '--logfile', default='test_suite.log')
    parser.add_argument('-p', '--perft', action='store_true')
    parser.add_argument('-d', '--depth', type=int, help='fixed search depth (deterministic; overrides --time)')
    parser.add_argument('--hash', type=int, help='hash size in MB (override config; small forces eviction)')
    parser.add_argument('-s', '--stats', help='stats output filename')
    parser.add_argument('-t', '--time', type=int, default=5000)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    configure_logger(args)
    read_config(args.config, echo=True)

    if args.hash:
        set_hash_size(args.hash)
        print(f'Hash size set to {get_hash_size()} MB')

    if args.stats:
        with open(args.stats, 'w'):
            pass

    # Fixed depth (deterministic) gives clean A/B numbers; a huge time limit lets depth bind.
    if args.depth:
        run_kwargs = dict(depth=args.depth, time_limit_ms=10**9, stats=args.stats)
    else:
        run_kwargs = dict(time_limit_ms=args.time, stats=args.stats)

    for suite in args.test_suites:
        with open(suite) as f:
            epd = f.read()
            test_epd(args, suite, epd, ALGORITHM[args.algo], **run_kwargs)

    if args.perft:
        rate = perft_total[0] / (perft_total[1] * 1000000)
        print(f'{perft_total}, {rate:.2f} Megamoves / sec')


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        pass
