#! /usr/bin/env python3
'''
Use chess engine to build a training sqlite3 DB of scored positions.
'''
import argparse
import os
from math import copysign

import chess
import chess.engine
import chess.pgn
from dbutils.sqlite import SQLConn
from tqdm.contrib import tenumerate

_create_table = '''CREATE TABLE IF NOT EXISTS position(
    epd text PRIMARY KEY,   -- Position
    depth integer,          -- Analysis Depth
    score integer           -- Score
)'''

_insert = '''INSERT INTO position(epd, depth, score) VALUES(?,?,?)'''

def analyse(args, sql_out, engine, epd):
    board = chess.Board(fen=epd)
    if not board.is_valid() or board.is_game_over():
        return

    limit = chess.engine.Limit(depth=args.depth) if args.depth else chess.engine.Limit(time=args.time_limit)
    info = engine.analyse(board, limit)
    score = info['score']
    depth = info['depth']
    if not score.is_mate():
        sql_out.exec(_insert, (epd,  info['depth'], score.pov(score.turn).score()))
    elif args.mate_score:
        mate_dist = score.pov(score.turn).mate()
        score = int(copysign(args.mate_score, mate_dist) - (mate_dist))
        sql_out.exec(_insert, (epd,  info['depth'], score))

def get_engine(args):
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    config = {'Threads': args.threads, 'Hash': args.hash}
    if 'Use NNUE' in engine.options:
        config['Use NNUE'] = 'true' if args.nnue else 'false'

    engine.configure(config)
    return engine

def filter_positions(args, row):
    #if args.popularity_threshold and row[4] < args.popularity_threshold:
    #    return False
    assert not args.popularity_threshold or row[4] >= args.popularity_threshold

    if args.min_win_rate or args.max_loss_rate:
        win_rate = row[5] / row[4]
        loss_rate = row[6] / row[4]
        if args.min_win_rate and win_rate < args.min_win_rate:
            return False
        if args.max_loss_rate and loss_rate > args.max_loss_rate:
            return False
    return True

def main(args):
    if args.cleanup and os.path.exists(args.output):
        os.unlink(args.output)
    engine = get_engine(args)
    with SQLConn(args.output) as sql_out:
        sql_out.exec(_create_table)
        with SQLConn(*args.input) as sql_in:
            count = sql_in.row_max_count('position')

            query = 'SELECT epd, prev, move, uci, cnt, win, loss FROM position'

            if args.popularity_threshold:
                query += f' WHERE cnt >= {args.popularity_threshold}'

            if args.reverse:
                query += ' ORDER BY _rowid_ DESC'

            for i, row in tenumerate(sql_in.exec(query), start=1, total=count, desc='Analysing'):
                if (i + args.offset) % args.step:
                    continue

                if not args.no_skip_existing:
                    res = sql_out.exec(f'''
                        SELECT EXISTS(SELECT (epd) FROM position WHERE epd="{row[0]}")'''
                    )
                    # skip over existing rows
                    if res.fetchone()[0]:
                        continue
                # Apply the filtering criteria to analyze only positions that meet the criteria
                if not filter_positions(args, row):
                    continue

                try:
                    analyse(args, sql_out, engine, row[0])
                except KeyboardInterrupt:
                    raise
                except:
                    try:
                        engine.quit()
                    except:
                        pass
                    # attempt to recover; reinitialize engine
                    engine = get_engine(args)
                    continue

                if i / args.step % 1000 == 0:
                    sql_out.commit()
    engine.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1, help='sqlite3 database')
    parser.add_argument('-c', '--cleanup', action='store_true', help='delete database if it exists')
    parser.add_argument('-d', '--depth', type=int)
    parser.add_argument('-e', '--engine', default='stockfish')
    parser.add_argument('--hash', type=int, default=1024, help='engine hashtable size in MB')
    parser.add_argument('-m', '--mate-score', type=int, default=29999)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-r', '--reverse', action='store_true')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('-s', '--step', type=int, help='sample every step-th row', default=1)
    parser.add_argument('-t', '--time-limit', type=float, default=0.1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--no-skip-existing', action='store_true')
    parser.add_argument('--nnue', dest='nnue', action='store_true', default=True)
    parser.add_argument('--no-nnue', dest='nnue', action='store_false')
    # Add command-line parameters for filtering criteria
    parser.add_argument('--popularity-threshold', type=int, help='minimum position occurrences')
    parser.add_argument('--min-win-rate', type=float, help='minimum win rate (0 to 1)')
    parser.add_argument('--max-loss-rate', type=float, help='maximum loss rate (0 to 1)')

    args = parser.parse_args()
    if args.min_win_rate:
        assert 0 < args.min_win_rate <= 1, args.min_win_rate
    if args.max_loss_rate:
        assert 0 < args.max_loss_rate <= 1, args.max_loss_rate

    try:
        main(args)
    except KeyboardInterrupt:
        pass
