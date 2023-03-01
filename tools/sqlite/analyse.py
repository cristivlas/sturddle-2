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


'''
Analyse one position given by EPD, and insert score into the output database.
'''
def analyse(args, sql_out, engine, epd):
    board = chess.Board(fen=epd)
    if not board.is_valid():
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
        config['Use NNUE'] = args.nnue

    engine.configure(config)
    return engine


def main(args):
    if args.cleanup and os.path.exists(args.output):
        os.unlink(args.output)
    engine = get_engine(args)
    with SQLConn(args.output) as sql_out:
        sql_out.exec(_create_table)
        with SQLConn(*args.input) as sql_in:
            count = sql_in.row_max_count('position')
            if args.reverse:
                query = '''SELECT DISTINCT(epd) FROM position ORDER BY _rowid_ DESC'''
            else:
                query = '''SELECT DISTINCT(epd) FROM position'''
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

                try:
                    analyse(args, sql_out, engine, *row)
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
    parser.add_argument('-m', '--mate-score', type=int)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-r', '--reverse', action='store_true')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('-s', '--step', type=int, help='sample every step-th row', default=1)
    parser.add_argument('-t', '--time-limit', type=float, default=0.1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--no-skip-existing', action='store_true')
    parser.add_argument('--nnue', dest='nnue', action='store_true', default=True)
    parser.add_argument('--no-nnue', dest='nnue', action='store_false')

    try:
        main(parser.parse_args())
    except KeyboardInterrupt:
        pass
