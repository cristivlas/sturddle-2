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
from tqdm import tqdm

_create_table = '''CREATE TABLE IF NOT EXISTS position(
    epd text PRIMARY KEY,   -- Position
    depth integer,          -- Analysis Depth
    score integer           -- Score
)'''

_insert = '''INSERT INTO position(epd, depth, score) VALUES(?,?,?)'''

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
        config['Use NNUE'] = 'true' if args.nnue else 'false'

    engine.configure(config)
    return engine

def main(args):
    engine = get_engine(args)
    with SQLConn(args.output) as sql_out:
        sql_out.exec(_create_table)

        for filepath in args.input:
            print(filepath)
            with open(filepath, 'r') as f:
                lines = f.readlines()

                for i in tqdm(range(len(lines)), maxinterval=1, desc='Analysing', dynamic_ncols=True):
                    row = lines[i]
                    cols = row.split(args.delimiter)

                    if len(cols) <= args.epd_index:
                        continue

                    epd = cols[args.epd_index].strip()
                    res = sql_out.exec(f'''SELECT EXISTS(SELECT (epd) FROM position WHERE epd="{epd}")''')
                    # skip over existing rows
                    if res.fetchone()[0]:
                        continue
                    try:
                        analyse(args, sql_out, engine, epd)
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

                    if i % 1000 == 0:
                        sql_out.commit()
    engine.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', help='text file')
    parser.add_argument('--depth', '-d', type=int)
    parser.add_argument('--delimiter', default='bm')
    parser.add_argument('--engine', '-e', default='stockfish')
    parser.add_argument('--epd-index', type=int, default=0)
    parser.add_argument('--hash', type=int, default=1024, help='engine hashtable size in MB')
    parser.add_argument('--mate-score', '-m', type=int, default=29999)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--time-limit', '-t', type=float, default=0.1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--nnue', dest='nnue', action='store_true', default=True)
    parser.add_argument('--no-nnue', dest='nnue', action='store_false')

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        pass
