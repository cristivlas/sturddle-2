#! /usr/bin/env python3
'''
Companion utility for mkposdb.py, make Polyglot opening book.
'''
import argparse
import sqlite3
import struct
from collections import defaultdict

import chess
import chess.engine
import chess.polyglot
from dbutils.sqlite import SQLConn
from tqdm.contrib import tenumerate

"""
A Polyglot book is a series of entries of 16 bytes
key    uint64
move   uint16
weight uint16
learn  uint32

Integers are stored big endian.
"""
ENTRY_STRUCT = struct.Struct('>QHHI')


'''
Get best move from engine analysis.
'''
def analyse(args, engine, epd):
    board = chess.Board(fen=epd)
    multipv = min(3, args.max_variations)
    limit = chess.engine.Limit(depth=args.depth) if args.depth else chess.engine.Limit(time=args.time_limit)
    info = engine.analyse(board, limit, multipv=multipv)
    return [i['pv'][0] for i in info]


def encode_move(move: chess.Move) -> int:
    promotion = ((move.promotion - 1) & 0x7) << 12 if move.promotion else 0
    return move.to_square | (move.from_square<<6) | promotion


def key(epd):
    return chess.polyglot.zobrist_hash(chess.Board(fen=epd))


def polyglot_entry(key, move, weight, learn=0):
    assert learn >= 0, learn
    entry = ENTRY_STRUCT.pack(key, encode_move(move), weight % 65535, learn % 65535)
    assert len(entry)==16
    return entry


def main(args):
    # initialize optional engine
    engine = chess.engine.SimpleEngine.popen_uci(args.engine) if args.engine else None
    if engine:
        engine.configure({'Threads': args.threads})

    with SQLConn(*args.input) as sql:
        query = f'''
            SELECT {{what}},(win * 1. / cnt) AS win_ratio FROM position
            WHERE move <= {args.ply}
            AND cnt >= {args.min_sample_size} AND win_ratio >= {args.min_win_ratio}
            ORDER BY move ASC, win_ratio DESC
        '''
        count = sql.exec(query.format(what='count(*)'), echo=True).fetchone()[0]
        print(count)

        result = sql.exec(query.format(what='*'), echo=True)
        book = defaultdict(list)
        pos = {}
        msg = 'Fetching moves'
        for _, row in tenumerate(result, total=count, desc=f'{msg:>20s}'):
            k = key(row[1])
            book[k].append([chess.Move.from_uci(row[3]), int(row[7] * 100)])
            pos[k] = row[1]

        msg = f'Generating {args.output}'[-20:]
        with open(args.output, 'wb') as output:
            for _, (k, moves) in tenumerate(sorted(book.items()), desc=f'{msg:>20s}'):

                if engine:
                    # map moves to win ratio
                    moves_dict = {m[0]: m[1] for m in moves}

                    # compute best move using the engine
                    for i, move in enumerate(analyse(args, engine, pos[k])):
                        # give the computed move the highest win rate
                        moves_dict[move] = 100 - i

                    # reconstruct list
                    moves = [(m,w) for m,w in moves_dict.items()]

                moves.sort(key=lambda t: t[1], reverse=True)

                for move in moves[:args.max_variations]:
                    output.write(polyglot_entry(k, move[0], move[1]))

        if engine:
            engine.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1, help='sqlite3 database')
    parser.add_argument('-c', '--threads', type=int, default=1, help='concurrency')
    parser.add_argument('-d', '--depth', type=int, help='engine depth limit')
    parser.add_argument('-e', '--engine', help='optional engine to analyse with')
    parser.add_argument('-m', '--max-variations', type=int, default=5)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-p', '--ply', type=int, default=10, help='max ply depth')
    parser.add_argument('-s', '--min-sample-size', type=int, default=10)
    parser.add_argument('-t', '--time-limit', type=float, default=0.1, help='engine time limit')
    parser.add_argument('-w', '--min-win-ratio', type=float, default=0.25)

    try:
        main(parser.parse_args())
    except KeyboardInterrupt:
        pass
