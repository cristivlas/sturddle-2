#!/usr/bin/env python3
'''
Prepare data read from sqlite3 database for use by a neural net.
Positions are converted from EPD into features, and stored as a
memory-mapped numpy array or an h5py dataset.
'''
import argparse
import os
import sys

import chess
import h5py
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sqlite')))
from dbutils.sqlite import SQLConn
from tqdm.contrib import tenumerate

'''
Convert chess.Board to array of features.
Looses castling rights and en-passant square info.
'''
def encode(board, test=False):
    mask_black = board.occupied_co[chess.BLACK]
    mask_white = board.occupied_co[chess.WHITE]

    bitboards = [[pcs & mask_black, pcs & mask_white] for pcs in (
        board.kings,
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens)
    ]
    f = np.unpackbits(np.asarray(bitboards, dtype='>u8').reshape(12,).view(np.uint8))
    f = np.append(f, [board.turn])
    # builtin unit test
    if test:
        board.castling_rights = 0
        board.ep_square = None
        expected = board.epd()
        actual = decode(f).epd()
        assert expected == actual, (expected, actual)
    return f


'''
Convert encoding back to board, for verification.
'''
def decode(f):
    turn = f[768]
    bitboards = [int(x) for x in list(np.packbits(f[:768]).view(np.uint64).byteswap())]
    assert len(bitboards) == 12, bitboards
    board = chess.Board(fen=None)
    for b in bitboards:
        board.occupied |= b
    for b in bitboards[::2]:
        board.occupied_co[chess.BLACK] |= b
    for b in bitboards[1::2]:
        board.occupied_co[chess.WHITE] |= b
    board.kings = bitboards[0] | bitboards[1]
    board.pawns = bitboards[2] | bitboards[3]
    board.knights = bitboards[4] | bitboards[5]
    board.bishops = bitboards[6] | bitboards[7]
    board.rooks = bitboards[8] | bitboards[9]
    board.queens = bitboards[10] | bitboards[11]
    board.turn = turn
    return board


def main(args):
    clip = args.clip
    with SQLConn(*args.input) as sql:
        count = sql.row_count('position')
        dtype = np.float16 if args.half else np.float32
        if args.h5:
            f = h5py.File(args.output, 'x')
            out = f.create_dataset('data', shape=(count, 770), dtype=dtype)
        else:
            out = np.memmap(args.output, dtype=dtype, mode='w+', shape=(count,770))

        query = 'SELECT epd, score from position'
        if args.randomize:
            query += ' ORDER BY RANDOM()'

        board = chess.Board()
        for i, row in tenumerate(sql.exec(query), start=0, total=count, desc='Encoding'):
            board.set_fen(row[0])
            score = np.clip(row[1], -clip, clip) / 100
            out[i, :-1] = encode(board, args.test).astype(dtype)
            out[i, -1] = score.astype(dtype)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Convert sqlite3 db to numpy array')
        parser.add_argument('input', nargs=1)
        parser.add_argument('-c', '--clip', type=int, default=2000)
        parser.add_argument('-d', '--debug', action='store_true')
        parser.add_argument('-o', '--output', required=True)
        parser.add_argument('-r', '--randomize', action='store_true')
        parser.add_argument('-t', '--test', action='store_true')
        parser.add_argument('--half', action='store_true')
        parser.add_argument('--h5', action='store_true')

        args = parser.parse_args()
        if os.path.splitext(args.output)[1].lower() == '.h5':
            print('Using H5 format.')
            args.h5 = True

        main(args)

    except KeyboardInterrupt:
        pass