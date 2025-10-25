#!/usr/bin/env python3
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
Convert chess.Board to array of features (packed as np.uint64)
Discard castling rights and en-passant square info.
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
    array = np.asarray([bitboards], dtype=np.uint64).ravel()
    array = np.append(array, np.uint64(board.turn))

    if test:  # run builtin unit test
        board.castling_rights = 0
        board.ep_square = None
        expected = board.epd()
        actual = decode(array).epd()
        assert expected == actual, (expected, actual)

    return array


'''
Convert encoding back to board, for verification.
'''
def decode(array):
    turn = array[12]
    bitboards = [int(x) for x in list(array[:12])]
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


def add_range_to_query(args, query):
    if args.begin or args.row_count:
        query += f' LIMIT {-1 if args.row_count is None else args.row_count}'
    if args.begin:
        query += f' OFFSET {args.begin}'
    return query


def check_schema(sql):
    """Check if the database has the new schema with best_move fields"""
    # Just check if the columns exist in the schema without loading all rows
    try:
        # Test query that will fail if columns don't exist
        sql.exec("SELECT best_move_from, best_move_to, outcome FROM position LIMIT 1")
        return True
    except:
        return False


def main(args):
    clip = args.clip
    dtype = np.uint64

    board = chess.Board()

    with SQLConn(*args.input) as sql:
        has_new_schema = check_schema(sql)

        if has_new_schema:
            print("Detected new schema with best_move fields and outcome.")
            output_shape = (0, 17)  # 13 board state + score + outcome + from + to
            query_fields = "epd, score, outcome, best_move_from, best_move_to"
        else:
            print("Detected old schema (epd, score, depth only).")
            output_shape = (0, 14)  # 13 board state + score
            query_fields = "epd, score"

        count_query = add_range_to_query(args, "SELECT * FROM position")

        if not args.begin and args.row_count:
            count = args.row_count
        else:
            count = sql.exec(f'SELECT COUNT(*) FROM ({count_query})', echo=True).fetchone()[0]

        if args.row_count is not None and count < args.row_count:
            args.row_count = count
            args.output = output_path(args)  # re-generate output file name to include actual row count

        print(f'\nWriting out: {args.output}')

        with h5py.File(args.output, 'x') as f:
            out = f.create_dataset('data', shape=(count, output_shape[1]), dtype=dtype)

            query = add_range_to_query(args, f'SELECT {query_fields} from position')
            for i, row in tenumerate(sql.exec(query, echo=True), start=0, total=count, desc='Encoding'):
                board.set_fen(row[0])
                score = np.clip(row[1], -clip, clip)
                encoded_board = encode(board, args.test).astype(dtype)

                # For old schema: just store board + score
                out[i, :13] = encoded_board
                out[i, 13] = score.astype(dtype)

                # For new schema: also store outcome and best_move_from/best_move_to
                if has_new_schema:
                    # Convert outcome from -1,0,1 to 0,1,2 for uint64 compatibility
                    outcome = row[2] if row[2] is not None else 0
                    outcome_uint = np.uint64(outcome + 1)

                    best_move_from = row[3] if row[3] is not None else 0
                    best_move_to = row[4] if row[4] is not None else 0

                    out[i, 14] = outcome_uint
                    out[i, 15] = np.uint64(best_move_from)
                    out[i, 16] = np.uint64(best_move_to)

                if i % 10000 == 0:
                    f.flush()


def format(num):
    if num >= 1e9:
        return f'{num / 1e9:.1f}G'
    elif num >= 1e6:
        return f'{num / 1e6:.1f}M'
    elif num >= 1e3:
        return f'{num / 1e3:.1f}K'
    else:
        return str(num)


def output_path(args):
    base_name = os.path.splitext(os.path.basename(args.input[0]))[0]

    if args.begin is None and args.row_count is None:
        return base_name + ".h5"

    return f"{base_name}-{format(args.begin) if args.begin else 0}-{format(args.row_count) if args.row_count else 'all'}.h5"


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Convert training data from sqlite3 to H5')
        parser.add_argument('input', nargs=1)
        parser.add_argument('-b', '--begin', type=int)
        parser.add_argument('-c', '--clip', type=int, default=15000, help="clip score (default: 15000)")
        parser.add_argument('-o', '--output')
        parser.add_argument('-r', '--row-count', type=int)
        parser.add_argument('-t', '--test', action='store_true')

        args = parser.parse_args()

        if args.output is None:
            args.output = output_path(args)

        main(args)

    except KeyboardInterrupt:
        pass
