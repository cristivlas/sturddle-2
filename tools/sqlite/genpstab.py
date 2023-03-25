#!/usr/bin/env python3
import argparse
import sqlite3

import chess
import numpy as np
from numba import njit
from tqdm import tqdm

COUNT = 0
WINS = 1
LOSSES = 2
ENDGAME_KING_TABLE = 6


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate piece-square tables from SQLite3 databases.')
    parser.add_argument('databases', nargs='+', help='SQLite3 database files containing chess positions.')
    parser.add_argument('-o', '--output', required=True, help='name of output header file')
    parser.add_argument('--limit', type=int, help='limit the number of entries processed')
    parser.add_argument('--scale', type=int, default=100, help='scale factor')
    return parser.parse_args()


def piece_map_to_arrays(piece_map):
    num_pieces = len(piece_map)
    squares = np.empty(num_pieces, dtype=np.int32)
    piece_types = np.empty(num_pieces, dtype=np.int32)
    colors = np.empty(num_pieces, dtype=np.bool_)

    for idx, (square, piece) in enumerate(piece_map.items()):
        squares[idx] = square
        piece_types[idx] = piece.piece_type
        colors[idx] = piece.color

    return squares, piece_types, colors


@njit
def update_aggregates(aggregates, square_to, piece_type, color, piece_count, cnt, wins, losses):
    j = square_to if color else square_to ^ 56
    i = piece_type - 1
    if piece_type == 6 and piece_count <= 12:  # 6 is chess.KING
        i = ENDGAME_KING_TABLE

    aggregates[i][j][COUNT] += cnt
    aggregates[i][j][WINS] += wins
    aggregates[i][j][LOSSES] += losses


@njit
def compute_piece_square_tables(aggregates, tables, scale):
    for i in range(7):
        for j in range(64):
            cnt = aggregates[i][j][COUNT]
            if cnt > 0:
                wins = aggregates[i][j][WINS]
                losses = aggregates[i][j][LOSSES]
                score = (wins - losses) * scale / cnt
                tables[i][j] = score

    return tables


def main():
    args = get_arguments()
    aggregates = np.zeros((7, 64, 3), dtype=int)

    for db_path in tqdm(args.databases, desc='Processing databases'):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        query = 'SELECT epd, uci, cnt, win, loss FROM position'
        if args.limit is not None:
            query += ' LIMIT ?'
            cur.execute(query, (args.limit,))
        else:
            cur.execute(query)
        for row in tqdm(cur.fetchall(), desc='Processing positions', leave=False):
            epd, uci, cnt, win, loss = row
            board = chess.Board(epd)
            move = chess.Move.from_uci(uci)
            if not move:
                continue
            to_square = move.to_square
            piece = board.piece_at(to_square)
            assert piece, (epd, uci)
            piece_count = chess.popcount(board.occupied)
            update_aggregates(aggregates, to_square, piece.piece_type, piece.color, piece_count, cnt, win, loss)
        conn.close()

    piece_square_tables = np.zeros((7, 64), dtype=int)
    compute_piece_square_tables(aggregates, piece_square_tables, args.scale)

    header_str = '#pragma once\n'
    header_str += '/*\n * Piece-square tables generated from chess databases.\n */\n'
    header_str += 'static constexpr int SQUARE_TABLE[][64] = {\n'
    header_str += '    {/* NONE */},\n'

    for i, table in enumerate(piece_square_tables[:6]):
        if i > 0:
            header_str += ',\n'
        header_str += '    {{/* {} */\n'.format(chess.piece_name(i + 1).upper())

        for rank in range(7, -1, -1):
            header_str += '        '
            for file in range(8):
                square = 8 * rank + file
                value = table[square]
                header_str += f'{value:4d},'
            header_str += '\n'
        header_str += '    }'

    endgame_king_table = piece_square_tables[6]
    header_str += '\n};\n'
    header_str += '\n'
    header_str += 'static constexpr int ENDGAME_KING_SQUARE_TABLE[64] = {\n'
    for rank in range(7, -1, -1):
        header_str += '    '
        for file in range(8):
            square = 8 * rank + file
            value = endgame_king_table[square]
            header_str += f'{value:4d},'
        header_str += '\n'
    header_str += '};\n'

    with open(args.output, 'w') as header_file:
        header_file.write(header_str)


if __name__ == '__main__':
    main()
