#!/usr/bin/env python3
import argparse
import sqlite3
import warnings

import chess
import matplotlib as mpl

warnings.filterwarnings('ignore', category=mpl.MatplotlibDeprecationWarning)

import matplotlib.pyplot as plt
import numpy as np
from dbutils.sqlite import SQLConn
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
    parser.add_argument('--threshold', type=int, default=1, help='square popularity threshold')
    parser.add_argument('--viz', type=str, default=None, help='name of output visualization PNG file')

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
def update_from_move(aggregates, square, piece_type, color, endgame, cnt, wins, losses):
    j = square if color else square ^ 56
    i = piece_type - 1
    if piece_type == 6 and endgame:  # 6 is chess.KING
        i = ENDGAME_KING_TABLE
    aggregates[i][j][COUNT] += cnt
    aggregates[i][j][WINS] += wins
    aggregates[i][j][LOSSES] += losses


def update_from_board(aggregates, board, endgame, cnt, wins, losses):
    for square in chess.SquareSet(board.occupied):
        piece = board.piece_at(square)
        assert piece
        i = piece.piece_type - 1
        j = square if piece.color else square ^ 56
        if endgame and piece.piece_type == chess.KING:
            i = ENDGAME_KING_TABLE
        aggregates[i][j][COUNT] += cnt
        aggregates[i][j][WINS] += wins
        aggregates[i][j][LOSSES] += losses


@njit
def compute_piece_square_tables(aggregates, tables, scale, threshold):
    for i in range(7):
        for j in range(64):
            cnt = aggregates[i][j][COUNT]
            if cnt > threshold:
                wins = aggregates[i][j][WINS]
                losses = aggregates[i][j][LOSSES]
                score = (wins - losses) * scale / cnt
                tables[i][j] = score
    return tables


def visualize_tables(tables, scale, output_filename):
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Create a list of positions for subplots
    subplot_positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

    for i, table in enumerate(tables):
        temp_table = np.zeros((8, 8), dtype=int)
        for rank in range(8):
            for file in range(8):
                square = 8 * rank + file
                temp_table[7 - rank, file] = table[square]

        ax = fig.add_subplot(2, 4, i+1)
        ax.set_title(chess.piece_name(i + 1).capitalize() if i < ENDGAME_KING_TABLE else 'Endgame King')
        cax = ax.matshow(temp_table, cmap='seismic_r', vmin=-scale, vmax=scale)
        for (m, n), z in np.ndenumerate(temp_table):
            ax.text(n, m, f'{z}', ha='center', va='center', fontsize=9, color='black' if -50 < z < 50 else 'white')

    if output_filename:
        plt.savefig(output_filename)


def main():
    args = get_arguments()
    aggregates = np.zeros((7, 64, 3), dtype=float)
    total_count = 0

    for db_path in tqdm(args.databases, desc='Processing databases'):
        with SQLConn(db_path) as conn:
            count = conn.row_max_count('position') # use max(_rowid_) for speed

            query = 'SELECT epd, prev, uci, cnt, win, loss FROM position'
            if args.limit is not None:
                count = min(count, args.limit)
                rows = conn.exec(query + ' LIMIT ?', (args.limit,))
            else:
                rows = conn.exec(query)

            total_count += count

            for row in tqdm(rows, desc='Processing positions', leave=False, total=count):
                epd, prev, uci, cnt, win, loss = row
                move = chess.Move.from_uci(uci)
                if not move:
                    continue
                board = chess.Board(prev)
                square = move.from_square
                piece = board.piece_at(square)
                if not piece:
                    continue
                endgame = chess.popcount(board.occupied) <= 12
                if board.turn != piece.color:
                    update_from_move(aggregates, square, piece.piece_type, piece.color, endgame, cnt, loss, win)
                else:
                    update_from_move(aggregates, square, piece.piece_type, piece.color, endgame, cnt, win, loss)

                # board = chess.Board(epd)
                # if board.turn:
                #     win, loss = loss, win # DB wins are for the side that just moved.
                # endgame = chess.popcount(board.occupied) <= 12
                # update_from_board(aggregates, board, endgame, cnt, win, loss)

    piece_square_tables = np.zeros((7, 64), dtype=int)
    compute_piece_square_tables(aggregates, piece_square_tables, args.scale, args.threshold)

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

    endgame_king_table = piece_square_tables[ENDGAME_KING_TABLE]
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

    if args.viz is not None:
        visualize_tables(piece_square_tables, args.scale, args.viz)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
