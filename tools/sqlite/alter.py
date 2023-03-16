#! /usr/bin/env python3
import argparse
import sqlite3
import chess

from tqdm.contrib import tenumerate


def main(args):
    # Connect to the database
    conn = sqlite3.connect(args.database)

    # Add the new columns to the position table
    conn.execute('ALTER TABLE position ADD COLUMN piece_count INTEGER')
    conn.execute('ALTER TABLE position ADD COLUMN white_count INTEGER')
    conn.execute('ALTER TABLE position ADD COLUMN black_count INTEGER')
    conn.execute('ALTER TABLE position ADD COLUMN side_to_move TEXT')

    count = conn.execute('SELECT count(*) from position').fetchall()[0][0]

    # Select all the rows from the position table
    cursor = conn.execute('SELECT * FROM position')

    # Loop through the rows and update the new columns
    for _, row in tenumerate(cursor, start=1, total=count):
        # Parse the EPD string into a chess.Board object
        board = chess.Board(row[0])

        # Count the number of pieces on the board
        piece_count = chess.popcount(board.occupied)

        # Count the number of white and black pieces
        white_count = chess.popcount(board.occupied_co[chess.WHITE])
        black_count = chess.popcount(board.occupied_co[chess.BLACK])

        # Get the side to move
        side_to_move = 'w' if board.turn == chess.WHITE else 'b'

        # Update the row with the new values
        conn.execute('UPDATE position SET piece_count=?, white_count=?, black_count=?, side_to_move=? WHERE epd=?',
                     (piece_count, white_count, black_count, side_to_move, row[0]))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add columns to a SQLite database with the results of analyzing chess positions.')
    parser.add_argument('database', type=str, help='The path to the SQLite database.')
    args = parser.parse_args()
    main(args)
