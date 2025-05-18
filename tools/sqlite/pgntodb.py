#!/usr/bin/env python3
# Construct training DBs from PGNs that contain evaluations.
import argparse
import os

import chess
import chess.pgn as pgn

from dbutils.sqlite import SQLConn
from math import copysign
from tqdm import tqdm


def pgn_to_epd(game, mate_score):
    '''
    Extracts positions with evaluations and their corresponding next move (best response)
    Returns a list of (epd, score, next_move_uci, next_move_san, next_move_from, next_move_to) tuples
    '''
    board = game.board()
    epd_list = []

    # We need to look at pairs of nodes to connect an evaluated position with the next move
    nodes = list(game.mainline())

    for i in range(len(nodes) - 1):  # Stop one before the end to ensure there's a next move
        current_node = nodes[i]
        next_node = nodes[i + 1]

        # Get the move from current position and make it
        current_move = current_node.move
        board.push(current_move)

        # Check if this position has an evaluation
        comment = current_node.comment.strip()
        if not comment.startswith('[%eval'):
            # If no evaluation, skip the rest of the game
            break

        # Current position after the move has been made
        epd = board.epd()

        # Get the next move (best response to the current position)
        next_move = next_node.move
        next_move_san = board.san(next_move)
        next_move_uci = next_move.uci()
        next_move_from = next_move.from_square
        next_move_to = next_move.to_square

        # Current side to move (for whom the evaluation applies)
        side_to_move = board.turn

        # Parse score in centipawns
        score_str = comment.split()[1][:-1]
        if score_str.startswith('#'):
            if not mate_score:
                continue

            # Convert mate score to centipawns
            mate_in = int(score_str[1:])
            score = int(copysign(mate_score, mate_in) - mate_in)
        else:
            score = int(float(score_str) * 100)

        # The evaluation is from white's perspective in the PGN
        # If side-to-move is black, we need to negate the score to get black's perspective
        if not side_to_move:  # False is black's turn
            score = -score

        epd_list.append((epd, score, next_move_uci, next_move_san, next_move_from, next_move_to))

    return epd_list

    return epd_list


def main(args):
    with SQLConn(args.output) as sqlconn:
        # Update the table schema to include move information
        sqlconn.exec('''CREATE TABLE IF NOT EXISTS position(
                        epd text PRIMARY KEY,
                        depth integer,
                        score integer,
                        best_move_uci text,
                        best_move_san text,
                        best_move_from integer,
                        best_move_to integer
                        )''')

        # Estimate total number of games based on file size
        file_size = os.path.getsize(args.pgn_file)
        avg_game_size = 2308  # bytes (adjust this value as needed)
        num_games = file_size // avg_game_size

        # Open PGN file
        with open(args.pgn_file, 'r') as pgn_data:
            game_iter = iter(lambda: pgn.read_game(pgn_data), None)
            game_count = 0
            for game in tqdm(game_iter, total=num_games):
                if game is None:
                    continue

                epd_list = pgn_to_epd(game, args.mate_score)

                if not epd_list:
                    continue

                for epd, cp_score, best_move_uci, best_move_san, best_move_from, best_move_to in epd_list:
                    sqlconn.exec('''INSERT OR IGNORE INTO position(epd, depth, score, best_move_uci, best_move_san, best_move_from, best_move_to)
                                    VALUES(?, ?, ?, ?, ?, ?, ?)''',
                                    (epd, -2, int(cp_score), best_move_uci, best_move_san, best_move_from, best_move_to))

                game_count += 1
                # Commit every 100 games with evals
                if game_count % 100 == 0:
                    sqlconn.commit()

            # Final commit for any remaining games
            sqlconn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse PGN file and output SQLite database of evaluated positions')
    parser.add_argument('pgn_file', help='PGN input file')
    parser.add_argument('-o', '--output', required=True, help='SQLite output file')
    parser.add_argument('--mate-score', type=int, default=15000, help='Mate score in centipawns (default: 15000, 0 ignores mate scores)')
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
