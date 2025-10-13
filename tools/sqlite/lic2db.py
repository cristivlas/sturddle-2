#!/usr/bin/env python3
# Construct training DBs from Lichess PGNs that contain evaluations (https://database.lichess.org/).
import argparse
import os

import chess.pgn as pgn

from dbutils.sqlite import SQLConn
from math import copysign
from tqdm import tqdm


def get_game_outcome(game):
    '''
    Extract game outcome from PGN headers
    Returns outcome from side-to-move perspective: 1 for win, 0 for draw, -1 for loss
    '''
    result = game.headers.get('Result', '*')

    if result == '1/2-1/2':
        return 0, 0  # Draw
    elif result == '1-0':
        return 1, -1  # White wins, Black loses
    elif result == '0-1':
        return -1, 1  # White loses, Black wins
    else:
        return None, None  # Game in progress or unknown result


def pgn_to_epd(args, game):
    '''
    Extracts positions with evaluations and their corresponding next move (best response)
    Returns a list of (epd, score, next_move_uci, next_move_san, next_move_from, next_move_to, outcome) tuples
    '''
    board = game.board()
    epd_list = []

    mate_score = args.mate_score

    # Get game outcome
    white_outcome, black_outcome = get_game_outcome(game)
    if white_outcome is None:
        return epd_list  # Skip games without clear outcomes

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

        if args.no_check and (board.is_check() or board.gives_check(next_move)):
            continue

        if args.no_capture and board.is_capture(next_move):
            continue

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

        if args.limit is not None and abs(score) > args.limit:
            continue

        # The evaluation is from white's perspective in the PGN
        # If side-to-move is black, we need to negate the score to get black's perspective
        if not side_to_move:  # False is black's turn
            score = -score

        # Get outcome from side-to-move perspective
        outcome = white_outcome if side_to_move else black_outcome

        epd_list.append((epd, score, next_move_uci, next_move_san, next_move_from, next_move_to, outcome))

    return epd_list


def main(args):
    with SQLConn(args.output) as sqlconn:
        # Update the table schema to include move information and game outcome.
        # Use EPD as primary key to eliminate duplicates.
        sqlconn.exec('''CREATE TABLE IF NOT EXISTS position(
                        epd text PRIMARY KEY,
                        score integer,
                        best_move_uci text,
                        best_move_san text,
                        best_move_from integer,
                        best_move_to integer,
                        outcome integer
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

                epd_list = pgn_to_epd(args, game)

                if not epd_list:
                    continue

                for epd, cp_score, best_move_uci, best_move_san, best_move_from, best_move_to, outcome in epd_list:
                    if args.limit is not None:
                        assert abs(cp_score) <= args.limit

                    sqlconn.exec('''INSERT OR IGNORE INTO position(epd, score, best_move_uci, best_move_san, best_move_from, best_move_to, outcome)
                                    VALUES(?, ?, ?, ?, ?, ?, ?)''',
                                    (epd, int(cp_score), best_move_uci, best_move_san, best_move_from, best_move_to, outcome))

                game_count += 1
                if game_count % 10000 == 0:
                    sqlconn.commit()

            # Final commit for any remaining games
            sqlconn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse PGN file and output SQLite database of evaluated positions')
    parser.add_argument('pgn_file', help='PGN input file')
    parser.add_argument('-o', '--output', required=True, help='SQLite output file')
    parser.add_argument('--limit', type=int, help='absolute eval limit')
    parser.add_argument('--mate-score', type=int, default=15000, help='Mate score in centipawns (default: 15000, 0 ignores mate scores)')
    parser.add_argument('--no-capture', action='store_true')
    parser.add_argument('--no-check', action='store_true')

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
