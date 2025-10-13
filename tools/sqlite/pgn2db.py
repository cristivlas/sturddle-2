#!/usr/bin/env python3
# Construct training DBs from PGNs that contain evaluations.
import argparse
import os
import chess.pgn as pgn
import logging
from dbutils.sqlite import SQLConn
from tqdm import tqdm


class Statistics:
    def __init__(self):
        self.positions_parsed = 0
        self.positions_filtered_check = 0
        self.positions_filtered_capture = 0
        self.positions_filtered_limit = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.file_size = 0
        self.estimated_avg_game_size = 0

    def log_summary(self):
        total_filtered = self.positions_filtered_check + self.positions_filtered_capture + self.positions_filtered_limit
        total_games = self.white_wins + self.black_wins + self.draws
        positions_inserted = self.positions_parsed - total_filtered

        logging.info("=" * 70)
        logging.info("Processing Summary:")
        logging.info("=" * 70)

        # Report actual vs estimated game size
        if total_games > 0 and self.file_size > 0:
            actual_avg_game_size = self.file_size / total_games
            logging.info(f"Average game size: {actual_avg_game_size:.1f} bytes (estimated: {self.estimated_avg_game_size} bytes)")
            logging.info("-" * 70)

        logging.info(f"Total positions parsed: {self.positions_parsed:,}")

        if self.positions_parsed > 0:
            logging.info(f"Total positions filtered: {total_filtered:,} ({total_filtered/self.positions_parsed*100:.2f}%)")
            logging.info(f"  - Filtered (check): {self.positions_filtered_check:,} ({self.positions_filtered_check/self.positions_parsed*100:.2f}%)")
            logging.info(f"  - Filtered (capture): {self.positions_filtered_capture:,} ({self.positions_filtered_capture/self.positions_parsed*100:.2f}%)")
            logging.info(f"  - Filtered (eval limit): {self.positions_filtered_limit:,} ({self.positions_filtered_limit/self.positions_parsed*100:.2f}%)")
            logging.info(f"Positions inserted: {positions_inserted:,} ({positions_inserted/self.positions_parsed*100:.2f}%)")
        else:
            logging.info(f"Total positions filtered: {total_filtered:,}")
            logging.info(f"Positions inserted: {positions_inserted:,}")

        logging.info("-" * 70)
        logging.info(f"Total games processed: {total_games:,}")

        if total_games > 0:
            logging.info(f"  - White wins: {self.white_wins:,} ({self.white_wins/total_games*100:.2f}%)")
            logging.info(f"  - Black wins: {self.black_wins:,} ({self.black_wins/total_games*100:.2f}%)")
            logging.info(f"  - Draws: {self.draws:,} ({self.draws/total_games*100:.2f}%)")
        else:
            logging.info(f"  - White wins: {self.white_wins:,}")
            logging.info(f"  - Black wins: {self.black_wins:,}")
            logging.info(f"  - Draws: {self.draws:,}")

        logging.info("=" * 70)


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


def pgn_to_epd(args, game, stats):
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

    # Track game outcomes
    result = game.headers.get('Result', '*')
    if result == '1-0':
        stats.white_wins += 1
    elif result == '0-1':
        stats.black_wins += 1
    elif result == '1/2-1/2':
        stats.draws += 1

    # We need to look at pairs of nodes to connect an evaluated position with the next move
    nodes = list(game.mainline())

    for i in range(len(nodes) - 1):  # Stop one before the end to ensure there's a next move
        current_node = nodes[i]
        next_node = nodes[i + 1]

        # Get the move from current position and make it
        current_move = current_node.move
        board.push(current_move)

        # Check if this position has an evaluation
        comment = current_node.comment.split('/')[0].strip()
        if not comment:
            # If no evaluation, skip the rest of the game
            break

        stats.positions_parsed += 1

        # Current position after the move has been made
        epd = board.epd()

        # Get the next move (best response to the current position)
        next_move = next_node.move

        if args.no_check:
            if board.is_check():
                logging.debug(f'skip in-check position: {board.fen()}')
                stats.positions_filtered_check += 1
                continue
            if board.gives_check(next_move):
                logging.debug(f'skip check: {board.fen()} {next_move}')
                stats.positions_filtered_check += 1
                continue

        if args.no_capture and board.is_capture(next_move):
            logging.debug(f'skip capture: {board.fen()} {next_move}')
            stats.positions_filtered_capture += 1
            continue

        next_move_san = board.san(next_move)
        next_move_uci = next_move.uci()
        next_move_from = next_move.from_square
        next_move_to = next_move.to_square

        # Current side to move (for whom the evaluation applies)
        side_to_move = board.turn

        # Parse score in centipawns from side-to-move perspective
        # Format: { +0.26/3 0.044s } or { -0.20/3 0.048s }
        parts = comment.strip('{}').split()
        score_str = parts[0]

        if score_str.startswith('M') or score_str.startswith('+M') or score_str.startswith('-M'):
            if not mate_score:
                continue

            # Remove leading +/- and # to get mate in moves
            mate_str = score_str.lstrip('+-M')
            mate_in = int(mate_str)
            # Preserve the sign from the original score
            sign = -1 if score_str.startswith('-') else 1
            score = int(sign * (mate_score - abs(mate_in)))
            # logging.debug(f'mate score: {score}')
        else:
            score = int(float(score_str) * 100)

        # color = ['white', 'black']
        # logging.debug(f'{color[side_to_move]}: {score}')

        if args.limit is not None and abs(score) > args.limit:
            stats.positions_filtered_limit += 1
            continue

        # Get outcome from side-to-move perspective
        outcome = white_outcome if side_to_move else black_outcome

        epd_list.append((epd, score, next_move_uci, next_move_san, next_move_from, next_move_to, outcome))

    return epd_list


def main(args):
    stats = Statistics()

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
        avg_game_size = 3500  # bytes
        num_games = file_size // avg_game_size

        # Store for statistics
        stats.file_size = file_size
        stats.estimated_avg_game_size = avg_game_size

        logging.info(f"Processing PGN file: {args.pgn_file}")
        logging.info(f"Output database: {args.output}")
        logging.info(f"Estimated games: {num_games}")

        # Open PGN file
        with open(args.pgn_file, 'r') as pgn_data:
            game_iter = iter(lambda: pgn.read_game(pgn_data), None)
            game_count = 0
            for game in tqdm(game_iter, total=num_games):
                if game is None:
                    continue

                epd_list = pgn_to_epd(args, game, stats)

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

    # Log statistics summary
    stats.log_summary()


def configure_logging(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        filename=args.logfile,
        format='%(asctime)s;%(levelname)s;%(message)s',
        level=log_level,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse PGN file and output SQLite database of evaluated positions')
    parser.add_argument('pgn_file', help='PGN input file')
    parser.add_argument('-o', '--output', help='SQLite output file (default: input filename with .db extension)')
    parser.add_argument('-v', '--debug', action='store_true')
    parser.add_argument('--limit', type=int, help='absolute eval limit')
    parser.add_argument('--logfile', type=str, help='Log file (default: input filename with .log extension)')
    parser.add_argument('--mate-score', type=int, default=15000, help='Mate score in centipawns (default: 15000, 0 ignores mate scores)')
    parser.add_argument('--no-capture', action='store_true')
    parser.add_argument('--no-check', action='store_true')

    args = parser.parse_args()

    # Infer output filenames from input if not provided
    if args.output is None:
        base_name = os.path.splitext(args.pgn_file)[0]
        args.output = base_name + '.db'

    if args.logfile is None:
        base_name = os.path.splitext(args.pgn_file)[0]
        args.logfile = base_name + '.log'

    configure_logging(args)

    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
