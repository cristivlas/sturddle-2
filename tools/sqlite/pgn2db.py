#!/usr/bin/env python3
# Construct training DBs from PGNs that contain evaluations.
import argparse
import os
import re
import chess
import chess.pgn as pgn
import logging
import random
from dbutils.sqlite import SQLConn
from tqdm import tqdm


class Statistics:
    def __init__(self, args):
        self.color = args.color
        self.positions_parsed = 0
        self.positions_filtered_check = 0
        self.positions_filtered_capture = 0
        self.positions_filtered_color = 0
        self.positions_filtered_limit = 0
        self.positions_inserted = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.file_size = 0
        self.estimated_avg_game_size = 0
        self.games = 0
        self.games_with_no_eval = 0
        self.games_with_no_result = 0
        self.games_below_min_elo = 0
        self.games_lost_on_time = 0

    def log_summary(self):
        total_filtered = self.positions_filtered_check + self.positions_filtered_capture + self.positions_filtered_color + self.positions_filtered_limit
        total_games = self.games
        positions_inserted = self.positions_inserted

        logging.info("=" * 70)
        logging.info("Processing Summary:")
        logging.info("=" * 70)

        if self.color is not None:
            logging.info(f"Filtered by color: {self.color}")

        # Report actual vs estimated game size
        if total_games > 0 and self.file_size > 0:
            actual_avg_game_size = self.file_size / total_games
            logging.info(f"Average game size: {actual_avg_game_size:.1f} bytes (estimated: {self.estimated_avg_game_size} bytes)")
            logging.info("-" * 70)

        logging.info(f"Total positions parsed: {self.positions_parsed:,}")

        if self.positions_parsed > 0:
            duplicates = self.positions_parsed - self.positions_inserted - total_filtered
            logging.info(f"Duplicates: {duplicates:,} ({duplicates/self.positions_parsed*100:.2f}%)")
            logging.info(f"Total positions filtered: {total_filtered:,} ({total_filtered/self.positions_parsed*100:.2f}%)")
            logging.info(f"  - Filtered (check): {self.positions_filtered_check:,} ({self.positions_filtered_check/self.positions_parsed*100:.2f}%)")
            logging.info(f"  - Filtered (capture): {self.positions_filtered_capture:,} ({self.positions_filtered_capture/self.positions_parsed*100:.2f}%)")
            logging.info(f"  - Filtered (color): {self.positions_filtered_color:,} ({self.positions_filtered_color/self.positions_parsed*100:.2f}%)")
            logging.info(f"  - Filtered (eval limit): {self.positions_filtered_limit:,} ({self.positions_filtered_limit/self.positions_parsed*100:.2f}%)")
            logging.info(f"Positions inserted: {positions_inserted:,} ({positions_inserted/self.positions_parsed*100:.2f}%)")
        else:
            logging.info(f"Total positions filtered: {total_filtered:,}")
            logging.info(f"Positions inserted: {positions_inserted:,}")

        logging.info("-" * 70)
        logging.info(f"Total games processed: {total_games:,}")
        logging.info(f"Games without eval: {self.games_with_no_eval:,}")
        logging.info(f"Games with no result: {self.games_with_no_result:,}")
        logging.info(f"Games below minimum ELO: {self.games_below_min_elo:,}")
        logging.info(f"Games lost on time: {self.games_lost_on_time:,}")

        if total_games > 0:
            logging.info(f"  - White wins: {self.white_wins:,} ({self.white_wins/total_games*100:.2f}%)")
            logging.info(f"  - Black wins: {self.black_wins:,} ({self.black_wins/total_games*100:.2f}%)")
            logging.info(f"  - Draws: {self.draws:,} ({self.draws/total_games*100:.2f}%)")
        else:
            logging.info(f"  - White wins: {self.white_wins:,}")
            logging.info(f"  - Black wins: {self.black_wins:,}")
            logging.info(f"  - Draws: {self.draws:,}")

        logging.info("=" * 70)


def get_game_outcome(game, stats):
    """Extract game outcome from PGN headers"""
    result = game.headers.get('Result', '*')

    if result == '1/2-1/2':
        stats.draws += 1
        return 0, 0  # Draw
    elif result == '1-0':
        stats.white_wins += 1
        return 1, -1  # White wins, Black loses
    elif result == '0-1':
        stats.black_wins += 1
        return -1, 1  # White loses, Black wins
    else:
        stats.games_with_no_result += 1
        return None, None  # Game in progress or unknown result


def pgn_to_epd(args, game, stats):
    """
    Extracts positions with evaluations and their corresponding next move (best response)
    Returns a list of (epd, score, move_uci, move_san, move_from, move_to, outcome) tuples

    IMPORTANT: This code assumes the eval in the comment is for the position BEFORE the
    move is made (based on the cutechess-cli output which matches the UCI engine response)
    but THIS COULD BE A WRONG ASSUMPTION (especially for lichess-annotated games).
    """
    lichess = args.lichess
    if game.headers.get('Site', '').lower().startswith('https://lichess.org/'):
        logging.debug('using lichess.org eval format')
        lichess = True

    board = game.board()
    epd_list = []

    mate_score = args.mate_score

    # Get game outcome
    white_outcome, black_outcome = get_game_outcome(game, stats)
    if white_outcome is None:
        return epd_list  # Skip games without clear outcomes

    for current_node in game.mainline():
        stats.positions_parsed += 1

        # Current board position
        epd = board.epd()

        # Side for whom the evaluation applies
        side_to_move = board.turn

        # Get the move from current position
        current_move = current_node.move

        if (args.color == 'white' and board.turn != chess.WHITE) or (args.color == 'black' and board.turn != chess.BLACK):
            board.push(current_move)
            stats.positions_filtered_color += 1
            logging.debug(f'filtered out: {epd}')
            continue

        # Check if this position has an evaluation
        if lichess:
            comment = current_node.comment.strip()
            if not comment.startswith('[%eval'):
                comment = None
        else:
            comment = current_node.comment.split('/')[0].strip()
        if not comment:
            # If no evaluation, skip the rest of the game
            stats.games_with_no_eval += 1
            logging.debug(f'no eval, skipping game {game.headers.get("GameStartTime", "")}')
            break

        if comment.lower().startswith('book'):
            board.push(current_move)
            continue

        if args.no_check:
            if board.is_check():
                logging.debug(f'skip in-check position: {epd}')
                stats.positions_filtered_check += 1
                board.push(current_move)
                continue
            if board.gives_check(current_move):
                logging.debug(f'skip check: {epd}, {current_move}')
                stats.positions_filtered_check += 1
                board.push(current_move)
                continue

        if args.no_capture and board.is_capture(current_move):
            logging.debug(f'skip capture: {epd}, {current_move}')
            stats.positions_filtered_capture += 1
            board.push(current_move)
            continue

        # assert side_to_move == board.turn

        move_san = board.san(current_move)

        # Make the move
        board.push(current_move)

        move_uci = current_move.uci()
        move_from = current_move.from_square
        move_to = current_move.to_square

        # logging.debug(comment)
        # Parse score in centipawns from side-to-move perspective
        # Format: { +0.26/3 0.044s } or { -0.20/3 0.048s }
        comment = re.sub(r'\([^)]*\)', '', comment).strip()
        # logging.debug(comment)

        if lichess:
            score_str = comment.split()[1][:-1]
        else:
            parts = comment.strip('{}').split()
            if not parts:
                continue
            score_str = parts[0]

        logging.debug(f'{epd}, {current_move}, {move_san}, {score_str}')

        if args.convert_comma:
            score_str = score_str.replace(',', '.')

        if score_str.startswith('#') or score_str.startswith('M') or score_str.startswith('+M') or score_str.startswith('-M'):
            if not mate_score:
                continue

            # Remove leading +/- and # to get mate in moves
            mate_str = score_str.lstrip('+-M#')
            mate_in = int(mate_str)
            # Preserve the sign from the original score
            sign = -1 if score_str.startswith('-') else 1
            score = int(sign * (mate_score - abs(mate_in)))
            # logging.debug(f'mate score: {score}')
        else:
            score = int(float(score_str) * 100)

        if args.limit is not None and abs(score) > args.limit:
            stats.positions_filtered_limit += 1
            continue

        # Lichess evaluation is from white's perspective in the PGN
        # If side-to-move is black, we need to negate the score to get black's perspective
        if lichess and not side_to_move:
            # logging.debug("flip score to black's perspective")
            score = -score

        # Get outcome from side-to-move perspective
        outcome = white_outcome if side_to_move else black_outcome
        logging.debug(f"{['black', 'white'][side_to_move]} score: {score}, result: {outcome} ({game.headers.get('Result', '*')})")

        epd_list.append((epd, score, move_uci, move_san, move_from, move_to, outcome))

    return epd_list


def check_minimum_elo(args, game):
    min_elo = args.min_elo or 0

    for player in ['White', 'Black']:
        elo_header = f'{player}Elo'
        elo = int(game.headers.get(elo_header, '0'))
        if elo and elo < min_elo:
            logging.debug(f'{elo_header}: {elo} < {min_elo}')
            return False

    return True


def main(args, stats):
    with SQLConn(args.output) as sqlconn:
        # Use EPD as primary key to eliminate duplicates.
        pk = 'PRIMARY KEY' if args.unique else ''

        sqlconn.exec(f'''
            CREATE TABLE IF NOT EXISTS position(
                epd text {pk},
                score integer,
                best_move_uci text,
                best_move_san text,
                best_move_from integer,
                best_move_to integer,
                outcome integer
            )''')

        # Estimate total number of games based on file size
        file_size = os.path.getsize(args.pgn_file)
        avg_game_size = args.game_size  # bytes
        num_games = file_size // avg_game_size

        # Store for statistics
        stats.file_size = file_size
        stats.estimated_avg_game_size = avg_game_size

        logging.info(f"Processing PGN file: {args.pgn_file}")
        logging.info(f"Output database: {args.output}")
        logging.info(f"Estimated games: {num_games}")

        if args.shuffle:
            logging.info("Shuffle mode: ENABLED (shuffling positions within each game)")

        # Open PGN file and process games sequentially
        with open(args.pgn_file, 'r') as pgn_data:
            game_iter = iter(lambda: pgn.read_game(pgn_data), None)

            for game in tqdm(game_iter, total=num_games):
                if game is None:
                    continue

                stats.games += 1

                if not check_minimum_elo(args, game):
                    stats.games_below_min_elo += 1
                    continue

                if game.headers.get('Termination', '') == 'time forfeit':
                    stats.games_lost_on_time += 1
                    continue

                epd_list = pgn_to_epd(args, game, stats)

                if not epd_list:
                    continue

                # Shuffle positions within this game if requested
                if args.shuffle:
                    random.shuffle(epd_list)

                for epd, cp_score, best_move_uci, best_move_san, best_move_from, best_move_to, outcome in epd_list:
                    if args.limit is not None:
                        assert abs(cp_score) <= args.limit

                    csr = sqlconn.exec('''
                        INSERT OR IGNORE INTO position(epd, score, best_move_uci, best_move_san, best_move_from, best_move_to, outcome)
                        VALUES(?, ?, ?, ?, ?, ?, ?)''',
                        (epd, int(cp_score), best_move_uci, best_move_san, best_move_from, best_move_to, outcome))

                    stats.positions_inserted += csr.rowcount

                if stats.positions_inserted % args.commit_threshold == 0:
                    sqlconn.commit()

            # Final commit for any remaining games
            sqlconn.commit()

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
    parser.add_argument('-c', '--commit-threshold', type=int, default=10000, help='commit when number of inserted positions exceeds this value')
    parser.add_argument('-g', '--game-size', default=2950, type=int, help='estimated bytes per game in PGN file (default: 2950)')
    parser.add_argument('-o', '--output', help='sqlite3 output file (default: input filename with .db extension)')
    parser.add_argument('-v', '--debug', action='store_true')
    parser.add_argument('--color', choices=['black', 'white'], help="add only moves from the specified side to the database")
    parser.add_argument('--convert-comma', action='store_true', help='replace comma with decimal point in score')
    parser.add_argument('--lichess', action='store_true', help='parse lichess eval format (https://database.lichess.org/)')
    parser.add_argument('--limit', type=int, help='absolute eval limit, in centipawns')
    parser.add_argument('--logfile', type=str, help='log file (default: output filename with .log extension)')
    parser.add_argument('--mate-score', type=int, default=15000, help='mate score in centipawns (default: 15000, 0 skips close-to-mate positions)')
    parser.add_argument('--min-elo', type=int, help='if specified, only include games with minimum player ELO')
    parser.add_argument('--no-capture', action='store_true', help='exclude capturing moves')
    parser.add_argument('--no-check', action='store_true', help='exclude in-check position and checking moves')
    parser.add_argument('--shuffle', action='store_true', help='shuffle positions within each game before inserting into database')
    parser.add_argument('--unique', action='store_true', dest='unique', default=True, help="store unique positions (use EPD as primary key, default: True)")
    parser.add_argument('--no-unique', action='store_false', dest='unique', help="allow multiple entries for a position (no EPD primary key)")

    args = parser.parse_args()

    # Infer output filenames from input if not provided
    def infer_output_name(filename, extension):
        base_name = os.path.splitext(filename)[0]
        return base_name + extension

    if args.output is None:
        args.output = infer_output_name(os.path.basename(args.pgn_file), '.db')

    if args.logfile is None:
        args.logfile = infer_output_name(args.output, '.log')

    configure_logging(args)

    stats = Statistics(args)

    try:
        main(args, stats)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        logging.warning("User interrupted")
        stats.log_summary()
