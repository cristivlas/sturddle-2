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

CCRL_TRANSLATE_EVAL = str.maketrans(',', '.', '#]')

class Statistics:
    def __init__(self, args):
        self.color = args.color
        self.positions_parsed = 0
        self.positions_filtered_book = 0
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
        self.games_filtered_opening = 0

    def log_summary(self):
        total_filtered = (
            self.positions_filtered_book +
            self.positions_filtered_check +
            self.positions_filtered_capture +
            self.positions_filtered_color +
            self.positions_filtered_limit
        )
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
            logging.info(f"  - Filtered (book): {self.positions_filtered_book:,} ({self.positions_filtered_book/self.positions_parsed*100:.2f}%)")
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
        logging.info(f"Games below ELO requirements: {self.games_below_min_elo:,}")
        logging.info(f"Games lost on time: {self.games_lost_on_time:,}")
        logging.info(f"Games filtered by opening: {self.games_filtered_opening:,}")

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

    IMPORTANT:
    The evaluation score is interpreted to be from the current node position, BEFORE the move
    is made -- for "normal" PGNs, assumed to be created by cutechess-cli from UCI responses,
    and AFTER the move is made -- for lichess evaluated/annotated games.
    """
    lichess = args.lichess
    if game.headers.get('Site', '').lower().startswith('https://lichess.org/'):
        # logging.debug('using lichess.org eval format')
        lichess = True

    board = game.board()
    epd_list = []

    mate_score = args.mate_score

    # Get game outcome
    white_outcome, black_outcome = get_game_outcome(game, stats)
    if white_outcome is None:
        return epd_list  # Skip games without clear outcomes

    mainline = list(game.mainline())
    ply_count = len(mainline)
    for node_index in range(ply_count):
        current_node = mainline[node_index]
        stats.positions_parsed += 1

        # Current board position
        epd = board.epd()

        # Side for whom the evaluation applies
        side_to_move = board.turn

        # Get the move from current position
        current_move = current_node.move

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
            # logging.debug(f'no eval, skipping game {game.headers.get("GameStartTime", "")}')
            assert stats.positions_parsed > 0
            stats.positions_parsed -= 1  # hack: keep stats straight
            break

        logging.debug(f'{node_index}: {epd}, {current_move}')

        if (args.color == 'white' and board.turn != chess.WHITE) or (args.color == 'black' and board.turn != chess.BLACK):
            logging.debug('--- color\n')
            stats.positions_filtered_color += 1
            board.push(current_move)
            continue

        if comment.lower().startswith('book'):
            logging.debug('--- book\n')
            stats.positions_filtered_book += 1
            board.push(current_move)
            continue  # skip book moves

        if args.no_check:
            if board.is_check():
                logging.debug('--- in-check\n')
                stats.positions_filtered_check += 1
                board.push(current_move)
                continue
            if board.gives_check(current_move):
                logging.debug('--- check\n')
                stats.positions_filtered_check += 1
                board.push(current_move)
                continue

        if args.no_capture and board.is_capture(current_move):
            logging.debug('--- capture\n')
            stats.positions_filtered_capture += 1
            board.push(current_move)
            continue

        move_san = board.san(current_move)

        # Make the move
        board.push(current_move)

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

            if parts[0].startswith('[%eval'):
                score_str = parts[1].translate(CCRL_TRANSLATE_EVAL)
                logging.debug(f'ccrl eval: {parts} {score_str}')
            else:
                score_str = parts[0]

        logging.debug(f'{epd}, {current_move}, {move_san}, {score_str}')

        if args.convert_comma:
            score_str = score_str.replace(',', '.')

        if score_str.startswith('#') or score_str.startswith('M') or score_str.startswith('+M') or score_str.startswith('-M'):
            if not mate_score:
                stats.positions_filtered_limit += 1
                continue

            # Remove leading +/- and # to get mate in moves
            mate_str = score_str.lstrip('+-M#')
            mate_in = int(mate_str)
            # Preserve the sign from the original score
            sign = -1 if score_str.startswith('-') else 1
            score = int(sign * (mate_score - abs(mate_in)))
            # logging.debug(f'mate score: {score}')
        else:
            try:
                score = int(float(score_str) * 100)
            except:
                logging.exception(comment)
                continue

        if args.limit is not None and abs(score) > args.limit:
            stats.positions_filtered_limit += 1
            continue

        if lichess:
            if args.eval_before_always:
                pass
            else:
                if node_index + 1 >= ply_count:
                    assert stats.positions_parsed > 0
                    stats.positions_parsed -= 1  # hack: keep stats straight
                    break

                epd = board.epd()  # the position after applying current move
                side_to_move = board.turn
                # Get the move following the EPD
                next_node = mainline[node_index + 1]
                current_move = next_node.move
                move_san = board.san(current_move)

            # Lichess evaluation is always from white's perspective in the PGN.
            # If side-to-move is black, negate the score to get black's perspective.
            if not side_to_move:
                score = -score

        move_from = current_move.from_square
        move_to = current_move.to_square

        # Get outcome from side-to-move perspective
        outcome = white_outcome if side_to_move else black_outcome
        logging.debug(f"{['black', 'white'][side_to_move]} score: {score}, result: {outcome} ({game.headers.get('Result', '*')})")
        logging.debug(f"+++ {epd}, {score}, {current_move.uci()}, {move_san}, {outcome}\n")
        epd_list.append((epd, score, current_move.uci(), move_san, move_from, move_to, outcome))

    return epd_list


def check_minimum_elo(args, game):
    min_elo = args.min_elo or 0

    elos = { color: int(game.headers.get(f'{color}Elo', '0')) for color in ['Black', 'White'] }
    for color, elo in elos.items():
        if elo and elo < min_elo:
            logging.debug(f'{elos}: {color} {elo} < {min_elo}')
            return False

        if args.elo_diff is not None:
            if abs(elos['White'] - elos['Black']) > args.elo_diff:
                logging.debug(f'{elos}: unbalanced game, ELO diff > {args.elo_diff}')
                return False

        if args.strong_black and elos['Black'] < elos['White']:
            logging.debug(f'{elos}: weaker black player')
            return False

    return True


def check_opening(allowed_openings, game):
    """Check if game opening is in the allowed list"""
    if allowed_openings is None:
        return True

    opening = game.headers.get('Opening', '').strip()
    if not opening:
        logging.debug(f'No opening header found, filtering out game')
        return False

    # Strip everything after colon to get base opening name
    # e.g., "Sicilian: Najdorf Variation" -> "Sicilian"
    base_opening = opening.split(':')[0].strip()

    # Normalize opening name for comparison (case-insensitive)
    opening_lower = opening.lower()
    base_opening_lower = base_opening.lower()

    # Check if either full opening or base opening matches
    match = opening_lower in allowed_openings or base_opening_lower in allowed_openings

    if match:
        logging.debug(f'Opening matched: "{opening}" (base: "{base_opening}")')
    # else:
    #     logging.debug(f'Opening not in allowed list: "{opening}" (base: "{base_opening}")')

    return match


def load_openings(filename):
    """Load opening names from a text file, one per line"""
    if filename is None:
        return None

    openings = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            opening = line.strip()
            if opening:
                # Store in lowercase for case-insensitive matching
                openings.add(opening.lower())

    logging.info(f"Loaded {len(openings)} opening(s) from {filename}")
    return openings


def main(args, stats):
    # Load allowed openings if specified
    allowed_openings = load_openings(args.openings)

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
        logging.info(f"Estimated games: {num_games}\n")

        if args.shuffle:
            logging.info("Shuffle mode: ENABLED (shuffling positions within each game)\n")

        # Open PGN file and process games sequentially
        with open(args.pgn_file, 'r', encoding='utf-8') as pgn_data:
            game_iter = iter(lambda: pgn.read_game(pgn_data), None)

            for game in tqdm(game_iter, total=num_games):
                if game is None:
                    continue

                stats.games += 1

                if not check_opening(allowed_openings, game):
                    stats.games_filtered_opening += 1
                    continue

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

                csr = sqlconn.executemany('''
                    INSERT OR IGNORE INTO position(epd, score, best_move_uci, best_move_san, best_move_from, best_move_to, outcome)
                    VALUES(?, ?, ?, ?, ?, ?, ?)''',
                    epd_list)

                assert args.unique or csr.rowcount == len(epd_list)
                stats.positions_inserted += csr.rowcount

                if stats.positions_inserted % args.commit_threshold == 0:
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
    parser.add_argument('-g', '--game-size', default=2300, type=int, help='estimated bytes per game in PGN file (default: 2300)')
    parser.add_argument('-o', '--output', help='sqlite3 output file (default: input filename with .db extension)')
    parser.add_argument('-v', '--debug', action='store_true')
    parser.add_argument('--color', choices=['black', 'white'], help="add only moves from the specified side to the database")
    parser.add_argument('--convert-comma', action='store_true', help='replace comma with decimal point in score')
    parser.add_argument('--elo-diff', type=int)
    parser.add_argument('--eval-before-always', action='store_true', help='Always apply eval score to position BEFORE making the move')
    parser.add_argument('--lichess', action='store_true', help='parse lichess eval format (https://database.lichess.org/)')
    parser.add_argument('--limit', type=int, help='absolute eval limit, in centipawns')
    parser.add_argument('--logfile', type=str, help='log file (default: output filename with .log extension)')
    parser.add_argument('--mate-score', type=int, default=15000, help='mate score in centipawns (default: 15000, 0 skips close-to-mate positions)')
    parser.add_argument('--min-elo', type=int, help='if specified, only include games with minimum player ELO')
    parser.add_argument('--openings', type=str, help='text file containing allowed opening names (one per line, case-insensitive)')
    parser.add_argument('--capture', action='store_false', dest='no_capture', help='include capturing moves (default: False)')
    parser.add_argument('--no-capture', action='store_true', default=True, help='exclude capturing moves (default: True)')
    parser.add_argument('--check', action='store_false', dest='no_check', help='include in-check position and checking moves (default: False)')
    parser.add_argument('--no-check', action='store_true', help='exclude in-check position and checking moves (default: True)')
    parser.add_argument('--shuffle', action='store_true', default=True, help='shuffle positions within each game before inserting into database (default: True)')
    parser.add_argument('--no-shuffle', action='store_false', dest='shuffle')
    parser.add_argument('--unique', action='store_true', dest='unique', default=True, help="store unique positions (use EPD as primary key, default: True)")
    parser.add_argument('--no-unique', action='store_false', dest='unique', help="allow multiple entries for a position (no EPD primary key)")
    parser.add_argument('--strong-black', action='store_true', help='discard games where black player is weaker than white')

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
