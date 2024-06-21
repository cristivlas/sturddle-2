#! /usr/bin/env python3
import argparse
import datetime
import glob
import logging
import os
import uuid

import chess
import chess.engine
import chess.pgn


def get_engine(args):
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    params = {'Threads': args.threads, 'Hash': args.hash}
    if args.db:
        params['DB'] = args.db
    engine.configure(params)
    return engine


def play_from_position(args, epd, engines):
    # Log the starting position
    logging.info(f'{args.count}/{args.total} playing from: "{epd}"')
    args.count += 1

    board = chess.Board(fen=epd)
    assert board.is_valid()

    # An arbitrary object that identifies the game. Will automatically inform the engine
    # if the object is not equal to the previous game (e.g., ucinewgame, new).
    game = chess.pgn.Game()
    game.headers['FEN'] = epd

    if args.depth:
        limit = chess.engine.Limit(depth=args.depth)
    else:
        limit = chess.engine.Limit(time=args.time)
    try:
        # Play the game until it is over
        while not board.is_game_over():
            assert board.is_valid()
            result = engines[board.turn].play(board, game=game, limit=limit, ponder=args.ponder)
            if result:
                board.push(result.move)

    except Exception as e:
        # Log the exception message
        logging.error(f'Error playing game from position "{epd}": {e}')
        quit()

    # Initialize the game in PGN format
    game = chess.pgn.Game()
    game.headers['Event'] = str(uuid.uuid4())
    game.headers['Date'] = datetime.datetime.now().strftime('%Y.%m.%d')
    game.headers['White'] = args.engine
    game.headers['Black'] = args.engine
    game.headers['FEN'] = epd
    game.headers['Result'] = board.result()
    node = game

    for move in board.move_stack:
        node = node.add_variation(move)

    # Save the game in PGN format to the output file
    with open(args.output, 'a') as f:
        f.write(game.accept(chess.pgn.StringExporter()) + '\n\n')


def cleanup(engines):
    for i in range(0, 2):
        try:
            if engines[i]:
                engines[i].quit()
        except Exception as e:
            logging.error(f"engine[{i}].quit(): {e}")
        engines[i] = None


def generate_games(args, input_paths):
    # Initialize an empty set to store the EPDS
    epds = set()

    # Iterate over the input paths
    for input_path in input_paths:
        # Open the file for reading
        with open(input_path, 'r') as f:
            # Read the file line by line
            for line in f:
                # Strip leading and trailing white spaces from the line
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Split the line into fields using the EPD delimiter (space)
                fields = line.split(' ')
                # Extract the EPD from the fields
                epd = ' '.join(fields[:4])
                # Try to parse the EPD using the python-chess library
                try:
                    board = chess.Board(epd)
                except Exception as e:
                    logging.error(f'Invalid EPD "{epd}" in file {input_path}, line "{line}": {e}')
                    raise
                else:
                    # Add the EPD to the set if it is valid
                    epds.add(epd)
    args.total = len(epds)

    for epd in epds:
        engines = [None, None]
        try:
            engines = [get_engine(args), get_engine(args)]
            play_from_position(args, epd, engines)
            cleanup(engines)
        except KeyboardInterrupt:
            logging.info("Interrupted.")
            cleanup(engines)
            return

def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', help='path to input file(s) or directory(ies)')
    parser.add_argument('-d', '--depth', type=int, help='engine depth limit')
    parser.add_argument('-e', '--engine', default='stockfish')
    parser.add_argument('-o', '--output', default='out.pgn', help='path to output PGN file')
    parser.add_argument('-p', '--ponder', action='store_true')
    parser.add_argument('-t', '--time', type=float, default=0.1, help='engine time limit')
    parser.add_argument('--hash', type=int, default=512, help='engine hashtable size in MB')
    parser.add_argument('--threads', type=int, default=1, help='engine threads')

    # Collect eval scores, requires sturddle-2 engine with DATAGEN enabled at compile-time
    parser.add_argument('--db', help='database file name for collection eval scores')

    args = parser.parse_args()
    args.count = 1
    logging.basicConfig(level=logging.INFO)

    # Collect the paths to the input files
    input_paths = []
    for input_arg in args.input:
        if os.path.isfile(input_arg):
            input_paths.append(input_arg)
        elif os.path.isdir(input_arg):
            input_paths.extend(glob.glob(os.path.join(input_arg, '*.epd')))
        else:
            logging.warning(f'Invalid input: {input_arg}')

    # Call the generate_games function with the input paths and output path
    generate_games(args, input_paths)

if __name__ == '__main__':
    main()
