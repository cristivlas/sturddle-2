#!/usr/bin/env python3

import argparse
import logging
import os
import shutil
import socket
import sys
import tempfile
from contextlib import closing
from datetime import datetime

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR) # silence off annoying warnings

import chess.engine
import chess.pgn
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from keras.models import load_model
from modeltojson import serialize_weights

# use EcoAPI for logging openings by name
chessutils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sqlite'))
sys.path.append(chessutils_path)
from chessutils.eco import EcoAPI


def encode(board):
    # One-hot encode the chess board position, return numpy array expected by neural net model.
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
    f = np.unpackbits(np.asarray(bitboards, dtype='>u8').reshape(12,).view(np.uint8))
    return np.append(f, [board.turn])

def parse_openings(openings_file):
    # Parse PGN file and return list of openings.
    openings = []
    with open(openings_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            openings.append(game)
    return openings

def append_pgn(board, file_path, game_num, names):
    # Create a new game from the board
    game = chess.pgn.Game().from_board(board)
    game.headers['White'] = names[0]
    game.headers['Black'] = names[1]
    game.headers['Date'] = datetime.now().strftime('%Y.%m.%d')
    game.headers['Event'] = 'AutoPlay'
    game.headers['Site'] = socket.gethostname()
    game.headers['Round'] = game_num

    # Open the file in append mode
    with open(file_path, 'a') as pgn_file:
        # Write the game to the file
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)

def main(args):
    eco = EcoAPI(args.eco_path)
    logging.info(f'Engine1: {os.path.abspath(args.engine1)}')
    logging.info(f'Engine2: {os.path.abspath(args.engine2)}')
    logging.info(f'Time Limit: {args.time_limit}')

    model = load_model(args.model, custom_objects={'_clipped_mae': None})
    logging.info(f'Writing: {os.path.abspath(args.json_path)}')
    serialize_weights(model, args.json_path)

    with closing(chess.engine.SimpleEngine.popen_uci(args.engine1)) as engine1, \
         closing(chess.engine.SimpleEngine.popen_uci(args.engine2)) as engine2:

        config = { 'Threads': args.threads, 'Hash': args.hash, 'OwnBook': False }
        engine1.configure(config)
        engine2.configure(config)

        # Load openings
        openings = []
        if args.openings:
            openings = parse_openings(args.openings)
        else:
            logging.warning('No openings file specified. Playing without openings.')

        num_openings = len(openings)
        logging.info(f'Using {num_openings} openings from: {args.openings}')

        opening_name = ''

        # Play the games
        for game_num in range(args.num_games):
            # Create the chess board
            board = chess.Board()

            if num_openings:
                # Each opening is played twice, once for each engine as white
                opening_idx = (game_num // 2 + args.opening_offset) % num_openings
                opening_game = openings[opening_idx]

                # Apply opening moves
                for move in opening_game.mainline_moves():
                    move_count = len(board.move_stack)
                    if move_count >= args.max_openings:
                        break
                    board.push(move)
                    row = eco.lookup(board)
                    if row:
                        opening_name = row['name']


                logging.info(f'Opening {opening_idx}: {[move.uci() for move in board.move_stack]}')

            # Determine which engine plays as white and black
            if game_num % 2 == 0:
                engines = engine1, engine2
                names = 'engine1', 'engine2'
            else:
                engines = engine2, engine1
                names = 'engine2', 'engine1'

            # Call on_begin_game callback
            on_begin_game(args, board, engine1, engine2)

            # Main game loop
            while not board.is_game_over():
                engine = engines[not board.turn]
                try:
                    result = engine.play(board, chess.engine.Limit(args.time_limit))
                except KeyboardInterrupt:
                    try:
                        engine1.quit()
                    except:
                        pass  # engine1 is already terminated
                    try:
                        engine2.quit()
                    except:
                        pass  # engine2 is already terminated
                    print('Terminated.')
                    os._exit(0)

                board.push(result.move)
                row = eco.lookup(board)
                if row:
                    opening_name = row['name']

            # Log the game result
            move_count = len(board.move_stack)
            logging.info(f'Game {game_num+1}/{args.num_games}: {opening_name} [{board.result()}], {move_count} moves')
            logging.info(f'Winner: {get_winner(board.result(), names)}')

            if args.pgn_path:
                append_pgn(board, args.pgn_path, game_num, names)

            # Call on_end_game callback for reinforcement learning
            on_end_game(args, board, engines, engine1, engine2)

def get_winner(result, engines):
    if result == '1-0':
        return f'{engines[0]} (playing as White)'
    elif result == '0-1':
        return f'{engines[1]} (playing as Black)'
    elif result == '1/2-1/2':
        return 'Draw'
    else:
        return 'Unknown'

def on_begin_game(args, board, engine1, engine2):
    # Force engine(s) to reload the model.
    if os.path.exists(args.json_path):
        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy2(args.json_path, temp_file.name)
            engine1.configure({'NNUEModel': temp_file.name})

            if args.engine1 == args.engine2:
                engine2.configure({'NNUEModel': temp_file.name})

def on_end_game(args, board, engines, engine1, engine2):
    # Time difference reinforcement learning
    @tf.function
    def _clipped_mae(y_true, y_pred):
        y_true = tf.clip_by_value(y_true, -args.clip, args.clip)
        return tf.keras.losses.mean_absolute_error(y_true, y_pred)

    if args.model:
        # Discount factor: the rewards are discounted by multiplying
        # each reward by gamma raised to the power of the time step.
        gamma = max(0, min(1, args.gamma))

        model = load_model(args.model, custom_objects={'_clipped_mae': _clipped_mae })
        optimizer=tf.keras.optimizers.Adam(
            amsgrad=True,
            beta_1=0.99,
            beta_2=0.995,
            learning_rate=args.learn_rate
        )
        model.compile(loss=_clipped_mae, optimizer=optimizer)

        reward = 0
        result = board.result()
        if result == '1-0':
            reward = 1
        elif result == '0-1':
            reward = -1
        logging.info(f'Reward: {reward}')

        replay = chess.Board()
        white_positions = [encode(replay)]
        black_positions = []
        for move in board.move_stack:
            replay.push(move)
            if replay.turn:
                white_positions.append(encode(replay))
            else:
                black_positions.append(encode(replay))

        # initialize empty lists to store training data for both players
        X_train_all = []
        y_train_all = []

        for positions in (white_positions, black_positions):
            X_train = np.array(positions[:-1])
            next_positions = np.array(positions[1:])

            X_predict = model.predict(X_train, verbose=False)
            next_preds = model.predict(next_positions, verbose=False)
            score_diffs = np.squeeze(next_preds - X_predict)

            # Apply discount factor
            discount_factors = np.array([gamma ** i for i in range(len(score_diffs))][::-1])
            discounted_score_diffs = score_diffs * discount_factors

            y_train = np.squeeze(X_predict) + reward * discounted_score_diffs

            X_train_all.append(X_train)
            y_train_all.append(y_train)

            reward = -reward

        # concatenate all training data
        X_train_all = np.concatenate(X_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)

        # Train the model on all data
        model.fit(X_train_all, y_train_all, epochs=args.epochs, verbose=True, batch_size=args.batch_size)

        logging.info(f'Updating: {os.path.abspath(args.model)}')
        model.save(args.model)
        logging.info(f'Writing: {os.path.abspath(args.json_path)}')
        serialize_weights(model, args.json_path)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Autoplay with Reinforcement Learning')
    parser.add_argument('--batch-size', '-b', type=int, default=4)
    parser.add_argument('--clip', '-c', type=int, default=5)
    parser.add_argument('--eco-path', help='Optional path to ECO data')
    parser.add_argument('--epochs', '-e', type=int, default=25, help='Number of epochs to train the model')
    parser.add_argument('--engine1', default='./main.py', help='Path to the first engine')
    parser.add_argument('--engine2', default='./main.py', help='Path to the second engine')
    parser.add_argument('--hash', type=int, default=512, help='Engine hash table size in MiB')
    parser.add_argument('--gamma', '-g', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--learn-rate', '-r', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--logfile', default='log.txt', help='Path to the logfile')
    parser.add_argument('--openings', help='Path to the PGN file with opening moves')
    parser.add_argument('--opening-offset', type=int, default=0, help='Offset for picking opening moves')
    parser.add_argument('--pgn-path', default='out.pgn', help='Path to file where to save games')
    parser.add_argument('--max-openings', type=int, default=2, help='Depth of opening moves to apply')
    parser.add_argument('--model', '-m', help='Path to NNUE model')
    parser.add_argument('--num-games', '-n', type=int, default=1, help='Number of games to play')
    parser.add_argument('--time-limit', type=float, default=0.1, help='Time limit for each move (in seconds)')
    parser.add_argument('--threads', type=int, default=1, help='Engine SMP threads')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    if args.model:
        args.json_path = os.path.abspath(os.path.basename(os.path.abspath(args.model)) + '.json')

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.logfile),
            logging.StreamHandler()
        ])

    # Play the games
    try:
        main(args)
    except KeyboardInterrupt:
        pass
