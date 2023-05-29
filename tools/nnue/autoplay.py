#!/usr/bin/env python3

import argparse
import logging
import os
from contextlib import closing

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import chess.engine
import chess.pgn
import numpy as np

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # silence off annoying warnings

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from keras.models import load_model
from modeltojson import serialize_weights


def encode(board):
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
    openings = []
    with open(openings_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            openings.append(game)
    return openings

def main(args):
    with closing(chess.engine.SimpleEngine.popen_uci(args.engine1)) as engine1, \
         closing(chess.engine.SimpleEngine.popen_uci(args.engine2)) as engine2:

        config = { 'Threads': args.threads, 'Hash': args.hash }
        engine1.configure(config)
        engine2.configure(config)

        # Load openings
        openings = []
        if args.openings:
            openings = parse_openings(args.openings)
        else:
            logging.warning('No openings file specified. Playing without openings.')

        num_openings = len(openings)
        if num_openings == 0:
            logging.warning('No valid openings found. Playing without openings.')

        # Play the games
        for game_num in range(args.num_games):
            # Create the chess board
            board = chess.Board()

            # Each opening is played twice, once for each engine as white
            opening_game = openings[game_num // 2 % num_openings] if num_openings > 0 else None

            # Apply opening moves
            if opening_game:
                for move in opening_game.mainline_moves():
                    move_count = len(board.move_stack)
                    if move_count >= args.max_openings:
                        break
                    logging.info(f'{move_count+1}: {move}')
                    board.push(move)

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
                result = engine.play(board, chess.engine.Limit(args.time_limit))
                board.push(result.move)

            # Log the game result
            move_count = len(board.move_stack)
            logging.info(f'Game {game_num+1}: {board.result()}, {move_count} moves')
            logging.info(f'Winner: {get_winner(board.result(), names)}')

            # Call on_end_game callback
            on_end_game(args, board, engines, engine1, engine2)

            # Reset the board for the next game
            board.reset()

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
    # Custom engine configurations before each game
    config = {}
    if os.path.exists(args.json_path):
        config['NNUEModel'] = args.json_path
        engine1.configure(config)

        if args.engine1 == args.engine2: # self-play?
            engine2.configure(config)

def on_end_game(args, board, engines, engine1, engine2):
    @tf.function
    def _clipped_mae(y_true, y_pred):
        y_true = tf.clip_by_value(y_true, -args.clip, args.clip)
        return tf.keras.losses.mean_absolute_error(y_true, y_pred)

    if args.model:
        model = load_model(args.model, custom_objects={'_clipped_mae': _clipped_mae })
        reward = 0
        result = board.result()
        if result == '1-0':
            reward = 1 if engines[0] == engine1 else -1
        elif result == '0-1':
            reward = -1 if engines[0] == engine1 else 1
        logging.info(f'Reward: {reward}')

        replay = chess.Board()
        positions = [encode(replay)]
        for move in board.move_stack:
            replay.push(move)
            positions.append(encode(replay))

        X_train = np.array(positions[:-1])  # all positions except the last one
        next_positions = np.array(positions[1:])  # all positions except the first one

        X_predict = model.predict(X_train, verbose=False)
        next_preds = model.predict(next_positions, verbose=False)

        # Discount factor for future rewards
        gamma = max(0, min(args.gamma, 1))
        discounted_rewards = np.zeros_like(next_preds)
        running_add = 0
        for t in reversed(range(0, len(next_preds))):
            running_add = running_add * gamma + next_preds[t]
            discounted_rewards[t] = running_add

        # Use these differences as targets for learning, modulated by the game result
        y_train = np.squeeze(X_predict) + reward * discounted_rewards

        # Train and save the model
        model.fit(X_train, y_train, epochs=args.epochs, verbose=True, batch_size=args.batch_size)

        model.save(args.model)
        serialize_weights(model, args.json_path)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Autoplay with Reinforcement Learning')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-c', '--clip', type=int, default=5)
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train the model after each game')
    parser.add_argument('-e1', '--engine1', type=str, default='./main.py', help='Path to the first engine')
    parser.add_argument('-e2', '--engine2', type=str, default='./main.py', help='Path to the second engine')
    parser.add_argument('-g', '--gamma', type=float, default=0.9, help='Discount for future rewards')
    parser.add_argument('--hash', type=int, default=512, help='Engine hash table size in MiB')
    parser.add_argument('-t', '--time-limit', type=float, default=0.1, help='Time limit for each move (in seconds)')
    parser.add_argument('--threads', type=int, default=4, help='Engine SMP threads')
    parser.add_argument('--openings', type=str, help='Path to the PGN file with opening moves')
    parser.add_argument('--max-openings', type=int, default=2, help='Depth of opening moves to apply')
    parser.add_argument('-m', '--model', help='Path to NNUE model')
    parser.add_argument('-n', '--num-games', type=int, default=1, help='Number of games to play')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--logfile', type=str, help='Path to the logfile')
    args = parser.parse_args()

    # Configure logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.logfile:
        logging.basicConfig(filename=args.logfile, level=log_level, format=log_format)
    else:
        logging.basicConfig(level=log_level, format=log_format)

    if args.model:
        args.json_path = os.path.basename(args.model) + '.json'

    # Play the games
    try:
        main(args)
    except KeyboardInterrupt:
        pass
