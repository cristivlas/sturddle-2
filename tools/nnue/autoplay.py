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
            opening_game = openings[(game_num // 2 + args.opening_offset) % num_openings] if num_openings > 0 else None

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

        for positions in (white_positions, black_positions):
            X_train = np.array(positions[:-1])
            next_positions = np.array(positions[1:])

            X_predict = model.predict(X_train, verbose=False)
            next_preds = model.predict(next_positions, verbose=False)
            score_diffs = np.squeeze(next_preds - X_predict)
            # Use these differences as targets for learning, modulated by the game result
            y_train = np.squeeze(X_predict) + reward * score_diffs

            # Train the model
            model.fit(X_train, y_train, epochs=args.epochs, verbose=True, batch_size=args.batch_size)

            reward = -reward

        model.save(args.model)
        serialize_weights(model, args.json_path)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Autoplay with Reinforcement Learning')
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--clip', '-c', type=int, default=5)
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train the model for each side')
    parser.add_argument('--engine1', default='./main.py', help='Path to the first engine')
    parser.add_argument('--engine2', default='./main.py', help='Path to the second engine')
    parser.add_argument('--hash', type=int, default=512, help='Engine hash table size in MiB')
    parser.add_argument('--logfile', default='log.txt', help='Path to the logfile')
    parser.add_argument('--openings', help='Path to the PGN file with opening moves')
    parser.add_argument('--opening-offset', type=int, default=0, help='Offset for picking opening moves')
    parser.add_argument('--max-openings', type=int, default=2, help='Depth of opening moves to apply')
    parser.add_argument('--model', '-m', help='Path to NNUE model')
    parser.add_argument('--num-games', '-n', type=int, default=1, help='Number of games to play')
    parser.add_argument('--time-limit', type=float, default=0.1, help='Time limit for each move (in seconds)')
    parser.add_argument('--threads', type=int, default=4, help='Engine SMP threads')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    if args.model:
        args.json_path = os.path.basename(args.model) + '.json'

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
