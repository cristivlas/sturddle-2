#! /usr/bin/env python3
import argparse

import chess
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tests = [
    'rnbqkbnr/ppp5/3pp3/5p2/2PP2p1/2NBP1Pp/PP1N1P1P/R1BQK2R b KQkq -',
    '5b1r/6pp/5p1k/5q2/1P5P/P5Q1/5PPK/8 w - -',
    'r4rk1/pb3ppp/1p6/3pP3/6q1/P1P2Nn1/1P1N2B1/R1BQR1K1 w - -',
    'r1b1k2r/pp2qppp/8/3p4/3n4/5N2/PP2BPPP/R2QK2R w KQkq -',
    '8/1R4p1/p1p4p/2k4n/P1P3B1/1P3KP1/1r3P2/8 b - -',
    '8/5kp1/4p2p/5p2/2r5/1N3P2/5PKP/8 b - -',
    '5bk1/1p3p2/6pp/pPR1p3/P7/1P4P1/3r3P/5BK1 w - -',
    'r4rk1/ppq2p1p/4pp2/1P1p4/P7/3N1P2/6Pb/R2Q1R1K w - -',
    'rnbq1rk1/pp3pbp/3ppnp1/2p3N1/2BPP3/2N2Q2/PPP2PPP/R1B1K2R w KQ -',
    'r7/8/5kp1/2p2p1p/1pRbbP1P/1P2p1PK/P3B3/5R2 b - -'
]

# Expected best moves and evaluations for each position
expected_moves = [
    'b7b6', 'g3e3', 'b2b4', 'd1d4', 'h5f6',
    'g7g5', 'c5c7', 'd3f2', 'e4e5', 'a8a2'
]

expected_evals = [
    0.09, -5.97, 5.62, 4.78, 0.44,
    -4.79, -0.20, 1.82, -0.52, -5.77
]

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
    array = np.asarray([bitboards], dtype=np.uint64).ravel()
    array = np.append(array, np.uint64(board.turn))

    return array


def load_model(args):
    path = args.input[0]
    return tf.keras.models.load_model(path, custom_objects={'_clipped_mae': None, 'clipped_loss': None})


def get_top_moves(from_probs, to_probs, board, num_moves=5):
    """Get the top predicted moves based on from/to probabilities"""
    # Convert probabilities to numpy arrays for easier manipulation
    from_probs = from_probs[0]  # Remove batch dimension
    to_probs = to_probs[0]  # Remove batch dimension

    # Calculate combined probabilities for all legal moves
    moves = []
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        # Combine probabilities (multiply from_prob and to_prob)
        combined_prob = from_probs[from_square] * to_probs[to_square]
        moves.append((move, combined_prob))

    # Sort moves by probability (highest first)
    moves.sort(key=lambda x: x[1], reverse=True)

    # Return top N moves
    return moves[:num_moves]


def run_tests(args, model):
    eval_errors = []
    move_accuracy = 0
    top_3_accuracy = 0

    for i, (fen, expected_move, expected_eval) in enumerate(zip(tests, expected_moves, expected_evals)):
        board = chess.Board(fen=fen)
        encoding = encode(board)
        encoding = encoding.T.reshape((1, 13))

        # Model prediction - handle both with and without move prediction
        predictions = model.predict(encoding, verbose=0)

        # Check if model outputs move predictions
        if isinstance(predictions, list):
            eval_score, from_probs, to_probs = predictions
            has_move_prediction = True
        else:
            eval_score = predictions
            has_move_prediction = False

        # Get evaluation score
        eval_value = eval_score[0][0]
        eval_error = abs(eval_value - expected_eval)
        eval_errors.append(eval_error)

        # Print position and evaluation
        print(f"\nPosition {i+1}: {fen}")
        print(f"Model eval: {eval_value:.2f}, Expected eval: {expected_eval:.2f}, Error: {eval_error:.2f}")

        # Convert expected move to chess.Move object
        expected_move_obj = chess.Move.from_uci(expected_move)
        expected_move_san = board.san(expected_move_obj) if expected_move_obj in board.legal_moves else expected_move

        print(f"Expected move: {expected_move} ({expected_move_san})")

        # If model has move prediction capability, show top moves
        if has_move_prediction:
            top_moves = get_top_moves(from_probs, to_probs, board)

            print("Top predicted moves:")
            for j, (move, prob) in enumerate(top_moves, 1):
                move_uci = move.uci()
                is_expected = move_uci == expected_move
                print(f"{j}. {board.san(move)} ({move_uci}) (probability: {prob:.4f}){' ✓' if is_expected else ''}")

                # Update accuracy metrics
                if j == 1 and is_expected:
                    move_accuracy += 1
                if j <= 3 and is_expected:
                    top_3_accuracy += 1

    # Print summary
    print("\n--- Summary ---")
    avg_eval_error = sum(eval_errors) / len(eval_errors)
    print(f"Average evaluation error: {avg_eval_error:.2f}")

    if has_move_prediction:
        print(f"Top-1 move accuracy: {move_accuracy}/{len(tests)} ({move_accuracy/len(tests)*100:.2f}%)")
        print(f"Top-3 move accuracy: {top_3_accuracy}/{len(tests)} ({top_3_accuracy/len(tests)*100:.2f}%)")


def main(args):
    model = load_model(args)
    if args.verbose:
        model.summary()
    run_tests(args, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1, help='Path to the model file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output including model summary')
    main(parser.parse_args())
