#! /usr/bin/env python3
import argparse

import chess
import numpy as np
import tensorflow as tf

# New tests from the rows provided with expected evals
tests = [
    'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq -',     # f7f5
    'rnbqkbnr/ppppp1pp/8/5p2/8/5N2/PPPPPPPP/RNBQKB1R w KQkq -',   # f3e5 (Ne5)
    'rnbqkbnr/ppppp1pp/8/4Np2/8/8/PPPPPPPP/RNBQKB1R b KQkq -',    # d7d6
    'rnbqkbnr/ppp1p1pp/3p4/4Np2/8/8/PPPPPPPP/RNBQKB1R w KQkq -',  # e2e4
    'rnbqkbnr/ppp1p1pp/3p4/4Np2/4P3/8/PPPP1PPP/RNBQKB1R b KQkq -', # d6e5 (dxe5)
    'rnbqkbnr/ppp1p1pp/8/4pp2/4P3/8/PPPP1PPP/RNBQKB1R w KQkq -',  # e4f5 (exf5)
    'rnbqkbnr/ppp1p1pp/8/4pP2/8/8/PPPP1PPP/RNBQKB1R b KQkq -',    # e5e4
    'rnbqkbnr/ppp1p1pp/8/5P2/4p3/8/PPPP1PPP/RNBQKB1R w KQkq -',   # d2d3 
    'rnbqkbnr/ppp1p1pp/8/5P2/4p3/3P4/PPP2PPP/RNBQKB1R b KQkq -',  # e4e3
    'rnbqkbnr/ppp1p1pp/8/5P2/8/3Pp3/PPP2PPP/RNBQKB1R w KQkq -',   # d1d2 (Qd2)
]

# Expected best moves and evaluations for each position
expected_moves = [
    'f7f5', 'f3e5', 'd7d6', 'e2e4', 'd6e5', 
    'e4f5', 'e5e4', 'd2d3', 'e4e3', 'd1d2'
]

expected_evals = [
    -0.14, 0.49, 0.20, -0.33, 3.58, 
    -4.23, 4.29, -1.71, 3.88, -1.45
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
