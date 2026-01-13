#! /usr/bin/env python3
import argparse

import chess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

tests = [
    # Bucket 3 (12-16 pawns): opening/early middlegame
    chess.STARTING_FEN,  # 16 pawns
    'r2r2k1/1pp2ppp/p2q1b2/3pN3/2PP4/PP1Q3P/5PP1/R3R1K1 b - - 0 22',  # 14 pawns
    'r4rk1/1ppnbppp/p2q4/3pNb2/3P4/PP5P/2PNBPP1/R2QK2R w KQ - 5 14',  # 14 pawns
    'r4rk1/ppp2ppp/5n2/2bPn3/4K3/2NP4/PPPBB1PP/R6R w - - 3 3',  # 13 pawns

    # Bucket 2 (8-11 pawns): middlegame
    '3r4/1pk2p1N/p1n1p3/4Pq2/2Pp1b1Q/8/PP4PP/R1K1R3 w - - 0 2',  # 11 pawns
    'rqr3k1/p4p1p/5Qp1/2b5/2N5/2Pn2NP/P2B1PP1/2R2RK1 w - - 0 24',  # 9 pawns
    '2r3k1/p5p1/4p3/1p1bP3/2pb2Q1/5N2/1q3P1P/3R1RK1 b - - 3 32',  # 8 pawns
    '1r1q1rk1/p3bBpp/2Q5/8/3Pb3/2n1BN2/P4PPP/R4RK1 b - - 0 18',  # 8 pawns

    # Bucket 1 (4-7 pawns): late middlegame/early endgame
    '8/pp2k3/8/8/8/8/3K1PP1/8 w - - 0 1',  # 4 pawns
    '3r2k1/pp3p2/8/8/8/5P2/PP4K1/3R4 w - - 0 1',  # 5 pawns
    'r3k3/pp6/8/3p4/3P4/8/PP2K3/R7 w q - 0 1',  # 6 pawns
    '2r2rk1/pp3p2/8/8/8/8/PP3PP1/2R2RK1 w - - 0 1',  # 7 pawns

    # Bucket 0 (0-3 pawns): endgame
    '8/8/4k3/4p3/4P3/4K3/8/8 w - - 0 1',  # 2 pawns
    '8/5k2/8/3p4/3P4/2K1P3/8/8 w - - 0 1',  # 3 pawns
    '4k3/8/8/8/8/8/4K3/4R3 w - - 0 1',  # 0 pawns - K+R vs K
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
    return tf.keras.models.load_model(path, custom_objects = {
            'ACCUMULATOR_SIZE': 1280,
            'ATTN_FAN_OUT': 32,
            'POOL_SIZE': 8,
            'combined_loss': None,
            'scaled_sparse_categorical_crossentropy': None,
            'top': None,
            'top_3': None,
            'top_5': None,
        })


def run_tests(args, model):
    evals = []
    for fen in tests:
        board = chess.Board(fen=fen)
        assert board.is_valid(), f"Invalid position: {fen}"

        pawn_count = chess.popcount(board.pawns)
        bucket = min(pawn_count // 4, 3)
        print(f"[Bucket {bucket}, {pawn_count:2d} pawns]")

        encoding = encode(board)
        encoding = encoding.T.reshape((1, 13))
        eval = model.predict(encoding)
        print(board.epd(), *eval)
        res = eval[0][0][0] if len(eval) > 1 else eval[0][0]
        evals.append(float(res) * 100)
    print(evals)


def main(args):
    model = load_model(args)
    model.summary()
    run_tests(args, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1)
    main(parser.parse_args())
