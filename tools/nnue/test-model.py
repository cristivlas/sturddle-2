#! /usr/bin/env python3
import argparse

import chess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from golds import TESTS as tests


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
            'ACCUMULATOR_SIZE': 2048,
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
        pawn_bucket = 0 if pawn_count <= 4 else min((pawn_count - 1) // 4, 3)
        wk_right = int(chess.square_file(board.king(chess.WHITE)) >= 4)
        bk_right = int(chess.square_file(board.king(chess.BLACK)) >= 4)
        king_bucket = wk_right * 2 + bk_right
        bucket = pawn_bucket * 4 + king_bucket
        print(f"[Bucket {bucket:2d} (pawn {pawn_bucket}, king {king_bucket}), {pawn_count:2d} pawns]")

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
