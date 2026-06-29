#! /usr/bin/env python3
"""
Canonical NNUE test positions and golden evals.

TESTS is the single source of truth for the FEN list shared by test-model.py
and test/unit_test.py, so the two can never drift out of order again.

Golds are stored in golds.json keyed by FEN (order-independent). Regenerate
after retraining:

    python tools/nnue/golds.py models\\KP44
"""
import json
import os

import chess

GOLDS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golds.json')

# Pawn dimension: 0 -> 0-4, 1 -> 5-8, 2 -> 9-12, 3 -> 13-16 pawns.
# King-file dimension (composed x4): both-left, W-left/B-right, W-right/B-left, both-right.
TESTS = [
    # Pawn bucket 3 (13-16 pawns): opening/early middlegame
    chess.STARTING_FEN,  # 16 pawns
    'r2r2k1/1pp2ppp/p2q1b2/3pN3/2PP4/PP1Q3P/5PP1/R3R1K1 b - - 0 22',  # 14 pawns
    'r4rk1/1ppnbppp/p2q4/3pNb2/3P4/PP5P/2PNBPP1/R2QK2R w KQ - 5 14',  # 14 pawns
    'r4rk1/ppp2ppp/5n2/2bPn3/4K3/2NP4/PPPBB1PP/R6R w - - 3 3',  # 13 pawns

    # Pawn bucket 2 (9-12 pawns): middlegame
    '3r4/1pk2p1N/p1n1p3/4Pq2/2Pp1b1Q/8/PP4PP/R1K1R3 w - - 0 2',  # 11 pawns
    'rqr3k1/p4p1p/5Qp1/2b5/2N5/2Pn2NP/P2B1PP1/2R2RK1 w - - 0 24',  # 9 pawns

    # Pawn bucket 1 (5-8 pawns): late middlegame/early endgame
    '2r3k1/p5p1/4p3/1p1bP3/2pb2Q1/5N2/1q3P1P/3R1RK1 b - - 3 32',  # 8 pawns
    '1r1q1rk1/p3bBpp/2Q5/8/3Pb3/2n1BN2/P4PPP/R4RK1 b - - 0 18',  # 8 pawns
    '3r2k1/pp3p2/8/8/8/5P2/PP4K1/3R4 w - - 0 1',  # 5 pawns
    'r3k3/pp6/8/3p4/3P4/8/PP2K3/R7 w q - 0 1',  # 6 pawns
    '2r2rk1/pp3p2/8/8/8/8/PP3PP1/2R2RK1 w - - 0 1',  # 7 pawns

    # Pawn bucket 0 (0-4 pawns): endgame
    '8/pp2k3/8/8/8/8/3K1PP1/8 w - - 0 1',  # 4 pawns
    '8/8/4k3/4p3/4P3/4K3/8/8 w - - 0 1',  # 2 pawns
    '8/5k2/8/3p4/3P4/2K1P3/8/8 w - - 0 1',  # 3 pawns
    '4k3/8/8/8/8/8/4K3/4R3 w - - 0 1',  # 0 pawns - K+R vs K

    # King-file coverage (all in pawn bucket 0, so composed bucket == king bucket):
    '1k6/3p4/8/8/8/8/3P4/2K5 w - - 0 1',  # king bucket 0: both left (WK c1, BK b8)
    '6k1/3p4/8/8/8/8/3P4/2K5 w - - 0 1',  # king bucket 1: W left, B right (WK c1, BK g8)
    '1k6/3p4/8/8/8/8/3P4/6K1 w - - 0 1',  # king bucket 2: W right, B left (WK g1, BK b8)
    '6k1/3p4/8/8/8/8/3P4/6K1 w - - 0 1',  # king bucket 3: both right (WK g1, BK g8)
]


def load_golds():
    """Return {fen: eval} dict, or None if the gold file is missing."""
    if not os.path.exists(GOLDS_JSON):
        return None
    with open(GOLDS_JSON) as f:
        return json.load(f)


def _generate(model_path):
    """Run the saved Keras model over TESTS and write golds.json keyed by FEN."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import numpy as np
    import tensorflow as tf

    def encode(board):
        mask_black = board.occupied_co[chess.BLACK]
        mask_white = board.occupied_co[chess.WHITE]
        bitboards = [[pcs & mask_black, pcs & mask_white] for pcs in (
            board.kings, board.pawns, board.knights, board.bishops, board.rooks, board.queens)]
        array = np.asarray([bitboards], dtype=np.uint64).ravel()
        return np.append(array, np.uint64(board.turn))

    model = tf.keras.models.load_model(model_path, custom_objects={
        'ACCUMULATOR_SIZE': 2048,
        'POOL_SIZE': 8,
        'combined_loss': None,
        'scaled_sparse_categorical_crossentropy': None,
        'top': None,
        'top_3': None,
        'top_5': None,
    })

    print('EXPORT ORDER:', [(l.name, [w.shape for w in l.get_weights()]) for l in model.layers if l.get_weights()])

    golds = {}
    for fen in TESTS:
        board = chess.Board(fen=fen)
        assert board.is_valid(), f'Invalid position: {fen}'
        encoding = encode(board).T.reshape((1, 13))
        out = model.predict(encoding, verbose=0)
        res = out[0][0][0] if len(out) > 1 else out[0][0]
        golds[fen] = float(res) * 100

    with open(GOLDS_JSON, 'w') as f:
        json.dump(golds, f, indent=2)
    print(f'Wrote {len(golds)} golds to {GOLDS_JSON}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <model_dir>')
        sys.exit(1)
    _generate(sys.argv[1])
