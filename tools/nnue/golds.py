#! /usr/bin/env python3
import json
import os

import chess

GOLDS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golds.json")

# King bucket per view: own king square, files e-h mirrored onto a-d (nnue.h KING_BUCKETS).
# Comments: king square (bucket, m = mirrored, q = opponent has a queen) for each view.

# Left-right mirror pair: golds must match
MIRROR_PAIR = (
    "1k6/ppp5/8/8/8/8/PPP5/1K6 w - - 0 1",  # WK b1 (1), BK b8 (1)
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",  # WK g1 (1m), BK g8 (1m)
)

TESTS = [
    # Opening/early middlegame (13-16 pawns)
    chess.STARTING_FEN,  # 16 pawns
    "r2r2k1/1pp2ppp/p2q1b2/3pN3/2PP4/PP1Q3P/5PP1/R3R1K1 b - - 0 22",  # 14 pawns
    "r4rk1/1ppnbppp/p2q4/3pNb2/3P4/PP5P/2PNBPP1/R2QK2R w KQ - 5 14",  # 14 pawns
    "r4rk1/ppp2ppp/5n2/2bPn3/4K3/2NP4/PPPBB1PP/R6R w - - 3 3",  # 13 pawns
    # Middlegame (9-12 pawns)
    "3r4/1pk2p1N/p1n1p3/4Pq2/2Pp1b1Q/8/PP4PP/R1K1R3 w - - 0 2",  # 11 pawns
    "rqr3k1/p4p1p/5Qp1/2b5/2N5/2Pn2NP/P2B1PP1/2R2RK1 w - - 0 24",  # 9 pawns
    # Late middlegame/early endgame (5-8 pawns)
    "2r3k1/p5p1/4p3/1p1bP3/2pb2Q1/5N2/1q3P1P/3R1RK1 b - - 3 32",  # 8 pawns
    "1r1q1rk1/p3bBpp/2Q5/8/3Pb3/2n1BN2/P4PPP/R4RK1 b - - 0 18",  # 8 pawns
    "3r2k1/pp3p2/8/8/8/5P2/PP4K1/3R4 w - - 0 1",  # 5 pawns
    "r3k3/pp6/8/3p4/3P4/8/PP2K3/R7 w q - 0 1",  # 6 pawns
    "2r2rk1/pp3p2/8/8/8/8/PP3PP1/2R2RK1 w - - 0 1",  # 7 pawns
    # Endgame (0-4 pawns)
    "8/pp2k3/8/8/8/8/3K1PP1/8 w - - 0 1",  # 4 pawns
    "8/8/4k3/4p3/4P3/4K3/8/8 w - - 0 1",  # 2 pawns
    "8/5k2/8/3p4/3P4/2K1P3/8/8 w - - 0 1",  # 3 pawns
    "4k3/8/8/8/8/8/4K3/4R3 w - - 0 1",  # 0 pawns - K+R vs K
    # Mirroring per view
    "1k6/3p4/8/8/8/8/3P4/2K5 w - - 0 1",  # WK c1 (2), BK b8 (1)
    "6k1/3p4/8/8/8/8/3P4/2K5 w - - 0 1",  # WK c1 (2), BK g8 (1m)
    "1k6/3p4/8/8/8/8/3P4/6K1 w - - 0 1",  # WK g1 (1m), BK b8 (1)
    "6k1/3p4/8/8/8/8/3P4/6K1 w - - 0 1",  # WK g1 (1m), BK g8 (1m)
    *MIRROR_PAIR,
    # Advanced kings
    "8/3p4/8/5k2/2K5/8/3P4/8 w - - 0 1",  # WK c4 (10), BK f5 (10m)
    "8/p7/5K2/8/8/1k6/7P/8 w - - 0 1",  # WK f6 (13m), BK b3 (12)
    "8/6K1/8/8/3p4/8/k7/8 w - - 0 1",  # WK g7 (14m), BK a2 (14)
    "7K/8/8/8/8/8/3p4/7k w - - 0 1",  # WK h8 (14m), BK h1 (14m)
    # Opponent queen per view
    "1k6/3pq3/8/8/8/8/3P4/2K5 w - - 0 1",  # WK c1 (2q), BK b8 (1)
    "1k6/3p4/8/8/8/8/3PQ3/2K5 w - - 0 1",  # WK c1 (2), BK b8 (1q)
    "6k1/3pq3/8/8/8/8/3P4/6K1 w - - 0 1",  # WK g1 (1mq), BK g8 (1m)
    # One queen capture apart: Qxe7
    "1k6/3pq3/8/8/8/8/3PQ3/2K5 w - - 0 1",  # WK c1 (2q), BK b8 (1q)
    "1k6/3pQ3/8/8/8/8/3P4/2K5 b - - 0 1",  # WK c1 (2), BK b8 (1q)
]


def load_golds():
    """Return {fen: eval} dict, or None if the gold file is missing."""
    if not os.path.exists(GOLDS_JSON):
        return None
    with open(GOLDS_JSON) as f:
        return json.load(f)


def _generate(bin_path):
    """Run a flat weights.bin through the torch NNUE over TESTS, write golds.json."""
    import numpy as np
    import torch
    import train_torch as tt

    def encode(board):
        mask_black = board.occupied_co[chess.BLACK]
        mask_white = board.occupied_co[chess.WHITE]
        bitboards = [
            [pcs & mask_black, pcs & mask_white]
            for pcs in (
                board.kings,
                board.pawns,
                board.knights,
                board.bishops,
                board.rooks,
                board.queens,
            )
        ]
        array = np.asarray([bitboards], dtype=np.uint64).ravel()
        return np.append(array, np.uint64(board.turn))

    model = tt.NNUE()
    tt.load_bin(model, bin_path)
    model.eval()  # quantized forward, matches the engine

    golds = {}
    with torch.no_grad():
        for fen in TESTS:
            board = chess.Board(fen=fen)
            assert board.is_valid(), f"Invalid position: {fen}"
            x = torch.from_numpy(encode(board).astype(np.int64)).reshape(1, 13)
            res = model(x)[0, 0].item()
            golds[fen] = res * 100

    assert golds[MIRROR_PAIR[0]] == golds[MIRROR_PAIR[1]], "mirror pair golds differ"

    with open(GOLDS_JSON, "w") as f:
        json.dump(golds, f, indent=2)
    print(f"Wrote {len(golds)} golds to {GOLDS_JSON}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("weights", help="weights.bin")
    _generate(p.parse_args().weights)
