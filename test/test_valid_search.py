#! /usr/bin/env python3
import argparse
import chess
import os
import sys
from tqdm import tqdm


def root_path():
    return os.path.abspath(os.path.join(os.path.split(sys.argv[0])[0], '..'))

sys.path.append(root_path())

from chess_engine import MTDf_i  # or desired algorithm

def is_legal_move(board, move):
    return move in board.legal_moves

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the EPD input file")
    parser.add_argument("-t", "--time", type=int, default=5000, help="Search time limit in ms")
    args = parser.parse_args()

    with open(args.input_file) as f:
        lines = [line.strip().split(';')[0] for line in f if line.strip()]

    for i, fen in enumerate(tqdm(lines, desc="Testing positions")):
        board = chess.Board(fen=fen)
        algo = MTDf_i(board, time_limit_ms=args.time)
        move, _ = algo.search()

        if not move or not is_legal_move(board, move):
            print(f"\n[!] Failed at position {i+1}")
            print(f"FEN: {fen}")
            print(f"Move: {move}")
            return

    print("\nAll positions passed.")

if __name__ == "__main__":
    main()
