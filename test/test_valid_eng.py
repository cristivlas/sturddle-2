#! /usr/bin/env python3
import argparse
import chess
import chess.engine
from tqdm import tqdm

def is_valid_move(board, move):
    return move is not None and move in board.legal_moves

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to EPD file")
    parser.add_argument("uci_engine", help="Path to UCI engine")
    parser.add_argument("-t", "--time", type=int, default=5000, help="Search time in ms")
    parser.add_argument("-n", "--threads", type=int, default=1, help="Number of engine threads")
    args = parser.parse_args()

    with open(args.input_file) as f:
        lines = [line.strip().split(';')[0] for line in f if line.strip()]

    with chess.engine.SimpleEngine.popen_uci(args.uci_engine) as engine:
        try:
            engine.configure({"Threads": args.threads})
        except chess.engine.EngineError:
            print("[!] Warning: Engine does not support 'Threads' option")

        for i, epd in enumerate(tqdm(lines, desc="Testing positions")):
            board = chess.Board(epd)

            try:
                res1 = engine.play(board, chess.engine.Limit(time=args.time / 1000))
            except Exception as e:
                print(f"\n[!] Engine error on first move at position {i+1}")
                print(f"EPD: {epd}\nError: {e}")
                return

            if not is_valid_move(board, res1.move):
                print(f"\n[!] Invalid/missing move on first search at position {i+1}")
                print(f"EPD: {epd}\nFirst engine response: {res1}")
                return

            board.push(res1.move)

            if board.is_game_over():  # handles checkmate, stalemate, etc.
                continue

            try:
                res2 = engine.play(board, chess.engine.Limit(time=args.time / 1000))
            except Exception as e:
                print(f"\n[!] Engine error on second move at position {i+1}")
                print(f"EPD: {epd}\nFirst move: {res1.move}\nError: {e}")
                return

            if not is_valid_move(board, res2.move):
                print(f"\n[!] Invalid/missing move on second search at position {i+1}")
                print(f"EPD: {epd}\nFirst move: {res1.move}\nSecond engine response: {res2}")
                return

    print("\nAll positions passed.")

if __name__ == "__main__":
    main()
