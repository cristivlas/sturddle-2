import argparse
import logging
import struct
from collections import defaultdict

import chess
import chess.polyglot

ENTRY_STRUCT = struct.Struct(">QHHI")

# back-rank moves from e1/e8 taken as broken castling, whatever the piece: (from, to) -> corrected destination
CASTLING_FIX = {
    (chess.E1, chess.H1): chess.G1,
    (chess.E1, chess.B1): chess.C1,
    (chess.E1, chess.A1): chess.C1,
    (chess.E8, chess.H8): chess.G8,
    (chess.E8, chess.B8): chess.C8,
    (chess.E8, chess.A8): chess.C8,
}


def encode_move(move):
    promotion = ((move.promotion - 1) & 0x7) << 12 if move.promotion else 0
    return move.to_square | (move.from_square << 6) | promotion


def fix_polyglot_book(input_path, output_path, repair, verbose):
    entries = defaultdict(list)
    total = invalid = 0

    logging.info(f"Reading book: {input_path}")
    with chess.polyglot.open_reader(input_path) as reader:
        for entry in reader:
            total += 1
            move = entry.move
            fixed = CASTLING_FIX.get((move.from_square, move.to_square))
            if fixed is None:
                entries[entry.key].append((move, entry.weight, entry.learn))
                continue
            invalid += 1
            if repair:
                corrected = chess.Move(move.from_square, fixed)
                entries[entry.key].append((corrected, entry.weight, entry.learn))
                if verbose:
                    logging.info(f"Repaired castling move: {move.uci()} -> {corrected.uci()}")
            elif verbose:
                logging.warning(f"Removing invalid castling move: {move.uci()}")

    action = "repaired" if repair else "removed"
    logging.info(f"Processed {total} entries, {action} {invalid} invalid castling moves")

    logging.info(f"Writing fixed book to {output_path}")
    with open(output_path, "wb") as out:
        for key, moves in sorted(entries.items()):
            for move, weight, learn in moves:
                out.write(ENTRY_STRUCT.pack(key, encode_move(move), weight, learn))
    logging.info(f"Fixed book saved to {output_path} with {sum(map(len, entries.values()))} entries")
    return invalid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix invalid castling moves in Polyglot opening books.")
    parser.add_argument("input", help="input Polyglot book")
    parser.add_argument("output", help="output Polyglot book")
    parser.add_argument(
        "-r", "--repair", action="store_true", help="repair invalid castling moves instead of removing them"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log each removed or repaired move")
    parser.add_argument("-l", "--log", help="log to this file instead of the console")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename=args.log)

    count = fix_polyglot_book(args.input, args.output, args.repair, args.verbose)
    print(f"Cleaning complete: {count} invalid castling moves {'repaired' if args.repair else 'removed'}.")
