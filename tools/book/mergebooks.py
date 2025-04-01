import chess
import chess.polyglot
import argparse
import struct
from collections import defaultdict
import logging

ENTRY_STRUCT = struct.Struct('>QHHI')

def encode_move(move: chess.Move) -> int:
    promotion = ((move.promotion - 1) & 0x7) << 12 if move.promotion else 0
    return move.to_square | (move.from_square << 6) | promotion

def write_polyglot_book(output_path, entries, compress):
    with open(output_path, 'wb') as output:
        for key, moves in sorted(entries.items()):
            if all(entry.weight == 0 for entry in moves):
                logging.warning(f"All entries for key {key} have zero weight!")
                if compress:
                    continue  # Skip writing entries with zero weight if compression is enabled
            for entry in moves:
                if compress and entry.weight == 0:
                    continue  # Skip zero-weight entries if compression is enabled
                packed_entry = ENTRY_STRUCT.pack(key, encode_move(entry.move), entry.weight, entry.learn)
                output.write(packed_entry)

def merge_polyglot_books(book1_path, book2_path, output_path, verbose=False, compress=False):
    entries = defaultdict(list)

    logging.info(f"Reading first book: {book1_path}")
    with chess.polyglot.open_reader(book1_path) as reader:
        for entry in reader:
            entries[entry.key].append(entry)
    logging.info(f"Loaded {sum(len(m) for m in entries.values())} entries from {book1_path}")

    logging.info(f"Reading second book: {book2_path}")
    with chess.polyglot.open_reader(book2_path) as reader:
        count_overwritten = 0
        for entry in reader:
            if entry in entries[entry.key]:
                if verbose:
                    logging.warning(f"Skipping duplicate")
                continue  # Avoid duplicate exact moves
            count_overwritten += 1
            existing_weights = [e.weight for e in entries[entry.key]]
            max_existing_weight = max(existing_weights) if existing_weights else -1
            if verbose:
                logging.info(f"Adding new move for key {entry.key}: {entry.move.uci()}, Weight: {entry.weight}, Max existing weight: {max_existing_weight}")
            entries[entry.key].append(entry)
    logging.info(f"Loaded {sum(len(m) for m in entries.values())} total entries after merging, {count_overwritten} new moves added")

    logging.info(f"Writing merged book to {output_path}")
    write_polyglot_book(output_path, entries, compress)
    logging.info(f"Merged book saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two Polyglot opening books.")
    parser.add_argument("book1", help="Path to the first Polyglot book.")
    parser.add_argument("book2", help="Path to the second Polyglot book.")
    parser.add_argument("output", help="Path to save the merged Polyglot book.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for overwrites.")
    parser.add_argument("-l", "--log", help="Log output to a file instead of the console.")
    parser.add_argument("-c", "--compress", action="store_true", help="Exclude entries with zero weight from the output.")
    args = parser.parse_args()

    # Configure logging
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, filename=args.log if args.log else None)

    merge_polyglot_books(args.book1, args.book2, args.output, args.verbose, args.compress)
