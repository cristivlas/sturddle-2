import chess
import chess.polyglot
import argparse
import struct
from collections import defaultdict
import logging

ENTRY_STRUCT = struct.Struct('>QHHI')

def encode_move(move: chess.Move) -> int:
    """Encode a chess move to a polyglot integer format."""
    promotion = ((move.promotion - 1) & 0x7) << 12 if move.promotion else 0
    return move.to_square | (move.from_square << 6) | promotion

def is_castling_attempt(move: chess.Move) -> bool:
    """Check if a move appears to be a castling attempt based on the from square and move distance."""
    # Must be from king's starting position
    if move.from_square != chess.E1 and move.from_square != chess.E8:
        return False

    # Check distance - castling is typically 2 squares horizontally
    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    file_diff = abs(from_file - to_file)

    # Same rank (castling is horizontal)
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)

    # Must be a horizontal move of 2 or more squares on the same rank to be considered castling
    return file_diff >= 2 and from_rank == to_rank

def is_valid_castling_destination(move: chess.Move) -> bool:
    """Check if a castling move has the correct destination square."""
    # White king starting position
    if move.from_square == chess.E1:
        # Valid castling destinations for white
        return move.to_square in (chess.G1, chess.C1)
    # Black king starting position
    elif move.from_square == chess.E8:
        # Valid castling destinations for black
        return move.to_square in (chess.G8, chess.C8)

    # Not from king's starting position
    return False

def get_corrected_castling_move(move: chess.Move) -> chess.Move:
    """Get the corrected castling move for an invalid castling move."""
    from_square = move.from_square
    to_file = chess.square_file(move.to_square)

    # Determine if it's kingside or queenside based on the destination file
    if to_file > chess.square_file(from_square):  # Moving to the right (kingside)
        if from_square == chess.E1:  # White king
            return chess.Move(from_square, chess.G1)
        elif from_square == chess.E8:  # Black king
            return chess.Move(from_square, chess.G8)
    else:  # Moving to the left (queenside)
        if from_square == chess.E1:  # White king
            return chess.Move(from_square, chess.C1)
        elif from_square == chess.E8:  # Black king
            return chess.Move(from_square, chess.C8)

    # Should never get here if used correctly
    return move

def fix_polyglot_book(input_path, output_path, repair=False, verbose=False):
    """Fix a polyglot opening book by removing or repairing invalid castling moves."""
    entries = defaultdict(list)
    invalid_entries = 0
    repaired_entries = 0
    total_entries = 0

    logging.info(f"Reading book: {input_path}")
    with chess.polyglot.open_reader(input_path) as reader:
        for entry in reader:
            total_entries += 1
            move = entry.move

            # Check if it's a castling attempt with an invalid destination
            if is_castling_attempt(move) and not is_valid_castling_destination(move):
                invalid_entries += 1
                if repair:
                    # Create a corrected move
                    corrected_move = get_corrected_castling_move(move)

                    # Calculate the raw move encoding for the corrected move
                    raw_move = encode_move(corrected_move)

                    # Create a new entry with the corrected move
                    corrected_entry = chess.polyglot.Entry(
                        key=entry.key,
                        raw_move=raw_move,
                        move=corrected_move,
                        weight=entry.weight,
                        learn=entry.learn
                    )
                    entries[entry.key].append(corrected_entry)
                    repaired_entries += 1

                    if verbose:
                        logging.info(f"Repaired castling move: {move.uci()} -> {corrected_move.uci()}")
                elif verbose:
                    logging.warning(f"Removing invalid castling move: {move.uci()} from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}")
            else:
                # Keep all other moves unchanged
                entries[entry.key].append(entry)

    if repair:
        logging.info(f"Processed {total_entries} entries, repaired {repaired_entries} invalid castling moves")
    else:
        logging.info(f"Processed {total_entries} entries, removed {invalid_entries} invalid castling moves")

    # Write the fixed book
    logging.info(f"Writing fixed book to {output_path}")
    with open(output_path, 'wb') as output:
        for key, moves in sorted(entries.items()):
            for entry in moves:
                packed_entry = ENTRY_STRUCT.pack(key, encode_move(entry.move), entry.weight, entry.learn)
                output.write(packed_entry)

    logging.info(f"Fixed book saved to {output_path} with {sum(len(m) for m in entries.values())} entries")
    return invalid_entries, repaired_entries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix invalid castling moves in Polyglot opening books.")
    parser.add_argument("input", help="Path to the input Polyglot book.")
    parser.add_argument("output", help="Path to save the fixed Polyglot book.")
    parser.add_argument("-r", "--repair", action="store_true", help="Repair invalid castling moves instead of removing them.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for removed/repaired moves.")
    parser.add_argument("-l", "--log", help="Log output to a file instead of the console.")
    args = parser.parse_args()

    # Configure logging
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, filename=args.log if args.log else None)

    invalid_count, repaired_count = fix_polyglot_book(args.input, args.output, args.repair, args.verbose)

    if args.repair:
        print(f"Cleaning complete: {repaired_count} invalid castling moves repaired.")
    else:
        print(f"Cleaning complete: {invalid_count} invalid castling moves removed.")