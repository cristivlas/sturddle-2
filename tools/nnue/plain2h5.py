#!/usr/bin/env python3
import argparse
import mmap
import os
import re
import signal
import time
import chess
import h5py
import numpy as np
from tqdm import tqdm
from queue import Queue
from threading import Thread

FEN_PATTERN = re.compile(rb'^fen (.+)\r?$')
SCORE_PATTERN = re.compile(rb'^score (-?\d+)\r?$')
MOVE_PATTERN = re.compile(rb'^move (.+)\r?$')
RESULT_PATTERN = re.compile(rb'^result (-?\d+)\r?$')


'''
Convert chess.Board to array of features (packed as np.uint64)
Discard castling rights and en-passant square info.
'''
def encode(board, test=False):
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

    if test:  # run builtin unit test
        board.castling_rights = 0
        board.ep_square = None
        expected = board.epd()
        actual = decode(array).epd()
        assert expected == actual, (expected, actual)

    return array

'''
Convert encoding back to board, for verification.
'''
def decode(array):
    turn = array[12]
    bitboards = [int(x) for x in list(array[:12])]
    assert len(bitboards) == 12, bitboards
    board = chess.Board(fen=None)
    for b in bitboards:
        board.occupied |= b
    for b in bitboards[::2]:
        board.occupied_co[chess.BLACK] |= b
    for b in bitboards[1::2]:
        board.occupied_co[chess.WHITE] |= b
    board.kings = bitboards[0] | bitboards[1]
    board.pawns = bitboards[2] | bitboards[3]
    board.knights = bitboards[4] | bitboards[5]
    board.bishops = bitboards[6] | bitboards[7]
    board.rooks = bitboards[8] | bitboards[9]
    board.queens = bitboards[10] | bitboards[11]
    board.turn = turn
    assert board.is_valid()
    return board

def uci_to_square_indices(uci_move):
    """Convert UCI move notation (e.g., 'e2e4') to square indices (from, to)."""
    if not uci_move or len(uci_move) < 4:
        return None, None

    # UCI notation uses algebraic coordinates like 'e2e4'
    # Convert to 0-63 indices where a1=0, b1=1, ..., h8=63
    file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

    try:
        from_file = file_map.get(uci_move[0].lower())
        from_rank = int(uci_move[1]) - 1
        to_file = file_map.get(uci_move[2].lower())
        to_rank = int(uci_move[3]) - 1

        if None in (from_file, to_file) or not (0 <= from_rank <= 7) or not (0 <= to_rank <= 7):
            return None, None

        from_square = from_rank * 8 + from_file
        to_square = to_rank * 8 + to_file

        return from_square, to_square
    except (IndexError, ValueError):
        return None, None

def format(num):
    if num >= 1e9:
        return f'{num / 1e9:.1f}G'
    elif num >= 1e6:
        return f'{num / 1e6:.1f}M'
    elif num >= 1e3:
        return f'{num / 1e3:.1f}K'
    else:
        return str(num)

def output_path(args):
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    return f"{base_name}-{format(args.begin) if args.begin else 0}-{format(args.row_count) if args.row_count else 'all'}.h5"

shutdown = []

def parse_fen_to_packed(fen_str):
   """Parse FEN directly to packed format, skip python-chess."""
   parts = fen_str.split()
   piece_placement = parts[0]
   turn = 1 if parts[1] == 'w' else 0

   # Initialize piece bitboards to match encode() order
   kings = 0
   pawns = 0
   knights = 0
   bishops = 0
   rooks = 0
   queens = 0

   # Track occupied squares by color
   mask_black = 0
   mask_white = 0

   rank = 7  # Start at rank 8 (index 7)
   file = 0  # Start at file a (index 0)

   for c in piece_placement:
       if c.isdigit():
           file += int(c)
       elif c == '/':
           rank -= 1  # Move to next rank down
           file = 0   # Reset to file a
       else:
           square = rank * 8 + file  # Correct square calculation
           square_bit = 1 << square

           if c.isupper():  # White piece
               mask_white |= square_bit
           else:  # Black piece
               mask_black |= square_bit

           piece_char = c.lower()
           if piece_char == 'k':
               kings |= square_bit
           elif piece_char == 'p':
               pawns |= square_bit
           elif piece_char == 'n':
               knights |= square_bit
           elif piece_char == 'b':
               bishops |= square_bit
           elif piece_char == 'r':
               rooks |= square_bit
           elif piece_char == 'q':
               queens |= square_bit

           file += 1

   # Build bitboards array to match encode() format:
   # [[pcs & mask_black, pcs & mask_white] for pcs in (kings, pawns, knights, bishops, rooks, queens)]
   bitboards = [
       [kings & mask_black, kings & mask_white],
       [pawns & mask_black, pawns & mask_white],
       [knights & mask_black, knights & mask_white],
       [bishops & mask_black, bishops & mask_white],
       [rooks & mask_black, rooks & mask_white],
       [queens & mask_black, queens & mask_white]
   ]

   # Flatten and add turn to match: np.asarray([bitboards], dtype=np.uint64).ravel()
   array = np.asarray([bitboards], dtype=np.uint64).ravel()
   array = np.append(array, np.uint64(turn))

   return array

def parse_and_process(map_file, queue, batch_size, clip, test, progress_bar, record_limit, max_score, filter):
    fen = b''
    score = 0
    move = b''
    result = 0
    line_count = 0
    record_count = 0

    for line in iter(map_file.readline, b''):
        if shutdown or (record_limit is not None and record_count >= record_limit):
            queue.put(None)
            break

        line_count += 1
        if line_count % batch_size == 0:
            time.sleep(0.01)  # Yield execution

        if (match := FEN_PATTERN.match(line)):
            fen = match.group(1)
        elif (match := SCORE_PATTERN.match(line)):
            score = int(match.group(1))
        elif (match := MOVE_PATTERN.match(line)):
            move = match.group(1)
        elif (match := RESULT_PATTERN.match(line)):
            result = int(match.group(1))
        elif line.startswith(b'e'):
            move_str = move.decode('utf-8')
            from_square, to_square = uci_to_square_indices(move_str)

            # Process the chess position
            fen_str = fen.decode('utf-8')
            if test or filter:
                expected = chess.Board(fen_str)
                move = chess.Move.from_uci(move_str)
                if not expected.is_legal(move):
                    # TODO: logging
                    # print (fen, expected.epd(), move)
                    continue
                # assert expected.is_legal(move)
                expected.castling_rights = 0
                expected.ep_square = None
                encoded_board = parse_fen_to_packed(fen_str)
                actual = decode(encoded_board).epd()
                assert expected.epd() == actual, (expected, actual)

                if filter and (expected.is_capture(move) or expected.is_check() or expected.gives_check(move)):
                    continue
            else:
                encoded_board = parse_fen_to_packed(fen_str)

            if max_score is None or abs(score) <= max_score:
                # Put the processed data in the queue
                clipped_score = score if clip is None else np.clip(score, -clip, clip)
                queue.put((
                    encoded_board,
                    clipped_score,
                    result,
                    from_square if from_square is not None else 0,
                    to_square if to_square is not None else 0
                ))

                progress_bar.update()
                record_count += 1

                # Check if we've reached the limit after processing a record
                if record_limit is not None and record_count >= record_limit:
                    queue.put(None)
                    break

def write_to_h5(h5_file, queue, progress_bar, total_records, batch_size, shuffle):
    dtype = np.uint64
    dataset = h5_file.create_dataset('data', shape=(total_records, 17), dtype=dtype, maxshape=(None, 17))

    batch_buffer = np.zeros((batch_size, 17), dtype=np.uint64)
    batch_idx = 0
    index = 0

    while True:
        record = queue.get()
        if record is None:
            # Write remaining batch
            if batch_idx > 0:
                if shuffle:
                    # Shuffle the filled portion of the batch
                    np.random.shuffle(batch_buffer[:batch_idx])
                dataset[index:index+batch_idx] = batch_buffer[:batch_idx]
            break

        # Fill batch buffer
        encoded_board, score, result, from_square, to_square = record
        batch_buffer[batch_idx, :13] = encoded_board
        batch_buffer[batch_idx, 13] = score & 0xFFFFFFFFFFFFFFFF
        assert result in [-1, 0, 1]
        batch_buffer[batch_idx, 14] = result + 1
        batch_buffer[batch_idx, 15] = from_square
        batch_buffer[batch_idx, 16] = to_square

        batch_idx += 1
        progress_bar.update()

        # Write when batch is full
        if batch_idx == batch_size:
            if shuffle:
                # Shuffle the batch before writing
                np.random.shuffle(batch_buffer)
            dataset[index:index+batch_size] = batch_buffer
            index += batch_size
            batch_idx = 0

def count_records(file_path):
    """Count the number of records (positions) in the input file."""
    print('Counting records...')
    with open(file_path, 'r') as file:
        map_file = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)
        record_count = 0

        for line in iter(map_file.readline, b''):
            if line.startswith(b'e'):
                record_count += 1

        map_file.close()
    print(f'Found {record_count} records in the input file.')
    return record_count

def skip_records(map_file, begin):
    """Skip records efficiently using sampling-based estimation."""
    assert begin > 0

    map_file.seek(0)
    sample_records = 0

    for _ in range(min(10000, begin // 10)):
        line = map_file.readline()
        if not line:
            break
        if line.startswith(b'e'):
            sample_records += 1

    if sample_records == 0:
        raise ValueError("No records found - file appears to be malformed or wrong format")

    sample_bytes = map_file.tell()

    # Jump to estimated position
    avg_record_size = sample_bytes // sample_records
    target_pos = int(begin * avg_record_size)

    map_file.seek(min(target_pos, map_file.size() - 1))
    map_file.readline()
    # Find next end-of-record 'e' marker.
    skipped = target_pos // avg_record_size
    for line in iter(map_file.readline, b''):
        if line.startswith(b'e'):
            break

    return skipped

def process_file(args):
    if not args.row_count:
        # Count records if not provided
        total_records = count_records(args.input)
    else:
        # Adjust total records to include skipped + to be processed
        total_records = args.row_count + (args.begin if args.begin else 0)

    # If begin is specified, we need to skip some records
    begin = args.begin if args.begin else 0

    queue = Queue(maxsize=1000 * args.batch_size)

    with open(args.input, 'r') as file:
        map_file = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)

        # Skip records if begin is specified
        if begin > 0:
            print(f"Skipping first {begin} records...")
            skipped = skip_records(map_file, begin)
            print(f"Skipped ~{skipped} records. Now processing {args.row_count} records.")

        # Calculate how many records to process
        records_to_process = total_records - begin if args.row_count is None else args.row_count

        # Create H5 file
        if args.output is None:
            args.output = output_path(args)
        print(f'\nWriting out: {args.output}')

        if args.shuffle:
            print('Shuffle mode: ENABLED')

        h5_file = h5py.File(args.output, 'w')

        read_progress = tqdm(desc='Reading', total=records_to_process, colour='cyan')
        write_progress = tqdm(desc='Writing', total=records_to_process, colour='green')

        # Start the processing threads
        parser_thread = Thread(
            target=parse_and_process,
            args=(map_file, queue, args.batch_size, args.clip, args.test, read_progress, records_to_process, args.max_score, args.filter_moves)
        )

        writer_thread = Thread(
            target=write_to_h5,
            args=(h5_file, queue, write_progress, records_to_process, args.batch_size, args.shuffle)
        )

        parser_thread.start()
        writer_thread.start()

        parser_thread.join()
        queue.put(None)  # Sentinel value to indicate completion to writer thread
        writer_thread.join()

        h5_file.close()
        map_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert chess position data from plain text to H5 format')
    parser.add_argument('input', help='Input plain text file path')
    parser.add_argument('-b', '--begin', type=int, help='Index of first record to process')
    parser.add_argument('-B', '--batch-size', type=int, default=10000, help='Processing batch size')
    parser.add_argument('-c', '--clip', type=int, help='Clip score values [-clip, +clip]')
    parser.add_argument('-f', '--filter-moves', action='store_true', help='Filter out checks and captures')
    parser.add_argument('-m', '--max-score', type=int, help='Filter out positions with score above limit')
    parser.add_argument('-o', '--output', help='Output H5 file path')
    parser.add_argument('-r', '--row-count', type=int, help='Number of records to process')
    parser.add_argument('-s', '--shuffle', action='store_true', help='shuffle each batch before writing to H5 file')
    parser.add_argument('-t', '--test', action='store_true', help='Run encode/decode verification tests')
    args = parser.parse_args()

    def signal_handler(signal, frame):
        if shutdown:
            os._exit(signal)
        shutdown.append(True)
        print(f'\nShutting down on signal {signal} in {frame}')

    signal.signal(signal.SIGINT, signal_handler)

    try:
        process_file(args)
        print(f"Conversion complete! Output saved to: {args.output}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        pass
    except Exception as e:
        print(f"Error during conversion: {e}")
