#!/usr/bin/env python3
import argparse
import mmap
import os
import signal
import time
import chess
import h5py
import numpy as np
from tqdm import tqdm
from queue import Queue
from threading import Thread

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

def parse_and_process(map_file, queue, batch_size, clip, test, progress_bar, record_limit):
    board = chess.Board()
    fen = b''
    score = 0
    move = b''
    line_count = 0
    record_count = 0

    for line in iter(map_file.readline, b''):
        if shutdown or (record_limit is not None and record_count >= record_limit):
            queue.put(None)
            break

        line_count += 1
        if line_count % batch_size == 0:
            time.sleep(0.01)  # Yield execution

        if line.startswith(b'fen'):
            fen = line[4:].strip()
        elif line.startswith(b'score'):
            score = int(line[6:])
        elif line.startswith(b'move'):
            move = line[5:].strip()
        elif line.startswith(b'e'):
            move_str = move.decode('utf-8')
            from_square, to_square = uci_to_square_indices(move_str)

            # Process the chess position
            fen_str = fen.decode('utf-8')
            board.set_fen(fen_str)
            clipped_score = np.clip(score, -clip, clip)
            encoded_board = encode(board, test).astype(np.uint64)

            # Put the processed data in the queue
            queue.put((
                encoded_board,
                clipped_score,
                from_square if from_square is not None else 0,
                to_square if to_square is not None else 0
            ))
            progress_bar.update()

            record_count += 1

            # Check if we've reached the limit after processing a record
            if record_limit is not None and record_count >= record_limit:
                queue.put(None)
                break

def write_to_h5(h5_file, queue, progress_bar, total_records):
    # Create dataset with the full shape
    # 13: board state (12 bitboards + turn), 1: score, 2: best_move_from, best_move_to
    dtype = np.uint64
    dataset = h5_file.create_dataset('data', shape=(total_records, 16), dtype=dtype)

    index = 0
    while True:
        record = queue.get()
        if record is None:  # Sentinel value to indicate completion
            break

        encoded_board, score, from_square, to_square = record

        # Store board state
        dataset[index, :13] = encoded_board

        # Store score
        dataset[index, 13] = np.uint64(score)

        # Store move information
        dataset[index, 14] = np.uint64(from_square)
        dataset[index, 15] = np.uint64(to_square)

        index += 1
        progress_bar.update()

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
            skipped = 0
            # Create a progress bar for skipping
            skip_progress = tqdm(desc='Skipping', total=begin)

            for line in iter(map_file.readline, b''):
                if line.startswith(b'e'):
                    skipped += 1
                    skip_progress.update(1)
                    if skipped >= begin:
                        break

            skip_progress.close()
            print(f"Skipped {skipped} records. Now processing {args.row_count} records.")

        # Calculate how many records to process
        records_to_process = total_records - begin if args.row_count is None else args.row_count

        # Create H5 file
        if args.output is None:
            args.output = output_path(args)
        print(f'\nWriting out: {args.output}')
        h5_file = h5py.File(args.output, 'w')

        read_progress = tqdm(desc='Reading', total=records_to_process, colour='cyan')
        write_progress = tqdm(desc='Writing', total=records_to_process, colour='green')

        # Start the processing threads
        parser_thread = Thread(
            target=parse_and_process,
            args=(map_file, queue, args.batch_size, args.clip, args.test, read_progress, records_to_process)
        )

        writer_thread = Thread(
            target=write_to_h5,
            args=(h5_file, queue, write_progress, records_to_process)
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
    parser.add_argument('-c', '--clip', type=int, default=15000, help='Clip score values to this range (-clip to +clip)')
    parser.add_argument('-r', '--row-count', type=int, help='Number of records to process')
    parser.add_argument('-o', '--output', help='Output H5 file path')
    parser.add_argument('-t', '--test', action='store_true', help='Run encode/decode tests')
    parser.add_argument('--batch-size', type=int, default=10000, help='Processing batch size')
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
