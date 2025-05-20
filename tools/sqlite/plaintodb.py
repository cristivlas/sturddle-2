#!/usr/bin/env python3
import argparse
import mmap
import os
import signal
import time

from dbutils.sqlite import SQLConn
from queue import Queue
from threading import Thread
from tqdm import tqdm


class SQLConnExtended(SQLConn):
    def _create_table_if_not_exists(self):
        _create_table = '''
        CREATE TABLE IF NOT EXISTS position(
            epd text PRIMARY KEY,
            depth integer,
            score integer,
            best_move_uci text,
            best_move_from integer,
            best_move_to integer
        )'''
        self.exec(_create_table)
        self._conn.commit()

    def __init__(self, db_file):
        super().__init__(db_file)
        self._create_table_if_not_exists()


shutdown = []


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


def parse_lines(map_file, queue, batch_size, progress_bar, record_limit):
    fen = b''
    score = 0
    move = b''
    depth = 0
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
        elif line.startswith(b'ply'):
            depth = int(line[4:])
        elif line.startswith(b'e'):
            move_str = move.decode('utf-8')
            from_square, to_square = uci_to_square_indices(move_str)
            queue.put((fen.decode('utf-8'), depth, score, move_str, from_square, to_square))
            progress_bar.update()
            record_count += 1

            # Check if we've reached the limit after processing a record
            if record_limit is not None and record_count >= record_limit:
                queue.put(None)
                break


def write_to_db(db_path, queue, batch_size, progress_bar):
    INSERT_SQL = 'INSERT OR IGNORE INTO position (epd, depth, score, best_move_uci, best_move_from, best_move_to) VALUES (?, ?, ?, ?, ?, ?)'

    with SQLConnExtended(db_path) as conn:
        batch = []
        while True:
            record = queue.get()
            if record is None:  # Sentinel value to indicate completion
                break

            batch.append(record)
            if len(batch) >= batch_size:
                conn.executemany(INSERT_SQL, batch)
                progress_bar.update(len(batch))
                batch.clear()

        if batch:  # handle any remaining records
            progress_bar.update(len(batch))
            conn.executemany(INSERT_SQL, batch)


def process_file(args):
    queue = Queue(maxsize=1000 * args.batch_size)

    with open(args.input, 'r') as file:
        map_file = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)

        record_count = args.records

        # Count records if not provided in the command line
        if record_count is None:
            print('Counting lines...')
            line_count = 0
            for _ in iter(map_file.readline, b''):
                line_count += 1
            record_count = line_count // 6
            print(f'Done: {line_count} lines, {record_count} records.')
            map_file.seek(0)  # Reset the mmap object to start of file

        read_progress = tqdm(desc='Reading', total=record_count)
        write_progress = tqdm(desc='Writing', total=record_count)

        parser_thread = Thread(target=parse_lines, args=(map_file, queue, args.batch_size, read_progress, record_count))
        db_thread = Thread(target=write_to_db, args=(args.output, queue, args.batch_size, write_progress))

        parser_thread.start()
        db_thread.start()

        parser_thread.join()
        queue.put(None)  # Sentinel value to indicate completion to db_thread
        db_thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert text file to SQLite3 database.')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('-b', '--batch-size', type=int, default=10000)
    parser.add_argument('-o', '--output', help='Output SQLite3 database path')
    parser.add_argument('-r', '--records', type=int, help='Number of records to process')
    args = parser.parse_args()

    def signal_handler(signal, frame):
        if shutdown:
            os._exit(signal)
        shutdown.append(True)
        print(f'\nShutting down on signal {signal} in {frame}')

    signal.signal(signal.SIGINT, signal_handler)

    process_file(args)
