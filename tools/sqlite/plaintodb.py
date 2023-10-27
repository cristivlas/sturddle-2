#!/usr/bin/env python3
import argparse
import mmap

from dbutils.sqlite import SQLConn
from queue import Queue
from threading import Thread
from tqdm import tqdm


class SQLConnExtended(SQLConn):
    def _create_table_if_not_exists(self):
        _create_table = '''
        CREATE TABLE IF NOT EXISTS position(
            epd text PRIMARY KEY,
            score integer
        )'''
        self.exec(_create_table)
        self._conn.commit()

    def __init__(self, db_file):
        super().__init__(db_file)
        self._create_table_if_not_exists()


def parse_lines(map_file, queue, progress_bar):
    fen = b''
    score = 0
    for line in iter(map_file.readline, b''):
        if line.startswith(b'fen'):
            fen = line[4:].strip()
        elif line.startswith(b'score'):
            score = int(line[6:])
        elif line.startswith(b'e'):
            queue.put((fen.decode('utf-8'), score))
            progress_bar.update()


def write_to_db(db_path, queue, batch_size):
    with SQLConnExtended(db_path) as conn:
        batch = []
        while True:
            record = queue.get()
            if record is None:  # Sentinel value to indicate completion
                break
            batch.append(record)
            if len(batch) >= batch_size:
                conn.executemany('INSERT OR IGNORE INTO position VALUES (?, ?)', batch)
                batch.clear()

        if batch:  # handle any remaining records
            conn.executemany('INSERT OR IGNORE INTO position VALUES (?, ?)', batch)


def process_file(file_path, db_path, batch_size, record_count=None):
    with open(file_path, 'r') as file:
        map_file = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)

        if record_count is None:
            print('Counting lines...')
            line_count = 0
            for _ in iter(map_file.readline, b''):
                line_count += 1
            record_count = line_count // 6
            print(f'Done: {line_count} lines, {record_count} records.')
            map_file.seek(0)  # Reset the mmap object to start of file

        progress_bar = tqdm(total=record_count)
        record_queue = Queue(maxsize=1000 * batch_size)

        parser_thread = Thread(target=parse_lines, args=(map_file, record_queue, progress_bar))
        db_thread = Thread(target=write_to_db, args=(db_path, record_queue, batch_size))

        parser_thread.start()
        db_thread.start()

        parser_thread.join()
        record_queue.put(None)  # Sentinel value to indicate completion to db_thread
        db_thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert text file to SQLite3 database.')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('-b', '--batch-size', type=int, default=10000)
    parser.add_argument('-o', '--output', help='Output SQLite3 database path')
    parser.add_argument('-r', '--records', type=int, help='Number of records to process')
    args = parser.parse_args()

    process_file(args.input, args.output, args.batch_size, args.records)
