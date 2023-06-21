#!/usr/bin/env python3
import argparse
import mmap
import sys

from dbutils.sqlite import SQLConn
from tqdm import tqdm


class SQLConnExtended(SQLConn):
    def create_table_if_not_exists(self):
        _create_table = '''
        CREATE TABLE IF NOT EXISTS position(
            epd text PRIMARY KEY,
            depth integer,
            score integer
        )'''
        self.exec(_create_table)
        self._conn.commit()


def process_file(file_path, db_path):
    with SQLConnExtended(db_path) as conn:
        conn.create_table_if_not_exists()

        with open(file_path, 'r') as file:
            map_file = mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)

        line_count = 0
        for _ in iter(map_file.readline, b''):
            line_count += 1
        record_count = line_count // 6
        map_file.seek(0)  # Reset the mmap object to start of file

        progress_bar = tqdm(total=record_count)

        fen = b''
        depth = 0
        score = 0
        processed_records = 0

        try:
            for line in iter(map_file.readline, b''):
                if line.startswith(b'fen'):
                    fen = line[4:].strip()
                elif line.startswith(b'score'):
                    score = int(line[6:])
                elif line.startswith(b'ply'):
                    depth = int(line[4:])
                elif line.startswith(b'e'):
                    conn.exec('INSERT OR IGNORE INTO position VALUES (?, ?, ?)', (fen.decode('utf-8'), depth, score))
                    processed_records += 1
                    if processed_records % 1000 == 0:
                        conn.commit()
                    progress_bar.update()
        except KeyboardInterrupt:
            print('\nInterrupted')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert text file to SQLite3 database.')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('-o', '--output', help='Output SQLite3 database path')
    args = parser.parse_args()

    process_file(args.input, args.output)
