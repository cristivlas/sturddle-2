#! /usr/bin/env python3
import argparse
import os

from dbutils.sqlite import SQLConn
from tqdm.contrib import tenumerate

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+', help='List of input DB filenames')
parser.add_argument('-o', '--output', required=True, help='Output DB filename')
args = parser.parse_args()


def merge_databases(input_dbs, output_db):
    insert_count = 0
    update_count = 0

    # Connect to the output database and create the position table if it doesn't exist
    with SQLConn(output_db) as conn:
        conn.exec('CREATE TABLE IF NOT EXISTS position(epd text PRIMARY KEY, depth integer, score integer)')

        # Loop over the input databases and insert the entries into the output database
        for db in input_dbs:
            print(db)
            with SQLConn(db) as input_conn:
                max_row = input_conn.row_max_count('position')
                for _, row in tenumerate(input_conn.exec('SELECT * FROM position'), total=max_row):
                    epd, depth, score = row
                    existing = conn.exec('SELECT depth FROM position WHERE epd = ?', (epd,)).fetchone()
                    if existing is None or depth > existing[0]:
                        conn.exec('REPLACE INTO position (epd, depth, score) VALUES (?, ?, ?)', (epd, depth, score))
                        if existing:
                            update_count += 1
                        else:
                            insert_count += 1

    print(f'Inserted: {insert_count} positions, updated: {update_count}.')

try:
    merge_databases(args.inputs, args.output)
except KeyboardInterrupt:
    pass
