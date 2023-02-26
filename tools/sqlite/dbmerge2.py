#! /usr/bin/env python3
import argparse
import os
import sqlite3

from tqdm.contrib import tenumerate

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+', help='List of input DB filenames')
parser.add_argument('-o', '--output', required=True, help='Output DB filename')
args = parser.parse_args()


def merge_databases(input_dbs, output_db):
    # Connect to the output database and create the position table if it doesn't exist
    conn = sqlite3.connect(output_db)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS position(epd text PRIMARY KEY, depth integer, score integer)')

    # Loop over the input databases and insert the entries into the output database
    for db in input_dbs:
        print(db)
        input_conn = sqlite3.connect(db)
        input_cursor = input_conn.cursor()

        max_row = input_cursor.execute(f'SELECT MAX(_ROWID_) FROM position').fetchone()[0]
        for _, row in tenumerate(input_cursor.execute('SELECT * FROM position'), total=max_row):
            epd, depth, score = row
            existing = c.execute('SELECT depth FROM position WHERE epd = ?', (epd,)).fetchone()
            if existing is None or depth > existing[0]:
                c.execute('REPLACE INTO position (epd, depth, score) VALUES (?, ?, ?)', (epd, depth, score))

        input_conn.close()

    conn.commit()
    conn.close()

merge_databases(args.inputs, args.output)
