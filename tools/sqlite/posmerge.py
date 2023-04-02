#!/usr/bin/env python3
import os
import sys
import argparse
from tqdm import tqdm
from dbutils.sqlite import SQLConn

def merge_databases(input_files, output_file):
    with SQLConn(output_file) as output_db:
        # Create the output table if it doesn't exist
        output_db.exec('''
            CREATE TABLE IF NOT EXISTS position (
                epd text,
                prev text,
                move integer,
                uci text,
                cnt integer,
                win integer,
                loss integer,
                PRIMARY KEY (epd, move)
            )
        ''')

        # Iterate through input files and copy data to the output database
        for input_file in input_files:
            with SQLConn(input_file) as input_db:
                # Get the number of rows in the input database
                num_rows = input_db.row_count('position')

                # Get all distinct rows from the input database
                rows = input_db.exec('SELECT * FROM position')

                # Iterate through rows with a progress bar
                for row in tqdm(rows, total=num_rows, desc=f'Processing {input_file}'):
                    epd, prev, move, uci, cnt, win, loss = row

                    # Check if the position already exists in the output database
                    existing_row = output_db.exec('''
                        SELECT cnt, win, loss FROM position WHERE epd=? AND move=?
                    ''', (epd, move)).fetchone()

                    if existing_row:
                        cnt += existing_row[0]
                        win += existing_row[1]
                        loss += existing_row[2]
                        output_db.exec('''
                            UPDATE position
                            SET cnt=?, win=?, loss=?
                            WHERE epd=? AND move=?
                        ''', (cnt, win, loss, epd, move))
                    else:
                        output_db.exec('''
                            INSERT INTO position (epd, prev, move, uci, cnt, win, loss)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (epd, prev, move, uci, cnt, win, loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge multiple SQLite3 databases with the specified schema.')
    parser.add_argument('input_files', metavar='input_file', type=str, nargs='+', help='input database files to merge')
    parser.add_argument('-o', '--output', metavar='output_file', type=str, required=True, help='output database file')

    args = parser.parse_args()

    try:
        merge_databases(args.input_files, args.output)
        print(f'Merged databases into {args.output}')
    except KeyboardInterrupt:
        pass
