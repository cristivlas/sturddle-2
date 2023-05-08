#!/usr/bin/env python3
import argparse
from dbutils.sqlite import SQLConn
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='+', help='List of input filenames')
parser.add_argument('-o', '--output', required=True, help='Output filename')
parser.add_argument('-d', '--depth-min', type=int, help='Minimum depth')
args = parser.parse_args()

create_table_query = '''
CREATE TABLE IF NOT EXISTS position(
    epd text PRIMARY KEY,   -- Position
    depth integer,          -- Analysis Depth
    score integer           -- Score
)'''

insert_query = '''INSERT OR IGNORE INTO position VALUES (?, ?, ?)'''

total_records = 0

# Iterate over the input filenames
for filename in args.filenames:
    print(filename)

    # Connect to the input database and count the number of records
    with SQLConn(filename) as input_conn:
        if args.depth_min is not None:
            record_count = input_conn.exec('SELECT COUNT(*) FROM position WHERE depth >= ?', (args.depth_min,)).fetchone()[0]
        else:
            record_count = input_conn.exec('SELECT COUNT(*) FROM position').fetchone()[0]

        total_records += record_count

# Iterate over the input filenames with a progress bar
with tqdm(total=total_records, desc='Merging positions', unit='position') as progress_bar:
    for filename in args.filenames:
        # Connect to the input database and select positions
        with SQLConn(filename) as input_conn:
            if args.depth_min is not None:
                input_positions = input_conn.exec('SELECT * FROM position WHERE depth >= ?', (args.depth_min,))
            else:
                input_positions = input_conn.exec('SELECT * FROM position')

            # Connect to the output database, create table if not exists and insert positions
            with SQLConn(args.output) as output_conn:
                output_conn.exec(create_table_query)
                for position in input_positions:
                    output_conn.exec(insert_query, position)
                    progress_bar.update(1)
