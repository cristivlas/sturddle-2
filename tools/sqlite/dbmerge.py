#! /usr/bin/env python3
import argparse
import sqlite3

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='+', help='List of input filenames')
parser.add_argument('-o', '--output', required=True, help='Output filename')
parser.add_argument('-d', '--depth-min', type=int, help='Minimum depth')
args = parser.parse_args()

# Connect to the output database
output_conn = sqlite3.connect(args.output)
output_cursor = output_conn.cursor()

# Create the output table if it doesn't exist
output_cursor.execute('''CREATE TABLE IF NOT EXISTS position(
    epd text PRIMARY KEY,   -- Position
    depth integer,          -- Analysis Depth
    score integer           -- Score
)''')

# Iterate over the input filenames
for filename in args.filenames:
    print(filename)
    # Connect to the input database
    input_conn = sqlite3.connect(filename)
    input_cursor = input_conn.cursor()

    # Select positions from the input database
    if args.depth_min:
        input_cursor.execute('''SELECT * FROM position WHERE depth >= ?''', (args.depth_min,))
    else:
        input_cursor.execute('''SELECT * FROM position''')

    # Insert the selected positions into the output database
    output_cursor.executemany('''INSERT OR IGNORE INTO position VALUES (?, ?, ?)''', input_cursor)

# Commit the changes to the output database
output_conn.commit()

# Close the connections
output_conn.close()
input_conn.close()
