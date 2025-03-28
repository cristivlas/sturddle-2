#! /usr/bin/env python3
import os
import csv
import sqlite3
import argparse
from tqdm import tqdm

# Define the database schema
CREATE_TABLE_QUERY = '''
CREATE TABLE IF NOT EXISTS position(epd text, depth integer, score integer)
'''

INSERT_QUERY = '''
INSERT OR REPLACE INTO position (epd, depth, score) VALUES (?, ?, ?)
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process CSV files and store data in SQLite database.')
    parser.add_argument('directory', type=str, help='Directory containing files')
    parser.add_argument('database', type=str, help='SQLite database file')
    return parser.parse_args()

def create_table(connection):
    with connection:
        connection.execute(CREATE_TABLE_QUERY)

def process_file(file_path, connection):
    with connection:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                epd, uci_move, depth, score = row
                depth = int(depth)
                score = int(score)
                connection.execute(INSERT_QUERY, (epd, depth, score))

def main():
    args = parse_arguments()

    # Connect to the SQLite database
    connection = sqlite3.connect(args.database)

    # Create the table if it does not exist
    create_table(connection)

    # Iterate over all files in the specified directory
    for root, _, files in os.walk(args.directory):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            try:
                process_file(file_path, connection)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Close the database connection
    connection.close()

if __name__ == '__main__':
    main()
