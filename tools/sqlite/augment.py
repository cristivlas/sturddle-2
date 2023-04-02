#!/usr/bin/env python3
import argparse
import os
import sys

from dbutils.sqlite import SQLConn
from tqdm import tqdm


def augment_database(primary_db_file, secondary_db_file, batch_size=1000):
    with SQLConn(primary_db_file) as primary_db, SQLConn(secondary_db_file) as secondary_db:
        # Add depth, score, and has_evaluation columns to the primary database schema if they don't exist
        primary_db.add_column_if_not_exists('position', 'depth', 'integer')
        primary_db.add_column_if_not_exists('position', 'score', 'integer')
        primary_db.add_column_if_not_exists('position', 'has_evaluation', 'integer')

        # Iterate through rows of the secondary database using the cursor
        secondary_db_cursor = secondary_db.exec('SELECT * FROM position')
        secondary_db_row_count = secondary_db.row_count('position')

        updates = []
        for epd, depth, score in tqdm(secondary_db_cursor, total=secondary_db_row_count, desc='Augmenting primary database'):
            updates.append((depth, score, 1, epd))

            if len(updates) >= batch_size:
                primary_db.executemany('''
                    UPDATE position
                    SET depth=?, score=?, has_evaluation=?
                    WHERE epd=?
                ''', updates)
                primary_db.commit()
                updates = []

        # Process any remaining updates
        if updates:
            primary_db.executemany('''
                UPDATE position
                SET depth=?, score=?, has_evaluation=?
                WHERE epd=?
            ''', updates)
            primary_db.commit()


if __name__ == '__main__':
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter
    ):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description='Augment the primary database with depth and score information from the secondary database.')
    parser.add_argument('primary_db', metavar='primary_db_file', type=str, help='Primary database file')
    parser.add_argument('secondary_db', metavar='secondary_db_file', type=str, help='Secondary database file containing depth and score information')
    parser.add_argument('-b', '--batch_size', metavar='batch_size', type=int, default=1000, help='Number of updates to process in a batch (default: 1000)')

    args = parser.parse_args()

    try:
        augment_database(args.primary_db, args.secondary_db, args.batch_size)
        print(f'Augmented primary database {args.primary_db} with depth and score information from {args.secondary_db}')
    except KeyboardInterrupt:
        pass
