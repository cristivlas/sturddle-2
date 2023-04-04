#!/usr/bin/env python3
import argparse
import os
import sys

from dbutils.sqlite import SQLConn
from tqdm import tqdm


def augment_database(primary_db, secondary_db):
    with SQLConn(primary_db) as primary_conn, SQLConn(secondary_db) as secondary_conn:
        primary_conn.add_column_if_not_exists('position', 'depth', 'integer')
        primary_conn.add_column_if_not_exists('position', 'score', 'integer')
        primary_conn.add_column_if_not_exists('position', 'has_evaluation', 'integer')

        secondary_data = secondary_conn.exec('SELECT epd, depth, score FROM position')
        for epd, depth, score in tqdm(secondary_data, total=secondary_conn.row_count('position')):
            primary_row = primary_conn.exec('SELECT depth, has_evaluation FROM position WHERE epd=?', (epd,)).fetchone()

            if primary_row:
                primary_depth, has_evaluation = primary_row
                if not has_evaluation or (depth > primary_depth):
                    primary_conn.exec('UPDATE position SET depth=?, score=?, has_evaluation=1 WHERE epd=?', (depth, score, epd))

        primary_conn.commit()
        print(f'Augmented primary database {primary_db} with depth and score information from {secondary_db}')

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

    args = parser.parse_args()

    try:
        augment_database(args.primary_db, args.secondary_db)
    except KeyboardInterrupt:
        pass
