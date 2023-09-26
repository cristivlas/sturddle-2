#! /usr/bin/env python3
import argparse
import logging
import os

from dbutils.sqlite import SQLConn
from tqdm.contrib import tenumerate

def configure_logging(args):
    log = logging.getLogger()
    format = '%(asctime)s %(levelname)-8s %(process)d %(message)s'
    filename = f'{args.logfile}.{os.getpid()}'
    logging.basicConfig(level=logging.INFO, filename=filename, format=format)


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+', help='List of input DB filenames')
parser.add_argument('-o', '--output', required=True, help='Output DB filename')
parser.add_argument('-l', '--logfile', default='dbmerge2.log')

args = parser.parse_args()

configure_logging(args)

def merge_databases(input_dbs, output_db):
    logging.info(f'Merging {input_dbs} into: {output_db}')
    # Connect to the output database and create the position table if it doesn't exist
    with SQLConn(output_db) as conn:
        conn.exec('CREATE TABLE IF NOT EXISTS position(epd text PRIMARY KEY, depth integer, score integer)')

        # Loop over the input databases and insert the entries into the output database
        for db in input_dbs:
            print(f'Processing {db}')
            logging.info(f'--- {db} ---')
            with SQLConn(db) as input_conn:
                max_row = input_conn.row_max_count('position')
                logging.info(f'Estimated rows in {db}: {max_row}')
                for _, row in tenumerate(input_conn.exec('SELECT * FROM position'), total=max_row):
                    try:
                        epd, depth, score = row
                    except ValueError:
                        break
                    conn.exec('''
                          INSERT OR REPLACE INTO position (epd, depth, score)
                            SELECT
                                ?1,
                                ?2,
                                ?3
                            WHERE
                                NOT EXISTS (SELECT 1 FROM position WHERE epd = ?1)
                                OR ?2 > (SELECT depth FROM position WHERE epd = ?1);
                            ''',
                            (epd, depth, score))

            logging.info(f'Changes so far: {conn.total_changes()}')
        total_changes = f'Total changes made: {conn.total_changes()}'

    logging.info(total_changes)
    print(total_changes)

try:
    merge_databases(args.inputs, args.output)
except KeyboardInterrupt:
    pass
