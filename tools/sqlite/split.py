#!/usr/bin/env python3
import argparse
from pathlib import Path

from dbutils.sqlite import SQLConn
from tqdm import tqdm


def split_db(input_db, ratio, table_name):
    input_db_path = Path(input_db)
    train_db_path = input_db_path.with_stem(f"{input_db_path.stem}-train")
    valid_db_path = input_db_path.with_stem(f"{input_db_path.stem}-valid")

    with SQLConn(input_db) as conn_input:
        # Create the specified table in train and valid databases
        create_table_sql = conn_input.exec(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'").fetchone()[0]

        with SQLConn(train_db_path) as conn_train, SQLConn(valid_db_path) as conn_valid:
            conn_train.exec(create_table_sql)
            conn_valid.exec(create_table_sql)

            # Get the total number of rows in the input database
            total_rows = conn_input.row_count(table_name)
            print(total_rows)

            # Calculate the number of rows for the validation set
            valid_rows = int(total_rows * ratio)
            train_rows = total_rows - valid_rows

            query = f"SELECT * FROM {table_name} ORDER BY RANDOM()"
            # Copy rows from input to train and valid databases with a progress indicator
            for idx, row in enumerate(tqdm(conn_input.exec(query), total=total_rows, desc="Processing rows")):
                columns = ','.join('?' * len(row))
                if idx < train_rows:
                    conn_train.exec(f"INSERT INTO {table_name} VALUES ({columns})", row)
                else:
                    conn_valid.exec(f"INSERT INTO {table_name} VALUES ({columns})", row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a SQLite3 database into train and validation databases.")
    parser.add_argument("input_db", type=str, help="Path to the input SQLite3 database")
    parser.add_argument("--ratio", type=float, required=True, help="Ratio of data to be used for validation")
    parser.add_argument("--table", type=str, required=True, help="Name of the table to split")
    args = parser.parse_args()

    if not 0 <= args.ratio <= 1:
        raise ValueError("The ratio must be between 0 and 1.")

    split_db(args.input_db, args.ratio, args.table)
