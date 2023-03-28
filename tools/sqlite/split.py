#!/usr/bin/env python3
import argparse
import os
import sqlite3
import shutil
from pathlib import Path

def split_db(input_db, ratio):
    input_db_path = Path(input_db)
    train_db_path = input_db_path.with_stem(f"{input_db_path.stem}-train")
    valid_db_path = input_db_path.with_stem(f"{input_db_path.stem}-valid")

    # Copy the input database to create train and valid databases
    shutil.copyfile(input_db, train_db_path)
    shutil.copyfile(input_db, valid_db_path)

    conn_train = sqlite3.connect(train_db_path)
    conn_valid = sqlite3.connect(valid_db_path)
    conn_input = sqlite3.connect(input_db)

    cur_input = conn_input.cursor()
    cur_train = conn_train.cursor()
    cur_valid = conn_valid.cursor()

    # Get the total number of rows in the input database
    cur_input.execute("SELECT COUNT(*) FROM position")
    total_rows = cur_input.fetchone()[0]
    conn_input.close()

    # Calculate the number of rows for the validation set
    valid_rows = int(total_rows * ratio)
    train_rows = total_rows - valid_rows

    # Delete rows from train and valid databases
    cur_train.execute(f"DELETE FROM position WHERE _rowid_ > {train_rows}")
    cur_valid.execute(f"DELETE FROM position WHERE _rowid_ <= {train_rows}")

    # Commit and close connections
    conn_train.commit()
    cur_train.execute("VACUUM")
    cur_train.close()
    conn_valid.commit()
    conn_valid.execute("VACUUM")
    conn_valid.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a SQLite3 database into train and validation databases.")
    parser.add_argument("input_db", type=str, help="Path to the input SQLite3 database")
    parser.add_argument("ratio", type=float, help="Ratio of data to be used for validation")
    args = parser.parse_args()

    if not 0 <= args.ratio <= 1:
        raise ValueError("The ratio must be between 0 and 1.")

    split_db(args.input_db, args.ratio)
