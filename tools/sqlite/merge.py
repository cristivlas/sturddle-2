#!/usr/bin/env python3

import sqlite3
import argparse
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Merge multiple SQLite position databases')
    parser.add_argument('output', help='Output database path')
    parser.add_argument('inputs', nargs='+', help='Input database paths')
    parser.add_argument('--replace', action='store_true', 
                        help='Replace duplicates instead of ignoring (keeps last seen)')
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    
    # Count total rows first
    total_rows = 0
    input_paths = []
    for db_path in tqdm(args.inputs, desc="Counting rows"):
        db_path = Path(db_path).resolve()
        if db_path == output_path or not db_path.exists():
            continue
        input_paths.append(db_path)
        with sqlite3.connect(db_path) as src:
            total_rows += src.execute("SELECT COUNT(*) FROM position").fetchone()[0]
    
    print(f"Total rows to process: {total_rows:,}")
    
    with sqlite3.connect(output_path) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS position(
            epd TEXT PRIMARY KEY,
            score INTEGER,
            best_move_uci TEXT,
            best_move_san TEXT,
            best_move_from INTEGER,
            best_move_to INTEGER,
            outcome INTEGER
        )''')
        
        insert_cmd = "INSERT OR REPLACE" if args.replace else "INSERT OR IGNORE"
        
        with tqdm(total=total_rows, desc="Inserting", unit="rows") as pbar:
            for db_path in input_paths:
                with sqlite3.connect(db_path) as src:
                    cursor = src.execute("SELECT * FROM position")
                    for row in cursor:
                        conn.execute(f"{insert_cmd} INTO position VALUES (?,?,?,?,?,?,?)", row)
                        pbar.update(1)
    
    with sqlite3.connect(output_path) as conn:
        final_count = conn.execute("SELECT COUNT(*) FROM position").fetchone()[0]
    print(f"Done. Unique positions: {final_count:,}")


if __name__ == '__main__':
    main()
