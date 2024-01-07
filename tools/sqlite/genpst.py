#!/usr/bin/env python3
import argparse
import sqlite3
import chess
import os
import json
from collections import defaultdict
from tqdm import tqdm

PIECE_VALUE = [85, 319, 343, 522, 986, 2000]
SIGN = [-1, 1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate piece-square tables from chess position databases."
    )
    parser.add_argument("db_files", nargs="*", help="SQLite database file paths")
    parser.add_argument("-o", "--output", help="Output file path for C code")
    parser.add_argument(
        "-i",
        "--intermediate",
        default="intermediate.db",
        help="Intermediate SQLite database file",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1.0,
        help="Scaling factor for the piece-square tables",
    )
    parser.add_argument("-m", "--max-score", type=int, default=15000)
    return parser.parse_args()


def init_intermediary_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS progress (file_path TEXT PRIMARY KEY, row_count INT, last_row INT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS piece_square_data (piece TEXT, square INT, score_sum INT, count INT, PRIMARY KEY (piece, square))"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS input_files (file_list TEXT)")
    return conn


def update_intermediary_db(conn, piece, square, score):
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO piece_square_data (piece, square, score_sum, count)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(piece, square) DO UPDATE SET score_sum=score_sum+?, count=count+1""",
        (piece, square, score, 1, score),
    )


def update_row_count(conn, file_path, row_count):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO progress (file_path, row_count) VALUES (?, ?) ON CONFLICT(file_path) DO UPDATE SET row_count = ?",
        (file_path, row_count, row_count),
    )


def update_progress(conn, file_path, last_row):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO progress (file_path, last_row) VALUES (?, ?) ON CONFLICT(file_path) DO UPDATE SET last_row = ?",
        (file_path, last_row, last_row),
    )


def get_last_processed_row(conn, file_path):
    cur = conn.cursor()
    cur.execute(
        "SELECT last_row, row_count FROM progress WHERE file_path = ?", (file_path,)
    )
    row = cur.fetchone()
    return row if row else (None, None)


def store_input_files(conn, input_files):
    file_metadata = [
        [file, os.path.getmtime(file), os.path.getsize(file)] for file in input_files
    ]
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO input_files (file_list) VALUES (?)", (json.dumps(file_metadata),)
    )
    conn.commit()


def verify_input_files(conn, input_files):
    cur = conn.cursor()
    cur.execute("SELECT file_list FROM input_files")
    stored_metadata = cur.fetchone()
    if stored_metadata:
        stored_files = json.loads(stored_metadata[0])
        current_files = [
            [file, os.path.getmtime(file), os.path.getsize(file)]
            for file in input_files
        ]
        if stored_files != current_files:
            print(stored_files, current_files)
            raise ValueError("Input files do not match the initial run.")
    else:
        store_input_files(conn, input_files)


def eval_material(board):
    material = 0  # material from white's perspective
    piece_masks = [board.pawns, board.knights, board.bishops, board.rooks, board.queens]
    for color in chess.COLORS:
        color_mask = board.occupied_co[color]
        for i, mask in enumerate(piece_masks):
            material += SIGN[color] * chess.popcount(mask & color_mask) * PIECE_VALUE[i]

    return material * SIGN[board.turn]  # material for side-to-move


def process_positions(db_file, intermediary_conn, max_score):
    last_processed_row, total_rows = get_last_processed_row(intermediary_conn, db_file)
    start_row = last_processed_row + 1 if last_processed_row is not None else 0
    initial = 0
    if last_processed_row is not None:
        print(f"Resuming {db_file} from row {last_processed_row}")
        initial = last_processed_row

    with sqlite3.connect(db_file) as db_conn:
        cursor = db_conn.cursor()

        if total_rows is None:
            cursor.execute("SELECT COUNT(*) FROM position")
            total_rows = cursor.fetchone()[0]
            update_row_count(intermediary_conn, db_file, total_rows)

        board = chess.Board()
        cursor.execute(
            "SELECT rowid, epd, score FROM position WHERE rowid >= ?", (start_row,)
        )
        for rowid, epd, score in tqdm(
            cursor,
            initial=initial,
            total=total_rows,
            desc=f"Processing {os.path.basename(db_file)}",
        ):
            if max_score is not None and abs(score) > max_score:
                continue
            board.set_epd(epd)

            piece_count = chess.popcount(board.occupied)
            avg_score = (score - eval_material(board)) / piece_count

            for square in chess.SquareSet(board.occupied):
                piece = board.piece_at(square)

                update_intermediary_db(
                    intermediary_conn,
                    piece.symbol(),
                    square,
                    avg_score * SIGN[piece.color == board.turn],
                )
            update_progress(intermediary_conn, db_file, rowid)
            if rowid % 10000 == 0:
                intermediary_conn.commit()


def generate_c_code(conn, scale, output_file):
    cursor = conn.cursor()
    cursor.execute("SELECT piece, square, score_sum, count FROM piece_square_data")
    data = cursor.fetchall()

    # Initialize tables for black and white pieces
    tables = [{}, {}]

    for piece_sym, square, score_sum, count in data:
        color = piece_sym.isupper()
        piece_sym = piece_sym.lower()
        pt = chess.PIECE_SYMBOLS.index(piece_sym) - 1

        if piece_sym not in tables[color]:
            tables[color][piece_sym] = {}
        tables[color][piece_sym][square] = score_sum * scale / count

    with open(output_file, "w") as f:
        f.write("#pragma once\n\n")
        f.write('#include "common.h"\n')
        f.write(
            (
                "/*\n"
                " * Piece-square tables.\n"
                " * https://www.chessprogramming.org/Simplified_Evaluation_Function\n"
                " */\n"
            )
        )
        f.write("static constexpr int SQUARE_TABLE[2][7][64] = {\n")

        # Output for black and then white
        for color in [chess.BLACK, chess.WHITE]:
            f.write(f"    {{ /* {chess.COLOR_NAMES[color].capitalize()} */\n")
            # First entry for each color is an empty table
            f.write("        { /* None */ 0 },\n")
            for p in chess.PIECE_TYPES:
                piece_sym = chess.PIECE_SYMBOLS[p]
                piece_name = chess.PIECE_NAMES[p]
                # print(p, piece_sym, piece_name)
                f.write(f"        {{ /* {piece_name} */\n")

                for rank in range(8):
                    f.write("            ")
                    f.write(
                        ", ".join(
                            f"{int(tables[color].get(piece_sym, {}).get(chess.square(file, rank), 0)):5d}"
                            for file in range(8)
                        )
                    )
                    f.write(",\n")
                f.write("        },\n")
            f.write("    },\n")
        f.write("};\n")


def main():
    args = parse_args()

    with init_intermediary_db(args.intermediate) as intermediary_conn:
        if args.output:
            generate_c_code(intermediary_conn, args.scale, args.output)
        else:
            input_files = [os.path.abspath(f) for f in args.db_files]
            try:
                verify_input_files(intermediary_conn, input_files)
            except ValueError as e:
                print(e)
                return

            for db_file in input_files:
                if os.path.exists(db_file):
                    process_positions(db_file, intermediary_conn, args.max_score)
                else:
                    print(f"Warning: Database file not found - {db_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        ...
