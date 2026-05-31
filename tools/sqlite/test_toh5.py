#!/usr/bin/env python3
"""
Tests for toh5.py — the sqlite -> HDF5 training-data converter.

Focus: the conversion is correct AND robust against the numpy-2.x / h5py-3.x
"no appropriate function for conversion path" failure that per-cell np.uint64(...)
scalar writes could intermittently trigger. Shapes mirror the production DBs in
C:\\Users\\cristian\\data (new schema: 17 cols, old schema: 14 cols).

Uses only the stdlib unittest runner. All artifacts go to a TemporaryDirectory;
nothing else on disk is touched.  Run:  python test_toh5.py
"""
import os
import sqlite3
import sys
import tempfile
import types
import unittest

import chess
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import toh5

# Real rows sampled from hnat-sturddle-0.db (new schema):
# (epd, score, outcome, best_move_from, best_move_to)
NEW_SCHEMA_ROWS = [
    ('r1bq3r/ppp1k2p/1b3p1B/3np2Q/2B5/2PP4/PP3PPP/R3K2R w KQ -', -67, 1, 19, 27),
    ('r1bqk2r/pppp2pp/5n2/4pP2/2Bb4/2NP4/PPP2PPP/R1BQK2R b KQkq -', -10, -1, 51, 35),
    ('r6r/ppp1k2p/1b2bpqB/3nP3/2B4Q/2P5/PP3PPP/R3K2R w KQ -', -61, 1, 4, 2),
    ('r3r3/ppp1k2p/4BnqB/8/7Q/2P5/PP4PP/2K1R3 b - -', -739, -1, 56, 59),
    # edge values: beyond clip boundary, outcome extremes, square 0 and 63
    ('8/8/5k2/2R3p1/7p/5P1P/1r4P1/6K1 b - -', 30000, -1, 0, 63),
    ('8/8/R7/5kp1/1r5p/5P1P/6PK/8 b - -', -30000, 0, 63, 0),
]

OLD_SCHEMA_ROWS = [(r[0], r[1]) for r in NEW_SCHEMA_ROWS]

CLIP = 15000


def _make_db(path, rows, new_schema):
    con = sqlite3.connect(path)
    cur = con.cursor()
    if new_schema:
        cur.execute(
            "CREATE TABLE position("
            "epd TEXT, score INTEGER, best_move_uci TEXT, best_move_san TEXT, "
            "best_move_from INTEGER, best_move_to INTEGER, outcome INTEGER)"
        )
        cur.executemany(
            "INSERT INTO position(epd,score,best_move_from,best_move_to,outcome) "
            "VALUES(?,?,?,?,?)",
            [(r[0], r[1], r[3], r[4], r[2]) for r in rows],
        )
    else:
        cur.execute("CREATE TABLE position(epd TEXT, score INTEGER, depth INTEGER)")
        cur.executemany("INSERT INTO position(epd,score) VALUES(?,?)", rows)
    con.commit()
    con.close()


def _args(input_db, output):
    a = types.SimpleNamespace()
    a.input = [input_db]
    a.begin = None
    a.row_count = None
    a.clip = CLIP
    a.output = output
    a.test = False
    return a


def _expected_score(raw):
    return np.uint64(np.int64(int(np.clip(raw, -CLIP, CLIP))).astype(np.uint64))


class ToH5Test(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def _run(self, rows, new_schema):
        db = os.path.join(self.dir, "in.db")
        out = os.path.join(self.dir, "out.h5")
        _make_db(db, rows, new_schema)
        toh5.main(_args(db, out))
        with h5py.File(out, "r") as f:
            return f["data"][...]

    # --- shape ---
    def test_new_schema_shape(self):
        data = self._run(NEW_SCHEMA_ROWS, True)
        self.assertEqual(data.shape, (len(NEW_SCHEMA_ROWS), 17))
        self.assertEqual(data.dtype, np.uint64)

    def test_old_schema_shape(self):
        data = self._run(OLD_SCHEMA_ROWS, False)
        self.assertEqual(data.shape, (len(OLD_SCHEMA_ROWS), 14))
        self.assertEqual(data.dtype, np.uint64)

    # --- correctness ---
    def test_board_roundtrips(self):
        data = self._run(NEW_SCHEMA_ROWS, True)
        for i, row in enumerate(NEW_SCHEMA_ROWS):
            board = chess.Board(row[0] + " 0 1")
            board.castling_rights = 0
            board.ep_square = None
            decoded = toh5.decode(data[i, :13])
            self.assertEqual(decoded.epd(), board.epd(), (i, row[0]))

    def test_score_clipped_and_wrapped(self):
        data = self._run(NEW_SCHEMA_ROWS, True)
        for i, row in enumerate(NEW_SCHEMA_ROWS):
            self.assertEqual(data[i, 13], _expected_score(row[1]), (i, row[1]))

    def test_outcome_offset(self):
        data = self._run(NEW_SCHEMA_ROWS, True)
        for i, row in enumerate(NEW_SCHEMA_ROWS):
            self.assertEqual(data[i, 14], np.uint64(row[2] + 1), (i, row[2]))

    def test_best_move_squares(self):
        data = self._run(NEW_SCHEMA_ROWS, True)
        for i, row in enumerate(NEW_SCHEMA_ROWS):
            self.assertEqual(data[i, 15], np.uint64(row[3]), (i, "from"))
            self.assertEqual(data[i, 16], np.uint64(row[4]), (i, "to"))

    # --- robustness regression: the OSError conversion-path bug ---
    def test_large_volume_no_conversion_error(self):
        rows = (NEW_SCHEMA_ROWS * 4000)[:20000]  # ~20k production-shaped rows
        data = self._run(rows, True)
        self.assertEqual(data.shape, (20000, 17))
        for i in (0, 1, 5, 19999):  # col 16 is the one that crashed in prod
            self.assertEqual(data[i, 16], np.uint64(rows[i][4]), i)


if __name__ == "__main__":
    unittest.main(verbosity=2)
