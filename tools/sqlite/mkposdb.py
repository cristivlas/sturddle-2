#! /usr/bin/env python3
'''
Create a database of positions from PGN games.
'''
import argparse
import os
import sqlite3

from chessutils.pgn import *
from dbutils.sqlite import SQLConn
from fileutils.zipcrawl import ZipCrawler
from tqdm import tqdm

_create_table = """CREATE TABLE IF NOT EXISTS position(
    epd text,               -- Position
    prev text,              -- Previous position
    move integer,           -- Move number
    uci text,               -- Move in UCI notation
    cnt integer,            -- Count total times played
    win integer,            -- Wins
    loss integer,           -- Losses
    PRIMARY KEY (epd, move)
)"""


def _read_games(args, sql):
    byte_count = 0
    for f in args._files:
        byte_count += os.path.getsize(f)
    file_count = len(args._files)

    # Take a guess at the average game size.
    avg_game_size = 1024

    total_game_count = 0
    pbar = tqdm(total=int(byte_count / avg_game_size))
    for n, filepath in enumerate(args._files):
        with open(filepath, 'r', encoding='latin1') as f:
            cur_file_size = os.path.getsize(filepath)
            file_game_count = 0
            pbar.desc = filepath
            for game in read_games(f, unique=False):
                try:
                    board = chess.Board(fen=game.headers['FEN'])
                except:
                    board = chess.Board()
                epd = board.epd()
                meta = game_metadata(game)
                for i, move in enumerate(game.mainline_moves(), start=1):
                    result = meta[chess.COLOR_NAMES[board.turn]]['result']
                    wdl = 2 * result - 1
                    assert wdl in [-1, 0, 1]
                    prev = epd
                    try:
                        board.push(move)
                    except:
                        break
                    if args.validate:
                        assert board.is_valid(), (filepath, game.headers['Event'], board.fen())
                    epd = board.epd()
                    sql.exec(
                        '''
                        INSERT OR IGNORE INTO position(epd, prev, move, uci, cnt, win, loss)
                        VALUES(?, ?, ?, ?, 0, 0, 0)
                        ''',
                        (epd, prev, i, move.uci())
                    )
                    sql.exec(
                        '''
                        UPDATE position SET cnt = cnt + 1,
                        win = win + ?,
                        loss = loss + ?
                        WHERE epd = ?
                        ''',
                        (wdl > 0, wdl < 0, epd))

                # Adjust the guess-timate for the total game count:
                file_game_count += 1
                if n == 0:
                    # first file
                    avg_game_size = f.tell() / file_game_count
                else:
                    avg_game_size = (
                        (file_count - n) * f.tell() / file_game_count
                        + n * byte_count / max(1, total_game_count)
                    ) / file_count

                pbar.total = int(byte_count / avg_game_size)
                pbar.update(1)

                total_game_count += 1
                if total_game_count % 10000 == 0:
                    sql.commit()

    pbar.pos = pbar.total = total_game_count


def main(args):
    if args.cleanup and os.path.exists(args.output):
        os.unlink(args.output)
    with SQLConn(args.output) as sql:
        sql.exec(_create_table)
        with ZipCrawler(args.input) as crawler:
            args._files = []
            crawler.set_action('.pgn', lambda f: args._files.append(f))
            try:
                crawler.crawl()
                _read_games(args, sql)
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create sqlite3 positions database from PGNs')
    parser.add_argument('input', nargs='+', help='list of PGN files and folders with PGNs, or zipped PGNs')
    parser.add_argument('-o', '--output', required=True, help='file name for the output sqlite3 database')
    parser.add_argument('-c', '--cleanup', action='store_true', help='delete database if already exists')
    parser.add_argument('-v', '--validate', action='store_true', help='validate board positions')
    args = parser.parse_args()

    main(args)
