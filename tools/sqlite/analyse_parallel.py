#! /usr/bin/env python3
'''
Use chess engine to build a training DB of scored positions.
'''
import argparse
import multiprocessing
import os
import signal
import time
from math import copysign

import chess
import chess.engine
import chess.pgn
import dbutils.sqlite
from tqdm import tqdm


def handle_sigint(sig, frame):
    os.killpg(0, signal.SIGTERM)

signal.signal(signal.SIGINT, handle_sigint)


_create_table = '''CREATE TABLE IF NOT EXISTS position(
    epd text PRIMARY KEY,   -- Position
    depth integer,          -- Analysis Depth
    score integer           -- Score
)'''

_insert = '''INSERT INTO position(epd, depth, score) VALUES(?,?,?)'''


def analyse(args, queue, lock, progress, event):
    engine = get_engine(args)
    try:
        while True:
            epd = queue.get()
            if epd is None:
                break
            board = chess.Board(fen=epd)
            if not board.is_valid():
                continue
            limit = chess.engine.Limit(depth=args.depth) if args.depth else chess.engine.Limit(time=args.time_limit)
            info = engine.analyse(board, limit)
            score = info['score']
            depth = info['depth']
            with lock:
                with dbutils.sqlite.SQLConn(args.output) as sql_out:
                    if not score.is_mate():
                        sql_out.exec(_insert, (epd,  info['depth'], score.pov(score.turn).score()))
                    elif args.mate_score:
                        mate_dist = score.pov(score.turn).mate()
                        score = int(copysign(args.mate_score, mate_dist) - (mate_dist))
                        sql_out.exec(_insert, (epd,  info['depth'], score))
                    progress.value += 1
                    event.set()
    finally:
        engine.quit()


def get_engine(args):
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({'Threads': args.threads, 'Hash': args.hash})
    return engine


processes = []


def main(args):
    if args.cleanup and os.path.exists(args.output):
        os.unlink(args.output)

    with dbutils.sqlite.SQLConn(args.output) as sql_out:
        sql_out.exec(_create_table)

    with dbutils.sqlite.SQLConn(*args.input) as sql_in:
        # Create a queue to store the EPDs
        queue = multiprocessing.Queue()
        lock = multiprocessing.Lock()

        count = sql_in.row_max_count('position')
        query = '''SELECT DISTINCT(epd) from position'''

        progress = multiprocessing.Value('i', 0)

        # progress event
        event = multiprocessing.Event()

        with tqdm(progress, total=count, desc='Analysing') as pbar:
            # Start worker processes
            for _ in range(args.num_workers):
                p = multiprocessing.Process(target=analyse, args=(args, queue, lock, progress, event))
                processes.append(p)
                p.start()

            # Put the EPDs in the queue
            for row in sql_in.exec(query):
                epd = row[0]
                with lock:
                    with dbutils.sqlite.SQLConn(args.output) as sql_out:
                        res = sql_out.exec(f'SELECT EXISTS(SELECT (epd) FROM position WHERE epd="{epd}")')
                        # skip over existing rows
                        if res.fetchone()[0]:
                            progress.value += 1
                            pbar.update(1)
                            continue
                    pbar.n = progress.value
                    pbar.refresh()
                queue.put(epd)

            # Put sentinels in the queue to signal the worker processes to stop
            for _ in range(args.num_workers):
                queue.put(None)

            # periodically check the progress and update the tqdm progress bar
            while multiprocessing.active_children():
                with lock:
                    pbar.n = progress.value
                    pbar.refresh()
                time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1, help='sqlite3 database')
    parser.add_argument('-c', '--cleanup', action='store_true', help='delete database if it exists')
    parser.add_argument('-d', '--depth', type=int)
    parser.add_argument('-e', '--engine', default='stockfish')
    parser.add_argument('--hash', type=int, default=1024, help='engine hashtable size in MB')
    parser.add_argument('-m', '--mate-score', type=int, default=29999)
    parser.add_argument('-n', '--num_workers', type=int, default=1)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-t', '--time-limit', type=float, default=0.1)
    parser.add_argument('--threads', type=int, default=1)

    main(parser.parse_args())
    # Wait for all worker processes to finish
    for p in processes:
        p.join()
