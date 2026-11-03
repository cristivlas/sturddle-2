#!/usr/bin/env python3
# Extract the positions of PGN games into an EPD book, to replay games from every ply.
import argparse
import multiprocessing
import os
import signal
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

import chess.pgn
from tqdm import tqdm


def positions(game, step):
    board = game.board()
    ply = 0
    for node in game.mainline():
        if not node.comment.startswith("book"):
            if ply % step == 0:
                yield board.fen()
            ply += 1
        board.push(node.move)
    if ply % step == 0 and not board.is_game_over():
        yield board.fen()


def read_file(step, path):
    fens = []
    bad_games = 0
    with open(path, errors="replace") as f:
        while (game := chess.pgn.read_game(f)) is not None:
            if game.errors:
                bad_games += 1
            else:
                fens.extend(positions(game, step))
    return fens, bad_games


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the main process handles Ctrl+C
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def interrupt(signum, frame):
    raise KeyboardInterrupt


def find_pgns(paths):
    for path in map(Path, paths):
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix.lower() == ".pgn" and p.is_file())
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(description="Extract positions from PGN games into an EPD book")
    parser.add_argument(
        "paths", nargs="+", help="PGN files, or folders to walk (symlinked subfolders are not followed)"
    )
    parser.add_argument("-o", "--output", required=True, help="EPD output file")
    parser.add_argument("-s", "--step", type=int, default=1, help="keep every Nth ply of each game (default: 1)")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, interrupt)

    if args.step < 1:
        parser.error("step must be >= 1")

    missing = [path for path in args.paths if not Path(path).exists()]
    if missing:
        parser.error(f'not found: {", ".join(missing)}')

    print("scanning for PGN files", file=sys.stderr)
    pgns = list(find_pgns(args.paths))
    if not pgns:
        parser.error("no PGN files found")
    folders = Counter(path.parent for path in pgns)
    for folder, count in sorted(folders.items()):
        print(f"{folder}: {count:,} PGN files", file=sys.stderr)
    print(f"found {len(pgns):,} PGN files in {len(folders):,} folders", file=sys.stderr)

    seen = set()
    total = 0
    bad_games = 0
    tmp = f"{args.output}.tmp"  # a failed run must not leave a partial book
    pool = ProcessPoolExecutor(initializer=init_worker)
    try:
        with open(tmp, "w") as out:
            results = pool.map(partial(read_file, args.step), pgns)
            with tqdm(results, total=len(pgns), unit="file", dynamic_ncols=True) as progress:
                for fens, bad in progress:
                    total += len(fens)
                    bad_games += bad
                    for fen in fens:
                        key = " ".join(fen.split()[:4])  # ignore move counters
                        if key not in seen:
                            seen.add(key)
                            out.write(fen + "\n")
                    progress.set_postfix(positions=f"{total:,}", unique=f"{len(seen):,}", bad=bad_games, refresh=False)
        if seen:
            os.replace(tmp, args.output)
    except (OSError, BrokenProcessPool) as e:
        sys.exit(f"{parser.prog}: error: {e}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for worker in multiprocessing.active_children():  # don't wait for files in progress
            worker.terminate()
        Path(tmp).unlink(missing_ok=True)

    if not seen:
        sys.exit(f"{parser.prog}: error: no positions found")

    print(f"{len(pgns)} files, {total} positions, {len(seen)} unique -> {args.output}")
    if bad_games:
        print(f"skipped {bad_games} bad games", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
