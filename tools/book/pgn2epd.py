#!/usr/bin/env python3
# Extract the positions of PGN games into an EPD book, to replay games from every ply.
import argparse
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path

import chess.pgn


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


def find_pgns(paths):
    for path in map(Path, paths):
        if path.is_dir():
            yield from sorted(path.rglob("*.pgn"))
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(description="Extract positions from PGN games into an EPD book")
    parser.add_argument("paths", nargs="+", help="PGN files, or folders to walk")
    parser.add_argument("-o", "--output", required=True, help="EPD output file")
    parser.add_argument("-s", "--step", type=int, default=1, help="keep every Nth ply of each game (default: 1)")
    args = parser.parse_args()

    if args.step < 1:
        parser.error("step must be >= 1")

    missing = [path for path in args.paths if not Path(path).exists()]
    if missing:
        parser.error(f'not found: {", ".join(missing)}')

    pgns = list(find_pgns(args.paths))
    if not pgns:
        parser.error("no PGN files found")

    seen = set()
    total = 0
    bad_games = 0
    tmp = f"{args.output}.tmp"  # a failed run must not leave a partial book
    try:
        with multiprocessing.Pool() as pool, open(tmp, "w") as out:
            for fens, bad in pool.imap(partial(read_file, args.step), pgns):
                total += len(fens)
                bad_games += bad
                for fen in fens:
                    key = " ".join(fen.split()[:4])  # ignore move counters
                    if key not in seen:
                        seen.add(key)
                        out.write(fen + "\n")
        if seen:
            os.replace(tmp, args.output)
    except OSError as e:
        sys.exit(f"{parser.prog}: error: {e}")
    finally:
        Path(tmp).unlink(missing_ok=True)

    if not seen:
        sys.exit(f"{parser.prog}: error: no positions found")

    print(f"{len(pgns)} files, {total} positions, {len(seen)} unique -> {args.output}")
    if bad_games:
        print(f"skipped {bad_games} bad games", file=sys.stderr)


if __name__ == "__main__":
    main()
