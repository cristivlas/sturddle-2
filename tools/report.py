#!/usr/bin/env python3
import argparse
import math
import os
from collections import defaultdict, namedtuple

import chess.pgn

RESULTS = {"1-0": (1, 0), "0-1": (0, 1), "1/2-1/2": (0.5, 0.5)}

Rating = namedtuple("Rating", "elo margin low high los")


class Record:
    def __init__(self):
        self.white = [0, 0, 0]  # wins, draws, losses
        self.black = [0, 0, 0]

    def add(self, side, points):
        getattr(self, side)[(1, 0.5, 0).index(points)] += 1

    def side(self, name):
        if name == "combined":
            return [w + b for w, b in zip(self.white, self.black)]
        return getattr(self, name)


def score(wdl):
    return wdl[0] + 0.5 * wdl[1]


def elo(p):
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    return -400.0 * math.log10(1.0 / p - 1.0)


def rating(wdl):
    """None when there are no games or the score is 0% or 100%."""
    wins, draws, losses = wdl
    games = sum(wdl)
    if not games:
        return None
    p = score(wdl) / games
    if p in (0, 1):
        return None
    variance = (wins * (1 - p) ** 2 + losses * p**2 + draws * (0.5 - p) ** 2) / games
    margin = 1.96 * math.sqrt(variance / games)
    low, high = elo(p - margin), elo(p + margin)
    decisive = wins + losses
    los = 50 * (1 + math.erf((wins - losses) / math.sqrt(2 * decisive))) if decisive else 50.0
    return Rating(elo(p), (high - low) / 2, low, high, los)


def sort_elo(wdl):
    """+inf for a perfect score, -inf for a zero score or no games."""
    r = rating(wdl)
    if r:
        return r.elo
    return math.inf if sum(wdl) and wdl[1] == wdl[2] == 0 else -math.inf


def overall(wdl):
    games = sum(wdl)
    return f"{score(wdl):.1f}/{games} ({100 * wdl[0] / games:.1f}% wins, {100 * wdl[1] / games:.1f}% draws)"


def print_side(label, wdl, indent):
    games = sum(wdl)
    if not games:
        return
    print(
        f"{indent}As {label}: {score(wdl):.1f}/{games} (+{wdl[0]} ={wdl[1]} -{wdl[2]}) [{100 * score(wdl) / games:.1f}%]"
    )
    r = rating(wdl)
    if r:
        print(f"{indent}   Elo: {r.elo:+.2f} +/- {r.margin:.2f} (LOS: {r.los:.1f}%)")


def parse_pgn(path):
    players = defaultdict(Record)
    openings = defaultdict(lambda: defaultdict(Record))
    total = 0
    with open(path) as f:
        while (headers := chess.pgn.read_headers(f)) is not None:
            points = RESULTS.get(headers.get("Result", "*"))
            if not points:
                continue
            total += 1
            opening = headers.get("Opening", "Unknown Opening")
            names = (headers.get("White", "Unknown"), headers.get("Black", "Unknown"))
            for side, name, p in zip(("white", "black"), names, points):
                players[name].add(side, p)
                openings[name][opening].add(side, p)
    return players, openings, total


def print_report(players, total_games):
    ranked = sorted(players.items(), key=lambda item: score(item[1].side("combined")), reverse=True)

    print(f"\nTotal games: {total_games}")
    print(
        f"\n{'Rank':<6} {'Player':<25} {'Score':<10} {'Games':<8} {'W':<6} {'D':<6} {'L':<6} {'Win%':<8} {'Draw%':<8}"
    )
    for rank, (name, record) in enumerate(ranked, 1):
        wdl = record.side("combined")
        games = sum(wdl)
        print(
            f"{rank:<6} {name:<25} {score(wdl):<10.1f} {games:<8} {wdl[0]:<6} {wdl[1]:<6} {wdl[2]:<6}"
            f" {100 * wdl[0] / games:<8.1f} {100 * wdl[1] / games:<8.1f}"
        )

    if len(ranked) == 2:
        (name1, record1), (name2, _) = ranked
        r = rating(record1.side("combined"))
        if r:
            print("\nHEAD-TO-HEAD\n")
            print(f"{name1} vs {name2}:")
            print(f"  Elo difference: {r.elo:+.2f} +/- {r.margin:.2f}")
            print(f"  95% confidence: [{r.low:+.2f}, {r.high:+.2f}]")
            print(f"  LOS: {r.los:.2f}%")

    print("\nWHITE vs BLACK")
    for rank, (name, record) in enumerate(ranked, 1):
        print(f"\n{rank}. {name}")
        print(f"   Overall: {overall(record.side('combined'))}")
        print_side("White", record.white, "   ")
        print_side("Black", record.black, "   ")

    print()
    for side in ("white", "black"):
        wdl = [sum(column) for column in zip(*(getattr(r, side) for r in players.values()))]
        if wdl and sum(wdl):
            print(f"{side.title()} overall performance: {100 * score(wdl) / sum(wdl):.2f}%")


def print_player_details(name, players, openings, sort_by):
    if name not in players:
        print(f"Error: Player '{name}' not found in tournament")
        print(f"Available players: {', '.join(players)}")
        return

    print(f"\nDETAILED REPORT FOR: {name}")
    print(f"\nOverall Performance: {overall(players[name].side('combined'))}")

    ranked = sorted(openings[name].items(), key=lambda item: sort_elo(item[1].side(sort_by)), reverse=True)
    order = "combined ELO" if sort_by == "combined" else f"{sort_by}-side ELO"
    print(f"\nPERFORMANCE BY OPENING ({len(ranked)} unique openings, sorted by {order})")
    for opening, record in ranked:
        games = sum(record.side("combined"))
        wdl = record.side(sort_by)
        r = rating(wdl)
        if r:
            label = f"{sort_by.title()} ELO: {r.elo:+.2f}"
        elif not sum(wdl):
            label = f"{sort_by.title()} ELO: N/A (no games)"
        else:
            label = f"ELO: N/A ({'perfect' if sort_elo(wdl) > 0 else 'zero'} score)"
        print(f"\n{opening} ({games} games) - {label}")
        print_side("White", record.white, "  ")
        print_side("Black", record.black, "  ")


def main():
    parser = argparse.ArgumentParser(description="Tournament report with white vs black statistics from a PGN file")
    parser.add_argument("pgn_file")
    parser.add_argument("--details", metavar="PLAYER", help="per-opening breakdown for one player")
    parser.add_argument(
        "--sort",
        choices=["combined", "white", "black"],
        default="combined",
        help="sort --details openings by combined (default), white-side or black-side Elo",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.pgn_file):
        parser.error(f"{args.pgn_file} not found")

    players, openings, total = parse_pgn(args.pgn_file)
    if args.details:
        print_player_details(args.details, players, openings, args.sort)
    else:
        print_report(players, total)


if __name__ == "__main__":
    main()
