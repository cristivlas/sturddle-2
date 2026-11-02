#!/usr/bin/env python3
import math
import sys


def elo(score):
    if score <= 0:
        return -math.inf
    if score >= 1:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def main():
    if len(sys.argv) != 4:
        sys.exit(f"usage: {sys.argv[0]} <wins> <losses> <draws>")
    try:
        wins, losses, draws = map(int, sys.argv[1:])
    except ValueError:
        sys.exit("wins, losses and draws must be integers")
    if min(wins, losses, draws) < 0:
        sys.exit("wins, losses and draws must be >= 0")
    games = wins + losses + draws
    if games <= 0:
        sys.exit("no games")

    score = (wins + 0.5 * draws) / games
    variance = (wins * (1 - score) ** 2 + losses * score**2 + draws * (0.5 - score) ** 2) / games
    margin = 1.96 * math.sqrt(variance / games)
    low, high = elo(score - margin), elo(score + margin)
    half = math.inf if math.isinf(low) and low == high else (high - low) / 2
    decisive = wins + losses
    los = 0.5 * (1 + math.erf((wins - losses) / math.sqrt(2 * decisive))) if decisive else 0.5

    print(f"\nGame Results: {wins} wins, {losses} losses, {draws} draws")
    print(f"Total Games: {games}")
    print(f"Performance Ratio: {100 * score:.2f}%")
    print(f"ELO Rating: {elo(score):.2f}")
    print(f"95% Confidence Interval: ±{half:.2f}")
    print(f"ELO Rating Range: [{low:.2f}, {high:.2f}]")
    print(f"LOS: {100 * los:.2f}%")


if __name__ == "__main__":
    main()
