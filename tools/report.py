#!/usr/bin/env python3
"""
PGN Tournament Report Generator
Generates cutechess-cli style tournament statistics from PGN files
"""

import chess.pgn
import sys
import math
from collections import defaultdict
# from typing import Dict, Tuple
from scipy.stats import norm

def calculate_elo_rating(wins, losses, draws):
    """
    Calculate ELO rating and confidence margin from game results.

    Args:
    wins (int): Number of wins
    losses (int): Number of losses
    draws (int): Number of draws

    Returns:
    dict: A dictionary containing ELO rating calculations
    """
    # Calculate total games and performance ratio
    total_games = losses + wins + draws

    if total_games == 0:
        return None

    performance_ratio = (wins + 0.5 * draws) / total_games

    # Avoid division by zero or log of zero
    if performance_ratio == 0 or performance_ratio == 1:
        return {
            'total_games': total_games,
            'performance_ratio': performance_ratio * 100,
            'elo_rating': None,
            'confidence_interval': None,
            'elo_range_low': None,
            'elo_range_high': None,
            'los': 100.0 if performance_ratio == 1 else 0.0
        }

    elo_rating = 400 * math.log10(performance_ratio / (1 - performance_ratio))
    standard_error = math.sqrt((performance_ratio * (1 - performance_ratio)) / total_games)
    confidence_interval = 1.96 * standard_error * 400

    # LOS calculation using cumulative normal distribution
    los = norm.cdf(elo_rating / (standard_error * 400)) * 100  # LOS as percentage

    return {
        'total_games': total_games,
        'performance_ratio': performance_ratio * 100,
        'elo_rating': round(elo_rating, 2),
        'confidence_interval': round(confidence_interval, 2),
        'elo_range_low': round(elo_rating - confidence_interval, 2),
        'elo_range_high': round(elo_rating + confidence_interval, 2),
        'los': round(los, 2)
    }


class PlayerStats:
    def __init__(self):
        self.total_games = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.white_games = 0
        self.white_wins = 0
        self.white_draws = 0
        self.white_losses = 0
        self.black_games = 0
        self.black_wins = 0
        self.black_draws = 0
        self.black_losses = 0

    def add_game(self, color, result):
        """Add a game result for a player"""
        self.total_games += 1

        if color == 'white':
            self.white_games += 1
            if result == 1:  # win
                self.wins += 1
                self.white_wins += 1
            elif result == 0.5:  # draw
                self.draws += 1
                self.white_draws += 1
            else:  # loss
                self.losses += 1
                self.white_losses += 1
        else:  # black
            self.black_games += 1
            if result == 1:  # win
                self.wins += 1
                self.black_wins += 1
            elif result == 0.5:  # draw
                self.draws += 1
                self.black_draws += 1
            else:  # loss
                self.losses += 1
                self.black_losses += 1

    def score(self):
        return self.wins + 0.5 * self.draws

    def white_score(self):
        return self.white_wins + 0.5 * self.white_draws

    def black_score(self):
        return self.black_wins + 0.5 * self.black_draws

    def win_rate(self):
        if self.total_games == 0:
            return 0.0
        return (self.wins / self.total_games) * 100

    def draw_rate(self):
        if self.total_games == 0:
            return 0.0
        return (self.draws / self.total_games) * 100

def parse_pgn(filename):
    """Parse PGN file and collect statistics"""
    stats = defaultdict(PlayerStats)
    total_games = 0

    with open(filename) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            white = game.headers.get("White", "Unknown")
            black = game.headers.get("Black", "Unknown")
            result = game.headers.get("Result", "*")

            if result == "*":
                continue  # Skip unfinished games

            total_games += 1

            # Parse result
            if result == "1-0":
                white_result = 1
                black_result = 0
            elif result == "0-1":
                white_result = 0
                black_result = 1
            elif result == "1/2-1/2":
                white_result = 0.5
                black_result = 0.5
            else:
                continue  # Skip unknown results

            stats[white].add_game('white', white_result)
            stats[black].add_game('black', black_result)

    return stats, total_games

def print_report(stats, total_games):
    """Print tournament report similar to cutechess-cli"""

    # Sort players by score
    sorted_players = sorted(stats.items(), key=lambda x: x[1].score(), reverse=True)

    print("=" * 100)
    print("TOURNAMENT REPORT")
    print("=" * 100)
    print(f"\nTotal games: {total_games}\n")

    # Overall standings
    print("-" * 100)
    print(f"{'Rank':<6} {'Player':<25} {'Score':<10} {'Games':<8} {'W':<6} {'D':<6} {'L':<6} {'Win%':<8} {'Draw%':<8}")
    print("-" * 100)

    for rank, (player, stat) in enumerate(sorted_players, 1):
        print(f"{rank:<6} {player:<25} {stat.score():<10.1f} {stat.total_games:<8} "
              f"{stat.wins:<6} {stat.draws:<6} {stat.losses:<6} "
              f"{stat.win_rate():<8.1f} {stat.draw_rate():<8.1f}")

    print("-" * 100)

    # Head-to-head Elo if exactly 2 players
    if len(sorted_players) == 2:
        player1, stats1 = sorted_players[0]
        player2, stats2 = sorted_players[1]

        elo_data = calculate_elo_rating(stats1.wins, stats1.losses, stats1.draws)

        if elo_data and elo_data['elo_rating'] is not None:
            print(f"\n{'HEAD-TO-HEAD ELO DIFFERENCE':^100}")
            print("-" * 100)
            print(f"{player1} vs {player2}:")
            print(f"  Elo difference: {elo_data['elo_rating']:+.2f} ± {elo_data['confidence_interval']:.2f}")
            print(f"  95% confidence: [{elo_data['elo_range_low']:+.2f}, {elo_data['elo_range_high']:+.2f}]")
            print(f"  LOS (Likelihood of Superiority): {elo_data['los']:.2f}%")
            print("-" * 100)

    # White vs Black breakdown
    print("\n" + "=" * 100)
    print("WHITE vs BLACK STATISTICS")
    print("=" * 100)

    for rank, (player, stat) in enumerate(sorted_players, 1):
        print(f"\n{rank}. {player}")
        print(f"   Overall: {stat.score():.1f}/{stat.total_games} ({stat.win_rate():.1f}% wins, {stat.draw_rate():.1f}% draws)")

        if stat.white_games > 0:
            white_perf = (stat.white_score() / stat.white_games) * 100
            print(f"   As White: {stat.white_score():.1f}/{stat.white_games} "
                  f"(+{stat.white_wins} ={stat.white_draws} -{stat.white_losses}) "
                  f"[{white_perf:.1f}%]")

            # Calculate Elo for white performance
            white_elo = calculate_elo_rating(stat.white_wins, stat.white_losses, stat.white_draws)
            if white_elo and white_elo['elo_rating'] is not None:
                print(f"      Elo: {white_elo['elo_rating']:+.2f} ± {white_elo['confidence_interval']:.2f} "
                      f"(LOS: {white_elo['los']:.1f}%)")

        if stat.black_games > 0:
            black_perf = (stat.black_score() / stat.black_games) * 100
            print(f"   As Black: {stat.black_score():.1f}/{stat.black_games} "
                  f"(+{stat.black_wins} ={stat.black_draws} -{stat.black_losses}) "
                  f"[{black_perf:.1f}%]")

            # Calculate Elo for black performance
            black_elo = calculate_elo_rating(stat.black_wins, stat.black_losses, stat.black_draws)
            if black_elo and black_elo['elo_rating'] is not None:
                print(f"      Elo: {black_elo['elo_rating']:+.2f} ± {black_elo['confidence_interval']:.2f} "
                      f"(LOS: {black_elo['los']:.1f}%)")

    print("\n" + "=" * 100)

    # Summary statistics
    total_white_score = sum(s.white_score() for s in stats.values())
    total_white_games = sum(s.white_games for s in stats.values())
    total_black_score = sum(s.black_score() for s in stats.values())
    total_black_games = sum(s.black_games for s in stats.values())

    if total_white_games > 0:
        white_avg = (total_white_score / total_white_games) * 100
        print(f"\nWhite overall performance: {white_avg:.2f}%")

    if total_black_games > 0:
        black_avg = (total_black_score / total_black_games) * 100
        print(f"Black overall performance: {black_avg:.2f}%")

    print("=" * 100)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pgn_tournament_report.py <pgn_file>")
        print("\nThis script generates a tournament report with white vs black statistics")
        print("from a PGN file, similar to cutechess-cli output.")
        sys.exit(1)

    pgn_file = sys.argv[1]

    try:
        stats, total_games = parse_pgn(pgn_file)
        print_report(stats, total_games)
    except FileNotFoundError:
        print(f"Error: File '{pgn_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing PGN file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
