#!/usr/bin/env python3
"""
PGN Tournament Report Generator
Generates cutechess-cli style tournament statistics from PGN files
"""

import chess.pgn
import sys
import math
import argparse
from collections import defaultdict
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


class OpeningStats:
    """Track statistics for a specific opening"""
    def __init__(self):
        self.white_wins = 0
        self.white_draws = 0
        self.white_losses = 0
        self.black_wins = 0
        self.black_draws = 0
        self.black_losses = 0

    def add_game(self, color, result):
        if color == 'white':
            if result == 1:
                self.white_wins += 1
            elif result == 0.5:
                self.white_draws += 1
            else:
                self.white_losses += 1
        else:  # black
            if result == 1:
                self.black_wins += 1
            elif result == 0.5:
                self.black_draws += 1
            else:
                self.black_losses += 1

    def white_games(self):
        return self.white_wins + self.white_draws + self.white_losses

    def black_games(self):
        return self.black_wins + self.black_draws + self.black_losses

    def total_games(self):
        return self.white_games() + self.black_games()

    def white_score(self):
        return self.white_wins + 0.5 * self.white_draws

    def black_score(self):
        return self.black_wins + 0.5 * self.black_draws

    def white_performance(self):
        games = self.white_games()
        return (self.white_score() / games * 100) if games > 0 else 0

    def black_performance(self):
        games = self.black_games()
        return (self.black_score() / games * 100) if games > 0 else 0


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
        self.opening_stats = defaultdict(OpeningStats)

    def add_game(self, color, result, opening=None):
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

        # Track opening stats
        if opening:
            self.opening_stats[opening].add_game(color, result)

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
            opening = game.headers.get("Opening", "Unknown Opening")

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

            stats[white].add_game('white', white_result, opening)
            stats[black].add_game('black', black_result, opening)

    return stats, total_games


def print_report(stats, total_games):
    """Print tournament report similar to cutechess-cli"""

    # Sort players by score
    sorted_players = sorted(stats.items(), key=lambda x: x[1].score(), reverse=True)

    print(f"\nTotal games: {total_games}")

    print(f"\n{'Rank':<6} {'Player':<25} {'Score':<10} {'Games':<8} {'W':<6} {'D':<6} {'L':<6} {'Win%':<8} {'Draw%':<8}")

    for rank, (player, stat) in enumerate(sorted_players, 1):
        print(f"{rank:<6} {player:<25} {stat.score():<10.1f} {stat.total_games:<8} "
              f"{stat.wins:<6} {stat.draws:<6} {stat.losses:<6} "
              f"{stat.win_rate():<8.1f} {stat.draw_rate():<8.1f}")

    # Head-to-head Elo if exactly 2 players
    if len(sorted_players) == 2:
        player1, stats1 = sorted_players[0]
        player2, stats2 = sorted_players[1]

        elo_data = calculate_elo_rating(stats1.wins, stats1.losses, stats1.draws)

        if elo_data and elo_data['elo_rating'] is not None:
            print(f"\nHEAD-TO-HEAD\n")
            print(f"{player1} vs {player2}:")
            print(f"  Elo difference: {elo_data['elo_rating']:+.2f} +/- {elo_data['confidence_interval']:.2f}")
            print(f"  95% confidence: [{elo_data['elo_range_low']:+.2f}, {elo_data['elo_range_high']:+.2f}]")
            print(f"  LOS: {elo_data['los']:.2f}%")

    # White vs Black breakdown
    print("\nWHITE vs BLACK")

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
                print(f"      Elo: {white_elo['elo_rating']:+.2f} +/- {white_elo['confidence_interval']:.2f} "
                      f"(LOS: {white_elo['los']:.1f}%)")

        if stat.black_games > 0:
            black_perf = (stat.black_score() / stat.black_games) * 100
            print(f"   As Black: {stat.black_score():.1f}/{stat.black_games} "
                  f"(+{stat.black_wins} ={stat.black_draws} -{stat.black_losses}) "
                  f"[{black_perf:.1f}%]")

            # Calculate Elo for black performance
            black_elo = calculate_elo_rating(stat.black_wins, stat.black_losses, stat.black_draws)
            if black_elo and black_elo['elo_rating'] is not None:
                print(f"      Elo: {black_elo['elo_rating']:+.2f} +/- {black_elo['confidence_interval']:.2f} "
                      f"(LOS: {black_elo['los']:.1f}%)")

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


def print_player_details(player_name, stats, sort_by='combined'):
    """Print detailed opening breakdown for a specific player"""
    if player_name not in stats:
        print(f"Error: Player '{player_name}' not found in tournament")
        available = ', '.join(stats.keys())
        print(f"Available players: {available}")
        return

    player_stats = stats[player_name]

    print(f"\nDETAILED REPORT FOR: {player_name}")
    print(f"\nOverall Performance: {player_stats.score():.1f}/{player_stats.total_games} "
          f"({player_stats.win_rate():.1f}% wins, {player_stats.draw_rate():.1f}% draws)")

    if not player_stats.opening_stats:
        print("\nNo opening information available in PGN file.")
        return

    # Calculate ELO for each opening based on sort preference
    openings_with_elo = []
    for opening, ostats in player_stats.opening_stats.items():
        if sort_by == 'combined':
            # Calculate combined ELO across white and black games
            total_wins = ostats.white_wins + ostats.black_wins
            total_draws = ostats.white_draws + ostats.black_draws
            total_losses = ostats.white_losses + ostats.black_losses
            elo_data = calculate_elo_rating(total_wins, total_losses, total_draws)
        elif sort_by == 'white':
            # Calculate white-only ELO
            elo_data = calculate_elo_rating(ostats.white_wins, ostats.white_losses, ostats.white_draws)
        else:  # black
            # Calculate black-only ELO
            elo_data = calculate_elo_rating(ostats.black_wins, ostats.black_losses, ostats.black_draws)

        # Use infinity for perfect scores so they sort to the top
        elo_value = elo_data['elo_rating'] if elo_data and elo_data['elo_rating'] is not None else float('inf')

        openings_with_elo.append((opening, ostats, elo_value))

    # Sort by ELO (highest first)
    sorted_openings = sorted(openings_with_elo, key=lambda x: x[2], reverse=True)

    sort_description = {
        'combined': 'combined ELO',
        'white': 'white-side ELO',
        'black': 'black-side ELO'
    }
    print(f"\nPERFORMANCE BY OPENING ({len(sorted_openings)} unique openings, sorted by {sort_description[sort_by]})")

    for opening, ostats, sort_elo_value in sorted_openings:
        total = ostats.total_games()

        # Calculate combined ELO for display
        total_wins = ostats.white_wins + ostats.black_wins
        total_draws = ostats.white_draws + ostats.black_draws
        total_losses = ostats.white_losses + ostats.black_losses
        combined_elo = calculate_elo_rating(total_wins, total_losses, total_draws)
        combined_elo_value = combined_elo['elo_rating'] if combined_elo and combined_elo['elo_rating'] is not None else None

        # Show the sorting ELO in the header
        if sort_elo_value != float('inf'):
            sort_label = {
                'combined': 'Combined',
                'white': 'White',
                'black': 'Black'
            }[sort_by]
            print(f"\n{opening} ({total} games) - {sort_label} ELO: {sort_elo_value:+.2f}")
        else:
            print(f"\n{opening} ({total} games) - ELO: N/A (perfect score)")

        # White performance
        if ostats.white_games() > 0:
            w_games = ostats.white_games()
            w_score = ostats.white_score()
            w_perf = ostats.white_performance()
            print(f"  As White: {w_score:.1f}/{w_games} "
                  f"(+{ostats.white_wins} ={ostats.white_draws} -{ostats.white_losses}) "
                  f"[{w_perf:.1f}%]")

            # Elo for this opening as white
            w_elo = calculate_elo_rating(ostats.white_wins, ostats.white_losses, ostats.white_draws)
            if w_elo and w_elo['elo_rating'] is not None:
                print(f"     Elo: {w_elo['elo_rating']:+.2f} +/- {w_elo['confidence_interval']:.2f} "
                      f"(LOS: {w_elo['los']:.1f}%)")

        # Black performance
        if ostats.black_games() > 0:
            b_games = ostats.black_games()
            b_score = ostats.black_score()
            b_perf = ostats.black_performance()
            print(f"  As Black: {b_score:.1f}/{b_games} "
                  f"(+{ostats.black_wins} ={ostats.black_draws} -{ostats.black_losses}) "
                  f"[{b_perf:.1f}%]")

            # Elo for this opening as black
            b_elo = calculate_elo_rating(ostats.black_wins, ostats.black_losses, ostats.black_draws)
            if b_elo and b_elo['elo_rating'] is not None:
                print(f"     Elo: {b_elo['elo_rating']:+.2f} +/- {b_elo['confidence_interval']:.2f} "
                      f"(LOS: {b_elo['los']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate tournament report with white vs black statistics from PGN files'
    )
    parser.add_argument('pgn_file', help='Path to PGN file')
    parser.add_argument('--details', metavar='PLAYER',
                        help='Show detailed opening breakdown for specific player')
    parser.add_argument('--sort', choices=['combined', 'white', 'black'], default='combined',
                        help='Sort openings by: combined ELO (default), white-side ELO, or black-side ELO')

    args = parser.parse_args()

    try:
        stats, total_games = parse_pgn(args.pgn_file)

        if args.details:
            print_player_details(args.details, stats, args.sort)
        else:
            print_report(stats, total_games)

    except FileNotFoundError:
        print(f"Error: File '{args.pgn_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing PGN file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
