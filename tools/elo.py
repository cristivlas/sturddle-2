import math
import sys

def calculate_elo_rating(wins, losses, draws):
    """
    Calculate ELO rating and confidence margin from game results.

    Args:
    losses (int): Number of losses
    wins (int): Number of wins
    draws (int): Number of draws

    Returns:
    dict: A dictionary containing ELO rating calculations
    """
    # Calculate total games and performance ratio
    total_games = losses + wins + draws
    performance_ratio = (wins + 0.5 * draws) / total_games

    # Calculate ELO rating
    elo_rating = 400 * math.log10(performance_ratio / (1 - performance_ratio))

    # Calculate standard error and confidence interval
    standard_error = math.sqrt((performance_ratio * (1 - performance_ratio)) / total_games)
    confidence_interval = 1.96 * standard_error * 400

    return {
        'total_games': total_games,
        'performance_ratio': performance_ratio * 100,
        'elo_rating': round(elo_rating, 2),
        'confidence_interval': round(confidence_interval, 2),
        'elo_range_low': round(elo_rating - confidence_interval, 2),
        'elo_range_high': round(elo_rating + confidence_interval, 2)
    }

def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python elo_calculator.py <wins> <losses> <draws>")
        sys.exit(1)

    try:
        # Convert command-line arguments to integers
        wins = int(sys.argv[1])
        losses = int(sys.argv[2])
        draws = int(sys.argv[3])
    except ValueError:
        print("Error: Arguments must be integers")
        sys.exit(1)

    # Calculate and display results
    result = calculate_elo_rating(wins, losses, draws)

    print(f"\nGame Results: {wins} wins, {losses} losses, {draws} draws")
    print(f"Total Games: {result['total_games']}")
    print(f"Performance Ratio: {result['performance_ratio']:.2f}%")
    print(f"ELO Rating: {result['elo_rating']}")
    print(f"95% Confidence Interval: ±{result['confidence_interval']}")
    print(f"ELO Rating Range: [{result['elo_range_low']}, {result['elo_range_high']}]")

# Allow running as a script
if __name__ == "__main__":
    main()