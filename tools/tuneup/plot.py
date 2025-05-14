#!/usr/bin/env python3

import argparse
import itertools
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict

try:
    matplotlib.use('TkAgg')
except:
    pass


BEST_PARAM_REGEX = re.compile(r'.*\|\s*INFO\s*\|\s*best param:\s*({.*})')
BUDGET_REGEX = re.compile(r'.*\|\s*INFO\s*\|\s*budget:\s*(\d+)')
LOSS_REGEX = re.compile(r'.*\|\s*INFO\s*\|\s*actual result:.*minimized result or loss:\s*([\d\.]+)')

"""Default smoothening for EMA"""
DEFAULT_ALPHAS = [0.3, 0.1, 0.05]


def scan_for_params(filename, sample_size=100):
    """Scan the first part of the log file to detect available parameter names."""
    param_names = set()

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size and param_names:
                break

            best_match = BEST_PARAM_REGEX.match(line)
            if best_match:
                param_str = best_match.group(1)
                params = eval(param_str)
                param_names.update(params.keys())

    return sorted(list(param_names))


def parse_log_file(args):
    """Parse the log file and extract best parameter values with budget numbers and minimized loss values."""
    param_data = defaultdict(list)
    budget_numbers = []
    loss_values = []
    loss_budgets = []  # Track budgets where we have loss values

    with open(args.logfile, 'r') as f:
        lines = f.readlines()

    current_budget = None

    for i, line in enumerate(lines):
        # Extract budget number
        budget_match = BUDGET_REGEX.match(line)
        if budget_match:
            current_budget = int(budget_match.group(1))

        # Extract best parameters
        best_match = BEST_PARAM_REGEX.match(line)
        if best_match and current_budget is not None:
            # Extract dictionary string
            param_str = best_match.group(1)
            # Convert string representation to dictionary
            params = eval(param_str)

            budget_numbers.append(current_budget)

            for param_name, value in params.items():
                param_data[param_name].append(value)

            if args.loss:
                # Look for "minimized result or loss" in the next few lines
                for j in range(i+1, min(i+10, len(lines))):
                    min_loss_match = LOSS_REGEX.match(lines[j])
                    if min_loss_match:
                        loss_values.append(float(min_loss_match.group(1)))
                        loss_budgets.append(current_budget)
                        break

    return budget_numbers, param_data, loss_values, loss_budgets


def calculate_ema(loss_values, alphas):
    """Calculate Exponential Moving Averages with different smoothing factors.
    https://en.wikipedia.org/wiki/Moving_average
    """
    ema_results = {}

    for alpha in alphas:
        ema = []
        if loss_values:
            ema_value = loss_values[0]  # Initialize with first value
            ema.append(ema_value)

            # Calculate EMA for the rest of the values
            for i in range(1, len(loss_values)):
                ema_value = alpha * loss_values[i] + (1 - alpha) * ema_value
                ema.append(ema_value)

        ema_results[alpha] = ema

    return ema_results


def plot_parameters(args, budget_numbers, param_data, selected_params, loss_values, loss_budgets):
    """Plot parameters and EMAs of loss values."""

    assert len(loss_values) == len(loss_budgets), "Bad data?"

    plot_loss = len(loss_values) > args.skip_loss
    loss_values = loss_values[args.skip_loss:]
    loss_budgets = loss_budgets[args.skip_loss:]

    # Create two separate subplots for cleaner visualization
    if plot_loss:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [1, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = None

    plt.get_current_fig_manager().set_window_title('Parameter Optimization')

    # Plot parameters on the top subplot or main plot
    cmap = plt.cm.tab20c
    indices = np.linspace(0, 1, len(selected_params))
    colors = [cmap(i) for i in indices]

    styles = [('-', 1), ('--', 1.5), ('-.', 2.5)]
    line_styles = itertools.cycle(styles)

    mark_spacing = max(1, len(budget_numbers) // 20)

    for idx, param_name in enumerate(selected_params):
        if param_name in param_data:
            values = param_data[param_name]
            style = next(line_styles)
            ax1.plot(budget_numbers, values,
                    label=param_name,
                    color=colors[idx // len(styles)],
                    linestyle=style[0],
                    linewidth=style[1],
                    marker='s',
                    markerfacecolor=colors[idx],
                    markevery=mark_spacing)

    ax1.set_xlabel('Iteration' if ax2 is None else '')
    ax1.set_ylabel('Parameter Value')
    ax1.set_title('Best Parameter Values over Iterations')
    ax1.grid(True, alpha=0.3)

    # Configure parameter legend
    fontsize = 'small' if len(selected_params) > 50 else 'medium'
    ax1.legend(loc='upper left', ncol=args.legend_ncols, fontsize=fontsize)

    # If loss values are provided, calculate EMAs and plot them on the bottom subplot
    if plot_loss:
        plot_ax = ax2 if ax2 is not None else ax1

        # Calculate EMAs
        ema_results = calculate_ema(loss_values, args.alpha)

        # Plot raw loss values with low opacity
        plot_ax.plot(loss_budgets, loss_values, 'r-', alpha=0.15, linewidth=0.8, label='Raw Loss')

        # Plot EMAs with different colors
        ema_colors = ['darkred', 'lightsalmon', 'orangered']
        for i, alpha in enumerate(args.alpha):
            if len(ema_results[alpha]) > 0:
                plot_ax.plot(loss_budgets, ema_results[alpha],
                        color=ema_colors[i % len(ema_colors)],
                        linewidth=1.5,
                        label=f'EMA (α={alpha})')

        plot_ax.set_xlabel('Iteration')
        plot_ax.set_ylabel('Loss Value and EMAs')
        plot_ax.set_title('Loss Values and Exponential Moving Averages')
        plot_ax.grid(True, alpha=0.3)
        plot_ax.legend(loc='upper left')

    plt.tight_layout()
    if ax2 is not None:
        plt.subplots_adjust(top=0.92)

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {args.output}')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot best parameters and minimized loss with EMAs from optimization log files')
    parser.add_argument('logfile', help='Path to the log file')
    parser.add_argument('params', nargs='*', help='Names of parameters to plot (optional)')
    parser.add_argument('--all', action='store_true', help='Plot all available parameters')
    parser.add_argument('--alpha', '-a', type=float, nargs='+', default=DEFAULT_ALPHAS, help='EMA smoothing factor(s) (between 0 and 1)')
    parser.add_argument('--list', action='store_true', help='List available parameters and exit')
    parser.add_argument('--loss', action='store_true', help='Plot loss values and EMA (exponential moving average)')
    parser.add_argument('--legend-ncols', '-n', type=int, default=1)
    parser.add_argument('--output', '-o', help='Output image filename')
    parser.add_argument('--skip-loss', '-s', type=int, default=0, help='Number of loss entries to skip')

    args = parser.parse_args()

    # If --list is specified, scan for parameters and display them
    if args.list:
        available_params = scan_for_params(args.logfile)
        print("Available parameters in the log file:")
        for i, param in enumerate(available_params, 1):
            print(f"{i:2d}. {param}")
        return

    if args.alpha != DEFAULT_ALPHAS:
        args.loss = True  # if user specified alphas, implicitly plot loss

    # Parse the log file
    budget_numbers, param_data, loss_values, loss_budgets = parse_log_file(args)

    if not budget_numbers:
        print("Error: No data found in the log file!")
        print("Make sure the log file contains 'best param:' entries.")
        return

    if args.loss and not all(0 < alpha < 1 for alpha in args.alpha):
        print("Warning: Alpha values should be between 0 and 1. Using default values.")
        args.alpha = DEFAULT_ALPHAS

    # Determine which parameters to plot
    if args.all:
        selected_params = list(param_data.keys())
    elif args.params:
        selected_params = []
        for param in args.params:
            if param in param_data:
                selected_params.append(param)
            else:
                print(f"Warning: Parameter '{param}' not found in log file")
    else:
        print("No parameters specified. Defaulting to --all.")
        selected_params = list(param_data.keys())

    if selected_params:
        plot_parameters(args, budget_numbers, param_data, selected_params, loss_values, loss_budgets)
    else:
        print("Error: None of the requested parameters were found in the log file")
        print(f"Available parameters: {', '.join(param_data.keys())}")


if __name__ == '__main__':
    main()
