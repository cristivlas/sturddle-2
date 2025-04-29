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


def scan_for_params(filename, sample_size=100):
    """Scan the first part of the lakas.py log file to detect available parameter names."""
    param_names = set()
    found_param_line = False

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size and found_param_line:
                break

            best_match = re.match(r'.*\|\s*INFO\s*\|\s*best param:\s*({.*})', line)
            if best_match:
                try:
                    param_str = best_match.group(1)
                    params = eval(param_str)
                    param_names.update(params.keys())
                    found_param_line = True
                except Exception as e:
                    print(f"Warning: Could not parse parameters on line {i+1}: {str(e)}")

    return sorted(list(param_names))


def parse_log_file(filename):
    """Parse the lakas.py log file and extract best parameter values with budget numbers."""
    param_data = defaultdict(list)
    budget_numbers = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    current_budget = None

    for i, line in enumerate(lines):
        # Extract budget number
        budget_match = re.match(r'.*\|\s*INFO\s*\|\s*budget:\s*(\d+)', line)
        if budget_match:
            current_budget = int(budget_match.group(1))

        # Extract best parameters
        best_match = re.match(r'.*\|\s*INFO\s*\|\s*best param:\s*({.*})', line)
        if best_match and current_budget is not None:
            try:
                # Extract dictionary string
                param_str = best_match.group(1)
                # Convert string representation to dictionary
                params = eval(param_str)

                budget_numbers.append(current_budget)

                for param_name, value in params.items():
                    param_data[param_name].append(value)

            except Exception as e:
                print(f"Warning: Could not parse parameters on line {i+1}: {str(e)}")

    return budget_numbers, param_data


def plot_parameters(budget_numbers, param_data, selected_params):
    _fig, _ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.tab20c
    indices = np.linspace(0, 1, len(selected_params))
    #np.random.shuffle(indices)  # Randomize to break adjacency similarity
    colors = [cmap(i) for i in indices]

    styles = [('-', 1), ('--', 1.5), ('-.', 2.5)]
    line_styles = itertools.cycle(styles)

    mark_spacing = max(1, len(budget_numbers) // 20)

    for idx, param_name in enumerate(selected_params):
        if param_name in param_data:
            values = param_data[param_name]
            # print(f'Plotting {param_name}...')
            style = next(line_styles)
            plt.plot(budget_numbers, values,
                     label=param_name,
                     color=colors[idx // len(styles)],
                     linestyle=style[0],
                     linewidth=style[1],
                     marker='s',
                     markerfacecolor=colors[idx],
                     markevery=mark_spacing)

    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Best Parameter Values over Iterations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('parameter_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to: parameter_plot.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot best parameters from lakas.py optimization log files')
    parser.add_argument('logfile', help='Path to the lakas.py log file')
    parser.add_argument('params', nargs='*', help='Names of parameters to plot (optional)')
    parser.add_argument('--list', action='store_true', help='List available parameters and exit')
    parser.add_argument('--all', action='store_true', help='Plot all available parameters')

    args = parser.parse_args()

    # If --list is specified, scan for parameters and display them
    if args.list:
        available_params = scan_for_params(args.logfile)
        print("Available parameters in the lakas.py log file:")
        for i, param in enumerate(available_params, 1):
            print(f"{i:2d}. {param}")
        return

    # Parse the log file
    budget_numbers, param_data = parse_log_file(args.logfile)

    if not budget_numbers:
        print("Error: No data found in the lakas.py log file!")
        print("Make sure the log file contains 'best param:' entries.")
        return

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
        print("Error: No parameters specified. Use --list to see available parameters.")
        print("       Use --all to plot all parameters, or specify parameters to plot.")
        return

    if not selected_params:
        print("Error: None of the requested parameters were found in the log file")
        print(f"Available parameters: {', '.join(param_data.keys())}")
        return

    # Print summary statistics
    print("\nBest Parameter Summary:")
    for param in selected_params:
        values = param_data[param]
        print(f"\n{param}:")
        print(f"  Initial value: {values[0]:.6f}")
        print(f"  Final value: {values[-1]:.6f}")
        print(f"  Min value: {min(values):.6f}")
        print(f"  Max value: {max(values):.6f}")
        print(f"  Average value: {np.mean(values):.6f}")
        print(f"  Standard deviation: {np.std(values):.6f}")
        print(f"  Change: {values[-1] - values[0]:.6f}")

    # Plot the parameters
    plot_parameters(budget_numbers, param_data, selected_params)


if __name__ == '__main__':
    main()
