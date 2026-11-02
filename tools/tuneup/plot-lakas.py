#!/usr/bin/env python3
import argparse
import ast
import itertools
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass

BEST_PARAM_REGEX = re.compile(r".*\|\s*INFO\s*\|\s*best param:\s*({.*})")
BUDGET_REGEX = re.compile(r".*\|\s*INFO\s*\|\s*budget:\s*(\d+)")
LOSS_REGEX = re.compile(r".*\|\s*INFO\s*\|\s*actual result:.*minimized result or loss:\s*([\d\.]+)")

DEFAULT_ALPHAS = [0.3, 0.1, 0.05]


def scan_for_params(path, sample_size=100):
    names = set()
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= sample_size and names:
                break
            if m := BEST_PARAM_REGEX.match(line):
                names.update(ast.literal_eval(m[1]))
    return sorted(names)


def parse_log_file(args):
    params = defaultdict(list)
    budgets, losses, loss_budgets = [], [], []
    with open(args.logfile) as f:
        lines = f.readlines()

    budget = None
    for i, line in enumerate(lines):
        if m := BUDGET_REGEX.match(line):
            budget = int(m[1])
        m = BEST_PARAM_REGEX.match(line)
        if not m or budget is None:
            continue
        budgets.append(budget)
        for name, value in ast.literal_eval(m[1]).items():
            params[name].append(value)
        if args.loss:
            for later in lines[i + 1 : i + 10]:  # the loss follows within a few lines
                if loss := LOSS_REGEX.match(later):
                    losses.append(float(loss[1]))
                    loss_budgets.append(budget)
                    break
    return budgets, params, losses, loss_budgets


def ema(values, alpha):
    out = values[:1]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def plot_parameters(args, budgets, params, selected, losses, loss_budgets):
    losses = losses[args.skip_loss :]
    loss_budgets = loss_budgets[args.skip_loss :]
    if losses:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    else:
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = None
    fig.canvas.manager.set_window_title("Parameter Optimization")

    colors = plt.cm.tab20c(np.linspace(0, 1, len(selected)))
    styles = [("-", 1), ("--", 1.5), ("-.", 2.5)]
    mark_every = max(1, len(budgets) // 20)
    for idx, (name, (linestyle, width)) in enumerate(zip(selected, itertools.cycle(styles))):
        ax1.plot(
            budgets,
            params[name],
            label=name,
            color=colors[idx // len(styles)],
            linestyle=linestyle,
            linewidth=width,
            marker="s",
            markerfacecolor=colors[idx],
            markevery=mark_every,
        )
    ax1.set_xlabel("" if ax2 is not None else "Iteration")
    ax1.set_ylabel("Parameter Value")
    ax1.set_title("Best Parameter Values over Iterations")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.minorticks_on()
    ax1.legend(loc="upper left", ncol=args.legend_ncols, fontsize="small" if len(selected) > 50 else "medium")

    if ax2 is not None:
        ax2.plot(loss_budgets, losses, "r-", alpha=0.15, linewidth=0.8, label="Raw Loss")
        for alpha, color in zip(args.alpha, itertools.cycle(["darkred", "lightsalmon", "orangered"])):
            ax2.plot(loss_budgets, ema(losses, alpha), color=color, linewidth=1.5, label=f"EMA (α={alpha})")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss Value and EMAs")
        ax2.set_title("Loss Values and Exponential Moving Averages")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")

    plt.tight_layout()
    if ax2 is not None:
        plt.subplots_adjust(top=0.92)
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {args.output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot best parameters and minimized loss with EMAs from a Lakas log")
    parser.add_argument("logfile")
    parser.add_argument("params", nargs="*", help="parameters to plot (default: all)")
    parser.add_argument("--all", action="store_true", help="plot all parameters")
    parser.add_argument(
        "--alpha", "-a", type=float, nargs="+", default=DEFAULT_ALPHAS, help="EMA smoothing factor(s), implies --loss"
    )
    parser.add_argument("--list", action="store_true", help="list the parameters in the log and exit")
    parser.add_argument("--loss", action="store_true", help="plot loss values and their EMAs")
    parser.add_argument("--legend-ncols", "-n", type=int, default=1)
    parser.add_argument("--output", "-o", help="also save the plot to this image file")
    parser.add_argument("--skip-loss", "-s", type=int, default=0, help="loss entries to skip")
    args = parser.parse_args()

    if args.list:
        print("Available parameters in the log file:")
        for i, name in enumerate(scan_for_params(args.logfile), 1):
            print(f"{i:2d}. {name}")
        return

    if args.alpha != DEFAULT_ALPHAS:
        args.loss = True
    if not all(0 < a < 1 for a in args.alpha):
        print("Warning: alpha values must be between 0 and 1; using the defaults.")
        args.alpha = DEFAULT_ALPHAS

    budgets, params, losses, loss_budgets = parse_log_file(args)
    if not budgets:
        print("Error: no 'best param:' entries in the log file")
        return

    if args.params and not args.all:
        for name in args.params:
            if name not in params:
                print(f"Warning: Parameter '{name}' not found in log file")
        selected = [name for name in args.params if name in params]
    else:
        if not args.all:
            print("No parameters specified. Defaulting to --all.")
        selected = list(params)

    if not selected:
        print("Error: none of the requested parameters are in the log file")
        print(f"Available parameters: {', '.join(params)}")
        return
    plot_parameters(args, budgets, params, selected, losses, loss_budgets)


if __name__ == "__main__":
    main()
