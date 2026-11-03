#!/usr/bin/env python3
import ast
import re
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")

# log key, title, y label, line style
METRICS = (
    ("loss", "Loss", "Loss", "b-"),
    ("mae", "Eval MAE", "MAE", "r-"),
    ("accuracy", "Outcome Accuracy", "Accuracy", "g-"),
)


def extract_metrics(path):
    values = {key: [] for key, *_ in METRICS}
    hyperparams = []
    with open(path) as f:
        for line in f:
            if m := re.search(r"epoch=\d+ loss=([\d.]+) hyperparam=(\{.*\})", line):
                values["loss"].append(float(m[1]))
                hyperparams.append(m[2])
            for key in ("mae", "accuracy"):
                if m := re.search(rf"epoch=\d+ .*{key}=([\d.]+)", line):
                    values[key].append(float(m[1]))
    return values, hyperparams


def plot_metrics(values, hyperparams):
    shown = [metric for metric in METRICS if values[metric[0]]]
    if not shown:
        print("No loss, MAE, or accuracy data found in file")
        return

    fig = plt.figure(figsize=(5 * len(shown), 6))
    grid = fig.add_gridspec(2, len(shown), height_ratios=[5, 1], hspace=0.3)
    info_ax = fig.add_subplot(grid[1, :])
    info_ax.axis("off")
    info = info_ax.text(
        0.5,
        0.5,
        "Hover over plots to see hyperparameters" if hyperparams else "",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    lines = {}
    for i, (key, title, ylabel, style) in enumerate(shown):
        ax = fig.add_subplot(grid[0, i])
        (line,) = ax.plot(values[key], style)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        lines[ax] = (line, key)

    def hover(event):
        if event.inaxes not in lines:
            return
        line, key = lines[event.inaxes]
        hit, ind = line.contains(event)
        if not hit or ind["ind"][0] >= len(hyperparams):
            return
        epoch = ind["ind"][0]
        params = ast.literal_eval(hyperparams[epoch])
        params["learn rate"] = f"{float(params['learn rate']):.1e}"
        params[key] = line.get_ydata()[epoch]
        info.set_text("  |  ".join(f"{k}: {v}" for k, v in params.items() if v is not None))
        fig.canvas.draw_idle()

    if hyperparams:
        fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.manager.set_window_title("Metrics Over Epochs")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <log_file>")
    plot_metrics(*extract_metrics(sys.argv[1]))
