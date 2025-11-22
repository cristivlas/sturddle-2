#!/usr/bin/env python3
import ast
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
matplotlib.use('Qt5Agg')

def extract_metrics(file_path):
    loss_values = []
    mae_values = []
    accuracy_values = []
    hyperparams = []

    with open(file_path, 'r') as f:
        for line in f:
            # Try to match loss with hyperparams
            loss_match = re.search(r'epoch=\d+ loss=([\d.]+) hyperparam=(\{.*\})', line)
            if loss_match:
                loss, hyperparam_str = loss_match.groups()
                loss_values.append(float(loss))
                hyperparams.append(hyperparam_str)

            # Try to match MAE
            mae_match = re.search(r'epoch=\d+ mae=([\d.]+)', line)
            if mae_match:
                mae = mae_match.groups()[0]
                mae_values.append(float(mae))

            # Try to match accuracy
            accuracy_match = re.search(r'epoch=\d+ accuracy=([\d.]+)', line)
            if accuracy_match:
                accuracy = accuracy_match.groups()[0]
                accuracy_values.append(float(accuracy))

    return loss_values, mae_values, accuracy_values, hyperparams

def plot_metrics(loss_values, mae_values, accuracy_values, hyperparams):
    # Determine what to plot based on available data
    has_loss = len(loss_values) > 0
    has_mae = len(mae_values) > 0
    has_accuracy = len(accuracy_values) > 0
    has_hyperparams = len(hyperparams) > 0

    if not has_loss and not has_mae and not has_accuracy:
        print("No loss, MAE, or accuracy data found in file")
        return

    # Create subplots if we have multiple metrics
    if has_loss and has_mae and has_accuracy:
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 1], hspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        info_ax = fig.add_subplot(gs[1, :])
        info_ax.axis('off')

        # Create info text box
        info_text = info_ax.text(0.5, 0.5, 'Hover over plots to see hyperparameters',
                                ha='center', va='center', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot loss
        points1, = ax1.plot(loss_values, 'b-', label='Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Loss')
        ax1.grid(True)

        # Plot MAE
        points2, = ax2.plot(mae_values, 'r-', label='MAE')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Eval MAE')
        ax2.grid(True)

        # Plot Accuracy
        points3, = ax3.plot(accuracy_values, 'g-', label='Accuracy')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_title('Outcome Accuracy')
        ax3.grid(True)

        # Add hover functionality for loss (with hyperparams if available)
        if has_hyperparams:
            def update_info(ind, metric_name, metric_value):
                x, y = points1.get_data()
                x = x[ind['ind'][0]]
                params = ast.literal_eval(hyperparams[ind['ind'][0]])
                params['learn rate'] = f"{float(params['learn rate']):.1e}"
                params[metric_name] = metric_value
                # Filter out None values
                params = {k: v for k, v in params.items() if v is not None}
                text = '  |  '.join([f'{k}: {v}' for k, v in params.items()])
                info_text.set_text(text)

            def hover_handler(event):
                if event.inaxes == ax1:
                    cont, ind = points1.contains(event)
                    if cont:
                        x, y = points1.get_data()
                        y_val = y[ind['ind'][0]]
                        update_info(ind, 'loss', y_val)
                        fig.canvas.draw_idle()
                elif event.inaxes == ax2:
                    cont, ind = points2.contains(event)
                    if cont:
                        x, y = points2.get_data()
                        y_val = y[ind['ind'][0]]
                        update_info(ind, 'mae', y_val)
                        fig.canvas.draw_idle()
                elif event.inaxes == ax3:
                    cont, ind = points3.contains(event)
                    if cont:
                        x, y = points3.get_data()
                        y_val = y[ind['ind'][0]]
                        update_info(ind, 'accuracy', y_val)
                        fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', hover_handler)

        fig.canvas.manager.set_window_title('Metrics Over Epochs')

    elif has_loss and has_mae:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot loss
        points1, = ax1.plot(loss_values, 'b-', label='Loss')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.grid(True)

        # Plot MAE
        points2, = ax2.plot(mae_values, 'r-', label='MAE')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Eval MAE')
        ax2.grid(True)

        # Add hover functionality for loss (with hyperparams if available)
        if has_hyperparams:
            annot1 = ax1.annotate('', xy=(0,0), xytext=(-35, -30),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round', fc='w', alpha=0.8))
            annot1.set_visible(False)

            def update_annot1(ind):
                x, y = points1.get_data()
                x, y = (x[ind['ind'][0]], y[ind['ind'][0]])
                params = ast.literal_eval(hyperparams[ind['ind'][0]])
                params['learn rate'] = f"{float(params['learn rate']):.1e}"
                params['loss'] = y
                # Filter out None values
                params = {k: v for k, v in params.items() if v is not None}
                text = '\n'.join([f'{k}: {v}' for k, v in params.items()])
                annot1.set_text(text)
                annot1.xy = x, y

            def hover1(event):
                vis = annot1.get_visible()
                if event.inaxes == ax1:
                    cont, ind = points1.contains(event)
                    if cont:
                        update_annot1(ind)
                        annot1.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot1.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', hover1)

        fig.canvas.manager.set_window_title('Loss and MAE')

    elif has_loss:
        # Plot only loss
        fig, ax = plt.subplots()
        points, = plt.plot(loss_values, 'b-')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Loss')
        plt.grid(True)

        # Add hover functionality with hyperparams
        if has_hyperparams:
            annot = ax.annotate('', xy=(0,0), xytext=(-35, -30),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round', fc='w', alpha=0.8))
            annot.set_visible(False)

            def update_annot(ind):
                x, y = points.get_data()
                x, y = (x[ind['ind'][0]], y[ind['ind'][0]])
                params = ast.literal_eval(hyperparams[ind['ind'][0]])
                params['learn rate'] = f"{float(params['learn rate']):.1e}"
                params['loss'] = y
                # Filter out None values
                params = {k: v for k, v in params.items() if v is not None}
                text = '\n'.join([f'{k}: {v}' for k, v in params.items()])
                annot.set_text(text)
                annot.xy = x, y

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = points.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', hover)

        fig.canvas.manager.set_window_title('Loss')

    elif has_mae:
        # Plot only MAE
        fig, ax = plt.subplots()
        plt.plot(mae_values, 'r-')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.title('MAE')
        plt.grid(True)
        fig.canvas.manager.set_window_title('MAE')

    elif has_accuracy:
        # Plot only Accuracy
        fig, ax = plt.subplots()
        plt.plot(accuracy_values, 'g-')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.title('Accuracy')
        plt.grid(True)
        fig.canvas.manager.set_window_title('Accuracy')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python script.py <log_file>')
        print('This script will plot loss, MAE, and/or accuracy metrics found in the log file')
    else:
        loss_values, mae_values, accuracy_values, hyperparams = extract_metrics(sys.argv[1])
        plot_metrics(loss_values, mae_values, accuracy_values, hyperparams)