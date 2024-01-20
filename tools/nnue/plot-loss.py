#!/usr/bin/env python3

import ast
import re
import sys
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

matplotlib.use('TkAgg')

def extract_values_and_hyperparams(file_path):
    values = []
    hyperparams = []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'epoch=\d+ loss=([\d.]+) hyperparam=(\{.*\})', line)
            if match:
                loss, hyperparam_str = match.groups()
                values.append(float(loss))
                hyperparams.append(hyperparam_str)
    return values, hyperparams

def plot(values, hyperparams):
    fig, ax = plt.subplots()
    points, = plt.plot(values)
    plt.grid(True)

    annot = ax.annotate('', xy=(0,0), xytext=(-35, -30),
            textcoords='offset points',
            bbox=dict(boxstyle='round', fc='w', alpha=0.8))
    annot.set_visible(False)
    #ax.xaxis.set_major_locator(MultipleLocator(5))
    #ax.yaxis.set_major_locator(MultipleLocator(0.015))

    def update_annot(ind):
        x, y = points.get_data()
        x, y = (x[ind['ind'][0]], y[ind['ind'][0]])
        params = ast.literal_eval(hyperparams[ind['ind'][0]])
        params['learn rate'] = f"{float(params['learn rate']):.1e}"
        params['loss'] = y
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

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Missing input file')
    else:
        values, hyperparams = extract_values_and_hyperparams(sys.argv[1])
        if values and hyperparams:
            plot(values, hyperparams)
