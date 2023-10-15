#!/usr/bin/env python3

import subprocess
import sys
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def plot(values):
    plt.plot(values)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Missing input file')
    else:
        command = f'grep loss {sys.argv[1]} | cut -f4 -d\  | cut -f2 -d='
        result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        values = [float(v) for v in result.stdout.decode().split()]
        if values:
            plot(values)
