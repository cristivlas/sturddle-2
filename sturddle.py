#!/usr/bin/env python3

# import everything for the benefit of pyinstaller
import argparse
import importlib
import logging
import math
import os
import sysconfig
import time

import chess
import chess.pgn
import chess.polyglot
import chess.syzygy

#try:
#    import cpufeature
#except:
#    pass
import psutil

import armcpu
import uci
import worker

if __name__ == '__main__':
    uci.main()
