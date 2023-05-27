# Sturddle

Sturddle 2 is a fork of the Sturddle Chess Engine https://github.com/cristivlas/sturddle-chess-engine
with a rewritten (and trained from scratch) neural network.

## Training the Neural Net

1) Start with games saved as PGN (downloaded from the Internet, or from engine tournaments).
2) Generate sqlite3 database(s) of positions from PGN games, using: tools/sqlite/mkposdb.py
3) Analyse positions with tools/sqlite/analyse.py or tools/sqlite/analyse\_parallel.py; both
scripts require a UCI engine for analysis (such Sturddle 1.xx, Stockfish, etc).
4) Generate numpy file from database of evaluations (generated in previous step), using
tools/nnue/one-hot-encode.py (use --half for half precision if storage capacity is limited)
5) Train the neural net running tools/nnue/train.py (requires Tensorflow and optionally cuda-toolkit)
6) Generate weights.h by exporting model trained at step 5), running:
./tools/nnue/train.py export -m <path-to-model> -o weights.h
7) Build engine (using tools/build.py, or by running "python3 setup.py build\_ext --inplace")

For more details on tools usage, invoke any of the scripts with the "--help" option.

