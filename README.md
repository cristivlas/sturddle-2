# Sturddle 2

Sturddle 2 is a fork of the Sturddle Chess Engine https://github.com/cristivlas/sturddle-chess-engine
with many bug fixes, and a rewritten (and trained from scratch) neural network.

## Building the Engine
<code>python3 tools\build.py</code> builds a native executable for the host OS. The executable
bundles binary images that support AVX512, AVX2, and generic SSE2.

To build just a python module:
<code>python3 setup.py build_ext --inplace</code>
or:
<code>CC=clang++ CFLAGS=-march=native python3 setup.py build_ext --inplace</code>
or:
<code>CC=clang++ CFLAGS=-march=native NATIVE_UCI=1 python3 setup.py build_ext --inplace</code>
etc.

Clang is recommended, the GNU C++ compiler may work but it is not supported.

If built with NATIVE_UCI, run the UCI engine by invoking main.py; without NATIVE_UCI, run
sturddle.py.

## Training the Neural Net

1) Start with games saved as PGN (downloaded from the Internet, or from engine tournaments).
2) Generate sqlite3 database(s) of positions from PGN games, using: tools/sqlite/mkposdb.py.
3) Analyse positions with tools/sqlite/analyse.py or tools/sqlite/analyse\_parallel.py; both
scripts require a UCI engine for analysis (such Sturddle 1.xx, Stockfish, etc).

Alternatively:
- Download PGNs from https://database.lichess.org/ and extract
evaluations using tools/sqlite/pgntoevals.py.
- Download binpack files, convert them to plain using a development version
of Stockfish, then use tools/sqlite/plain_to_db.py.

4) Generate HDF5 file(s) from database(s) produced by any of the methods above: use tools/nnue/toh5.py.

5) Train the neural net running tools/nnue/train-v3.py (requires Tensorflow and optionally cuda-toolkit)
6) Generate weights.h by exporting model trained at step 5), running:
./tools/nnue/train.py export -m <path-to-model> -o weights.h
7) Build engine (using tools/build.py, or by running "python3 setup.py build\_ext --inplace")

For more details on tools usage, invoke any of the scripts with the "--help" option.


## Tuning the Engine

There are two ways to tune parameters defined in config.h:
- Using https://chess-tuning-tools.readthedocs.io/en/latest/
- or using the Lakas optimizer https://github.com/fsmosca/Lakas

Install the preferred tool and cutechess-cli.

Edit the config.h file, and replace DECLARE_VALUE with DECLARE_PARAM for the parameters to be tuned.
Build the python module, preferably with NATIVE_UCI:
<code>CC=clang++ CFLAGS=-march=native NATIVE_UCI=1 python3 setup.py build_ext --inplace</code>
Run ./main.py, and enter the 'uci' command. The parameters of interest should be listed in the output.
Quit the engine.

Run tuneup/gentune.py to generate a JSON configuration file for chess-tuning-tools, or
tuneup/genlakas.py to generate a wrapper script to invoke the Lakas optimizer.

Run the optimizer. Once the optimizer converges, edit the config.h file, and change the values
of the parameters; change DECLARE_PARAM back to DECLARE_VALUE.


