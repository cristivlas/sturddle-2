# Sturddle 2

Sturddle 2 is a fork of my Sturddle Chess Engine (https://github.com/cristivlas/sturddle-chess-engine)
with many bug fixes and a rewritten (and trained from scratch) neural network.

## Building the Engine

Python (3.9 or higher) and a working C++ compiler are required. The compiler must support
the C++ 20 standard. Clang 16 or higher is recommended (clang-cl.exe on Windows).
The GNU C++ compiler and the Microsoft Compiler may work but are not well tested.

The Python libraries in requirements.txt must be installed.

To build a self-contained executable, PyInstaller is also needed. It is recommended to use a Python
virtual environment for building.

<code>python3 tools\build.py</code> builds a native executable for the host OS.

On x64_64, the executable bundles binary images that support AVX512, AVX2, AVX, VNNI, and generic SSE2.
At runtime, a "bootstrap" python scripts selects the right engine module based on processor features.

To build just a python module:
<code>python3 setup.py build_ext --inplace</code>
or:
<code>CC=clang++ CFLAGS=-march=native python3 setup.py build_ext --inplace</code>

# Neural Net Architecture
This version of the engine uses an original model nicknamed **Raptor-III**, which expands on the bucketing idea introduced
in 2.4.0 with the **Bluejay** architecture, by adding bucket-shifting in the "spatial attention" modulation path, and to the
second hidden layer.

*NOTE*: Some internal details omitted for simplicity.

![plot](./model.png)

Inference runs on the CPU using vectorized instructions.
The x86_64 and ARM64 with NEON architectures are supported.
To enable half precision on ARM processors that support it, set CXXFLAGS or CFLAGS:
CFLAGS="march=armv8.2-a+fp16"


## Tuning the Engine

Search and evaluation parameters are defined in `config.h` using tuning macros
(`DECLARE_VALUE`, `DECLARE_PARAM`, etc.). Edit `config.h` to select which parameters
to expose for tuning — see the comments in the file for the available macros and options.

Build the engine. On POSIX systems, building just the python module is sufficient and faster:
<code>CC=clang++ CFLAGS=-march=native python3 setup.py build_ext --inplace</code>
On Windows, a full build via `tools/build.py` is recommended due to default stack size
limitations in `python.exe` that can cause crashes at higher search depths.

Run `./main.py` and enter the `uci` command. Tunable parameters will be listed in the output.

### Chess Tuning Tools / Lakas

- https://chess-tuning-tools.readthedocs.io/en/latest/
- or the Lakas optimizer https://github.com/fsmosca/Lakas

1) Install the preferred tool and `cutechess-cli`.

2) Run `tuneup/gentune.py` to generate a JSON configuration file for chess-tuning-tools, or
`tuneup/genlakas.py` to generate a wrapper script to invoke the Lakas optimizer.

3) Run the optimizer. Once the optimizer converges, run `tools/tuneup/apply_lakas.py`
to apply the results to `config.h`, or edit it manually; change `DECLARE_PARAM` back
to `DECLARE_VALUE`.

### Distributed SPSA Tuner (Experimental)

Starting with version 2.5.1, a distributed SPSA tuner is available. It uses a
coordinator/worker architecture designed for small heterogeneous LANs and provides
a live web dashboard for monitoring progress.

See [tools/tuneup/spsa/README.md](tools/tuneup/spsa/README.md) for setup and usage.
Use `tools/tuneup/apply_spsa.py` to apply tuning results from `spsa_state.json` to `config.h`.


