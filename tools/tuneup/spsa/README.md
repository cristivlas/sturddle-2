# Distributed SPSA Tuner

A distributed SPSA (Simultaneous Perturbation Stochastic Approximation) tuner
for chess engines, using cutechess-cli to run games. Designed for small
heterogeneous LANs (2-3 machines, mixed Linux/Windows).

## Architecture

- **Coordinator**: HTTP server managing SPSA state. Generates perturbations,
  distributes work, collects scores, updates parameters.
- **Workers**: Poll the coordinator for game batches, run cutechess-cli locally,
  report scores back. Each worker saves PGNs and logs locally.

## Prerequisites

- [cutechess-cli](https://github.com/cutechess/cutechess) installed on each worker machine
- Engine source tree (for genconfig.py parameter discovery)

## Quick Start (single machine)

### 1. Generate a tuning project

From the repo root:

```bash
python tools/tuneup/spsa/genconfig.py my-test -D 8 -b 2000 -g 100
```

This creates `tuneup/my-test/` with:
- `tuning.json` — session config (parameters, SPSA settings, search control)
- `worker.json` — local worker config (engine path, book, concurrency)
- `engine.bat` — engine wrapper (Windows only)

Options:
- First argument is the project name
- `all` (default) — tune all parameters, or list specific names
- `-D` — fixed search depth (mutually exclusive with `-t`)
- `-t` — time control, e.g. `1+0.1` (default)
- `-H` — hash table size in MB (default: 256)
- `-T` — engine threads (default: 1)
- `-b` — total games budget (default: 10000)
- `-g` — games per SPSA iteration (default: 200)
- `-c` — SPSA perturbation magnitude (default: 2.0)
- `-a` — SPSA learning rate (default: 1.0)

### 2. Review and edit configs

Open `tuneup/my-test/tuning.json` and adjust:
- Remove parameters you don't want to tune
- Adjust bounds (`lower`/`upper`) if needed
- Tweak SPSA hyperparameters (`a`, `c`, `budget`, etc.)

Open `tuneup/my-test/worker.json` and verify:
- `engine` path is correct
- `opening_book` path is correct (defaults to absolute path to `tuneup/books/8moves_v3.pgn`)
- `concurrency` matches your CPU count
- `cutechess_cli` is in your PATH (or set full path)

### 3. Start the coordinator

```bash
cd tuneup/my-test
python ../../tools/tuneup/spsa/coordinator.py -c tuning.json
```

### 4. Start a worker (separate terminal)

```bash
cd tuneup/my-test
python ../../tools/tuneup/spsa/worker.py -c worker.json
```

The worker connects, fetches the tuning session, and starts playing games.

## Multi-Machine Setup

### Coordinator machine

```bash
cd tuneup/my-test
python ../../tools/tuneup/spsa/coordinator.py -c tuning.json -p 8080
```

The coordinator binds to `0.0.0.0:8080` and accepts connections from any worker.

### Each worker machine

1. Copy or create a `worker.json` with machine-specific paths:

```json
{
  "coordinator": "http://192.168.1.10:8080",
  "engine": "/home/user/engines/sturddle/main.py",
  "cutechess_cli": "/usr/local/bin/cutechess-cli",
  "concurrency": 8,
  "opening_book": "/home/user/books/8moves_v3.pgn",
  "book_format": "pgn",
  "book_depth": 8,
  "games_dir": "/home/user/spsa/my-test/games",
  "log_file": "/home/user/spsa/my-test/worker.log"
}
```

2. Start the worker:

```bash
python /path/to/tools/tuneup/spsa/worker.py -c worker.json
```

Workers self-balance: faster machines with higher concurrency naturally process
more game batches.

## Monitoring

### Coordinator logs

The coordinator prints iteration progress to stdout and to
`logs/coordinator.log` in the project directory:

```
Iteration 42: c_k=1.8234, a_k=0.012345
Assigned 50 games (iter 42, 50/200 assigned)
Result: iter 42, 50 games, +0.530 / -0.470 (50/200 done)
...
============================================================
Iteration 42 complete
Scores: +0.5250 (ELO +17.4) / -0.4750 (ELO -17.4)
Updated parameters:
  ParamA: 100.0000 -> 102.3400 (engine: 102)
  ParamB: 0.5000 -> 0.4820 (engine: 0.482)
============================================================
```

### Status endpoint

```bash
curl http://localhost:8080/status
```

Returns JSON with current iteration, parameter values, and progress.

## Resuming

If the coordinator is interrupted, restarting it will resume the existing session:

```bash
python ../../tools/tuneup/spsa/coordinator.py -c tuning.json

```

Picks up from the last completed iteration (`spsa_state.json`). Workers
reconnect automatically.

To reinitialize:
```bash
python ../../tools/tuneup/spsa/coordinator.py -c tuning.json --clean

```

## Project Directory Layout

After a tuning run, a project directory looks like:

```
tuneup/my-test/
  tuning.json           # session config
  worker.json           # local worker config
  engine.bat            # engine wrapper (Windows)
  spsa_state.json       # checkpoint (auto-generated)
  logs/
    coordinator.log     # theta progression, iteration results
  games/
    iter_0001_PC1.pgn   # PGNs from each iteration (per worker)
    iter_0002_PC1.pgn
    ...
  worker.log            # worker activity log
```

## Configuration Reference

### tuning.json (session-level, shared)

| Field | Description | Default |
|---|---|---|
| `engine.protocol` | Engine protocol | `"uci"` |
| `engine.fixed_options` | Fixed UCI options (Hash, Threads, etc.) | `{}` |
| `time_control` | Time control string | `"1+0.1"` |
| `depth` | Fixed search depth (overrides time_control if set) | `null` |
| `games_per_iteration` | Games per SPSA iteration | `200` |
| `output_dir` | Coordinator output (logs, checkpoint) | project dir |
| `spsa.budget` | Total games budget | `10000` |
| `spsa.a` | Learning rate | `1.0` |
| `spsa.c` | Perturbation magnitude | `2.0` |
| `spsa.A_ratio` | Stabilization constant (fraction of max iterations) | `0.1` |
| `spsa.alpha` | Learning rate decay exponent | `0.602` |
| `spsa.gamma` | Perturbation decay exponent | `0.101` |
| `parameters.<name>.init` | Initial value | — |
| `parameters.<name>.lower` | Lower bound | — |
| `parameters.<name>.upper` | Upper bound | — |
| `parameters.<name>.type` | `"int"` or `"float"` | `"int"` |

### worker.json (per-machine)

| Field | Description | Default |
|---|---|---|
| `coordinator` | Coordinator URL | `"http://localhost:8080"` |
| `engine` | Absolute path to engine (or wrapper script) | auto-detected |
| `cutechess_cli` | Path to cutechess-cli | `"cutechess-cli"` |
| `concurrency` | Concurrent games | CPU count |
| `opening_book` | Absolute path to opening book | auto-detected |
| `book_format` | Book format (`pgn` or `epd`) | `"pgn"` |
| `book_depth` | Opening book depth in plies | `8` |
| `games_dir` | Absolute path for PGN output | auto-detected |
| `log_file` | Absolute path to worker log | auto-detected |

## SPSA Algorithm

The tuner uses standard SPSA with Bernoulli perturbations:

- Each iteration generates a random +/-1 vector delta
- Two engine configs are created: theta + c_k * delta and theta - c_k * delta
- Games are played between them; scores are used to estimate the gradient
- Parameters are updated: theta += a_k * gradient_estimate
- Learning rates decay: a_k = a/(A+k+1)^alpha, c_k = c/(k+1)^gamma
- Integer parameters are rounded after update but tracked as floats internally
