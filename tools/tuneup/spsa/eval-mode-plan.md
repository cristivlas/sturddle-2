# Eval Mode for SPSA Tuner

## Context

The engine's piece square tables (PSTs) are currently disabled (`USE_PIECE_SQUARE_TABLES = false`)
because the current values hurt performance. PST values matter primarily through their effect on
qsearch capture ordering, not just static eval — so optimizing them requires actual engine search,
not a pure-Python linear model.

This adds an `"eval"` mode to the SPSA tuner where workers evaluate positions from local PGN files.
The Texel-style loss replaces WDL as the SPSA signal. This mode is general-purpose — it works for
any tunable parameters, not just PSTs.

**Design principles:**
- Game mode (`"mode": "games"`) remains the default; no changes to existing behavior
- Reuse existing `depth` field from TuningConfig for eval search depth
- Workers map "N games" (chunk size) to "N positions" — coordinator doesn't know about positions
- Per-position losses reported as `{"plus": [...], "minus": [...]}`, coordinator aggregates
- `spsa.py` is untouched — the SPSA math is mode-agnostic
- Minimize branching — avoid unnecessary if/else for mode where the code can just work for both

---

## Worker Architecture (Eval Mode)

Workers manage a pool of **persistent engine subprocesses** via UCI protocol
(`python-chess` `chess.engine.SimpleEngine.popen_uci`). Engines stay alive across
evaluations — no startup/shutdown overhead per position.

### PGN Streaming

Positions are read from PGN files using `chess.pgn.read_game()`, which is already
streaming (one game at a time, no bulk RAM load). The worker assigns **one game per
engine instance** per eval cycle:

1. Read N games from PGN file handles (N = number of engine instances)
2. Each engine evaluates all positions in its assigned game at configured depth
3. Each position's loss is tagged with that game's `Result` header
4. Collect per-position losses across all engines, report to coordinator
5. Read next N games, repeat
6. At EOF, reopen from the top (or advance to next PGN file)

This avoids buffer management and position-to-outcome bookkeeping across game boundaries.
Games from the same TC have roughly similar ply counts, so engines finish around the same time.

### Engine Lifecycle

- Spawn at worker startup, keep alive for the entire session
- Between parameter sets (theta_plus → theta_minus): `setoption` for each param + `ucinewgame`
- Between positions within a batch: `position fen ...` + `go depth N`
- `OwnBook` must be disabled; `depth` must be > 0
- Hash size set via `setoption name Hash value N` (from `eval_hash` in worker config)

### Engine Crash Recovery

If an engine process dies mid-evaluation, `python-chess` raises `EngineTerminatedError`.
The worker handles this gracefully:
- Collect losses for positions already evaluated in that game (partial result)
- Respawn the engine process, re-apply fixed UCI options
- Continue with the next game — don't lose the whole chunk
- This is more resilient than game mode, where a cutechess-cli crash loses the entire chunk

### Position Filtering

- Skip first N plies per game (reuse existing `book_depth` from WorkerConfig)
- Skip positions where side to move is in check
- Skip games with no result (`*`)

---

## Files to Modify

### 1. `tools/tuneup/spsa/config.py`

**TuningConfig** — add fields:
```python
mode: str = "games"              # "games" or "eval"
eval_scale: float = 400.0        # Texel sigmoid denominator
```

`depth` (already exists) is reused for eval search depth.
`book_depth` (already in WorkerConfig) is reused for skip-plies in eval mode.

Add to `validate()`:
```python
if self.mode not in ("games", "eval"):
    errors.append('mode must be "games" or "eval"')
if self.mode == "eval" and not self.depth:
    errors.append("eval mode requires depth")
```

Add `mode`, `eval_scale` to `to_json()` dict. `from_json()` already picks up new scalar
fields automatically via the generic loop (lines 214-221).

**WorkerConfig** — add fields:
```python
pgn_files: list = field(default_factory=list)    # PGN paths for eval mode
eval_hash: int = 16                               # hash size (MB) for eval mode
```

`from_json()` already handles new fields via `cls.__dataclass_fields__` filtering.

**WorkResult** — add fields:
```python
losses_plus: list = field(default_factory=list)
losses_minus: list = field(default_factory=list)
```

Existing `from_dict()` picks these up when present, defaults to `[]` when absent (game mode).

### 2. `tools/tuneup/spsa/spsa.py`

**SPSAState** — add fields for eval-mode checkpoint:
```python
total_loss_plus: float = 0.0     # sum of Texel losses (theta_plus)
total_loss_minus: float = 0.0    # sum of Texel losses (theta_minus)
total_loss_count: int = 0        # number of positions evaluated
```

These are kept separate from the WDL fields (different types: float sums vs int counts).
Mixing them risks subtle bugs during checkpoint/resume if a state file is loaded with the
wrong mode. The cost is 3 extra fields — cheap insurance.

Add to `to_dict()` and `from_dict()` (with `.get()` defaults for backward compat with
existing state files).

### 3. `tools/tuneup/spsa/coordinator.py`

**`__init__`** — store mode, add eval accumulators:
```python
self.mode = config.mode
self.loss_plus_sum = 0.0
self.loss_minus_sum = 0.0
self.loss_count = 0
```

**`_prepare_iteration()`** — restore or reset eval accumulators alongside WDL:
```python
if st.current_delta:
    # existing WDL restore...
    self.loss_plus_sum = st.total_loss_plus
    self.loss_minus_sum = st.total_loss_minus
    self.loss_count = st.total_loss_count
else:
    # existing WDL reset...
    self.loss_plus_sum = 0.0
    self.loss_minus_sum = 0.0
    self.loss_count = 0
```

**`get_work()`** — no changes. Even-rounding stays (keeps things clean for both modes).

**`submit_result()`** — after existing WDL accumulation (line 609-612), add:
```python
if self.mode == "eval":
    self.loss_plus_sum += sum(result.losses_plus)
    self.loss_minus_sum += sum(result.losses_minus)
    self.loss_count += len(result.losses_plus)
```

Adjust log message (line 634-640) to show loss info instead of WDL when in eval mode.

**`_complete_iteration()`** — branch on mode for score computation:
```python
if self.mode == "games":
    # existing WDL -> score logic (lines 664-678)
else:
    mean_loss_plus = self.loss_plus_sum / self.loss_count
    mean_loss_minus = self.loss_minus_sum / self.loss_count
    # Negate: lower loss = better, SPSA expects higher = better
    avg_score_plus = -mean_loss_plus
    avg_score_minus = -mean_loss_minus
```

ELO estimate: let it compute whatever it computes — the values are meaningless in eval mode
but won't crash anything. No extra branching needed.

In the reset block (lines 712-720), also reset eval accumulators.

**`_sync_and_save()`** — sync eval accumulators alongside WDL:
```python
st.total_loss_plus = self.loss_plus_sum
st.total_loss_minus = self.loss_minus_sum
st.total_loss_count = self.loss_count
```

**Dashboard/SSE** — add `mode` to status data. Dashboard changes are optional for v1;
the existing display works, just with WDL showing zeros and ELO being nonsensical.

### 4. `tools/tuneup/spsa/worker.py`

**New function: `texel_loss(eval_cp, outcome, scale)`**
```
sigmoid = 1 / (1 + 10^(-eval_cp / scale))
loss = (outcome - sigmoid)^2
```

**New class or section: Engine pool management**
- Spawn N engine instances at startup via `chess.engine.SimpleEngine.popen_uci(engine_path)`
- N = `concurrency` from WorkerConfig
- Set fixed UCI options on all engines (Hash, OwnBook=false, etc.)
- Keep engines alive for the session; close on shutdown

**New function: `eval_game(engine, game, skip_plies, depth, scale, params)`**
- Takes one engine instance + one parsed PGN game
- Sets params via `setoption` + `ucinewgame` to clear hash/state
- Replays moves, evaluating each position after `skip_plies`:
  - Skip positions in check
  - `position fen ...` + `go depth N` -> get score
  - Compute `texel_loss(score, outcome_from_stm, scale)`
- On `EngineTerminatedError`: return losses collected so far (partial result),
  respawn engine, re-apply fixed UCI options. The caller continues with next game.
- Returns list of per-position losses (may be partial if engine crashed mid-game)

**New function: `run_eval_chunk(engines, pgn_reader, work, config)`**

Orchestrates one chunk of work:
1. Read one game per engine from PGN stream
2. Fan out: each engine evaluates its game with theta_plus params → losses_plus
3. Same games again with theta_minus params → losses_minus
4. Flatten per-position losses across all engines
5. Check `_shutdown_requested` between games for fast interrupt
6. Returns `{"plus": [...], "minus": [...]}`

**New class: `PGNStream`**

Wraps the list of PGN file paths with streaming iteration:
- Opens files sequentially, calls `chess.pgn.read_game()` one game at a time
- At EOF of current file, advance to next
- After last file, loop back to first
- Skips games with no result (`*`)
- `next_game()` returns a `chess.pgn.Game` or blocks until available

**Modify `worker_loop()`** — branch on mode after fetching tuning config:

Eval mode startup:
1. Validate `pgn_files` exist
2. Spawn engine pool (N = concurrency)
3. Set fixed UCI options on all engines
4. Create `PGNStream` from pgn_files list

Main loop: when `mode == "eval"`, call `run_eval_chunk()` instead of `run_games()`.
Build result dict with `losses_plus`/`losses_minus` instead of `wins`/`draws`/`losses`.

### 5. `tools/tuneup/spsa/genconfig.py`

Add CLI arguments:
- `--mode {games,eval}` (default: games)
- `--eval-scale` (default: 400)

Pass to TuningConfig. When mode is eval:
- Add `pgn_files: []` placeholder to worker.json template
- Add `eval_hash: 16` to worker.json template

### 6. Dashboard template (optional, v2)

Low priority. The dashboard works as-is with eval mode — it just shows zeros for WDL.
If desired later:
- Relabel "Games" -> "Positions" in progress display
- Show "Loss +" / "Loss -" / "Loss Diff" in history
- Driven by `mode` field in SSE data

---

## Implementation Order

1. **config.py** — foundation; add all new fields
2. **spsa.py** — add checkpoint fields to SPSAState
3. **worker.py** — PGNStream, engine pool, texel_loss, eval_game, run_eval_chunk, mode branch
4. **coordinator.py** — eval-mode aggregation, score conversion, checkpoint
5. **genconfig.py** — CLI args and template generation
6. **dashboard** — optional, defer to v2

---

## Verification

1. **Game mode regression**: Run existing game-mode tuning config -> verify no behavioral change
2. **Eval mode smoke test**:
   - Generate eval-mode project: `python genconfig.py -m eval --depth 8 myproject`
   - Edit `worker.json` to point at actual PGN files
   - Start coordinator + worker
   - Verify: positions evaluated, losses reported, SPSA iterations advance, loss trends down
3. **Checkpoint/resume**: Kill coordinator mid-iteration, restart -> verify it resumes correctly
4. **Multi-engine**: Test with concurrency > 1, verify engines get different games and
   losses are correctly aggregated

---

## Open Questions / Notes

- **SPSA hyperparameters**: The `a` and `c` values tuned for game mode may need different
  defaults for eval mode (different gradient magnitudes). Not a code issue — just a tuning
  concern. Start with defaults and adjust.
- **Eval clamp**: Mate scores (~30000 cp) saturate the sigmoid to 1.0 or 0.0. The loss
  for those positions is either 0 (correct prediction) or 1.0 (wrong prediction). This is
  mathematically correct but produces no gradient (flat sigmoid region). Could optionally
  clamp eval to e.g. +/- 1000 cp to keep gradients alive. Not critical for v1 — most
  positions won't be mates.
- **python-chess dependency**: Already a transitive dependency of the engine. Only needed
  in eval mode (PGN parsing + UCI engine management).
- **Concurrency**: Maps naturally to engine pool size. Each engine evaluates one game at
  a time, single-threaded. Parallelism comes from multiple engines per worker + multiple
  workers.
