# Native Executable Build — Plan of Action

## Status

Windows and Linux milestones **complete** on branch `2.5.1-hnat`. Native
`dist/native/sturddle-2.5.1-hnat[.exe]` builds cleanly on both platforms,
runs UCI end-to-end, and (on Windows) produces bit-identical search output
to the Cython `.pyd` at the same depth. Per-arch variants are pending.

## Goal

Build Sturddle as a standalone native executable (no Python runtime, no
Cython) on both Linux and Windows, reusing the existing `uci_native.cpp`
UCI loop.

## Guiding decisions

- **Windows first, clang-cl only.** `sturddle*.exe` is built with
  `clang-cl.exe` (LLVM). MSVC `cl.exe` is **not** a supported compiler —
  the codebase uses clang-specific flags (`-march=native`, `-Ofast`,
  `-Wno-nan-infinity-disabled`). Linux and macOS follow.
- **Keep the Python/Cython build path working.** All changes are gated by
  a `NATIVE_BUILD` define. `setup.py` / `tools/make-win.bat` continues to
  produce `chess_engine*.pyd` from the same source tree.
- **Bootstrap with `SHARED_WEIGHTS=1`.** `weights.bin` is loaded at
  startup from the exe directory. Single-file distribution uses
  `#embed "weights.bin"` (enabled by `make-native.py --embed`).
- **Minimum viable UCI first.** Python-only callbacks (`_pgn`,
  `_print_state`, `_report`, `_on_next`, `_engine`, etc.) stay `nullptr`
  in native mode; call sites already null-check them.

## Step 1 — Native EPD serializer ✓ (commit `fe57881`)

Added `chess::epd::to_string(const State&)` in `chess.h` alongside the
existing parsers. Under `NATIVE_BUILD`, `Context::epd()` calls it
directly instead of routing through the Python callback.

**Non-obvious detail discovered during test authoring:** python-chess's
`Board.epd()` suppresses the en-passant square if no pawn could legally
capture (pseudo-legal ep check). The native serializer must match — a
raw FEN field like `... e6` silently becomes `... -` in python-chess when
there's no capturing pawn. Implemented with a short bitboard check
(~20 LoC).

Tests (`test/unit_test.py`): 10 functions cross-checking native vs.
python-chess output — startpos, empty board, all 13 castling subsets,
legal/illegal ep targets, empty-run digit compression, every piece
symbol, move sequences (Ruy Lopez, QGD, Sicilian, Italian, Closed
Sicilian, KID), FEN corpus, round-trip stability.

## Step 2 — No-op `cython_wrapper` under `NATIVE_BUILD` ✓ (commit `018943f`)

`utility.h`:
- `#include "Python.h"` gated behind `#if !NATIVE_BUILD`.
- Native `cython_wrapper`: empty `GIL_State`, `call` is a direct
  invocation (with `ASSERT(fn)`), `call_nogil` keeps the `noexcept`
  contract (required by the search entry wrappers in `search.h`) and
  logs uncaught exceptions to `stderr` via `std::fprintf`.
- `CancelReason` enum and `cancel_search` declaration stay visible to
  both builds — they're generic cancellation primitives, not
  Python-specific.

`context.cpp:cancel_search`: only the `PyErr_Print()` line is gated under
`#if !NATIVE_BUILD`; the rest (Context::cancel + log + _exit) remains
identical in both builds.

## Step 3 — Gate `Python.h` + install default callback sinks ✓ (commit `dce7a35`)

`context.h:30`: opaque `using PyObject = struct _object;` replaces
`#include "Python.h"` under `NATIVE_BUILD`. Callback signatures compile
unchanged.

`context.cpp`: added `native_log_message` (writes to stderr with level
prefix) and `native_vmem_avail` (returns 0 — OS-native paths in
`search.cpp` handle Windows/Linux/Mac natively). Installed as defaults
for `_log_message` and `_vmem_avail` under `NATIVE_BUILD`. Other
Python-only callbacks stay `nullptr` and are already null-checked at
every call site.

(This fold-in also covers what the plan originally called Step 4 — a
separate logger was unnecessary since the minimal stderr sink is enough
for the first milestone.)

## Step 4 — `main_native.cpp` + `version.h` ✓ (commit `ab5accb`)

New `version.h` is the single source of truth for version numbers:

```c
#define STURDDLE_VERSION_MAJOR 2
#define STURDDLE_VERSION_MINOR 5
#define STURDDLE_VERSION_PATCH "1-hnat"
#define STURDDLE_VERSION "2.5.1-hnat"   // assembled via preprocessor concat
```

Consumed by:
- `main_native.cpp` — uses `STURDDLE_VERSION` literal.
- `__init__.pyx` — `cdef extern` reads the three macros, preserves
  `__major__` / `__minor__` / `__patch__` / `__build__` shape so
  `tools/build.py` is untouched.
- `tools/make-native.py` — regex-extracts MAJOR/MINOR/PATCH to build the
  output filename.

`main_native.cpp` is 83 LoC: parses `-D/--dev-mode` and `-v/--verbose`,
derives `exe_dir` from `argv[0]`, calls `search::Context::init(exe_dir)`
inside a try/catch, populates the `params` map (`name`, `version`,
`dir`, optional `debug`/`dev_mode`), calls `uci_loop(params)`.

## Step 5 — Build system ✓ (commit `12771ce`)

**Python script, not a .bat.** `tools/make-native.py` handles Windows
now and is shaped to accept a Linux branch later. Benefits over a bat
file: regex-based version extraction (vs. `findstr` + delayed
expansion), native `ThreadPoolExecutor` for parallel compile,
cross-platform dispatch via `sys.platform`, clean `tempfile.TemporaryDirectory`
for intermediates.

### Shape

```
python tools/make-native.py [ARCH]
```

`ARCH ∈ {native (default), AVX, AVX2, AVX2_VNNI, AVX512, AVX512_BF16}`.
Arch-flag table mirrors `tools/build.py:109-120`.

### Output

| Variant | Filename |
|---------|----------|
| PyInstaller bundle (existing `tools/build.py`) | `dist/sturddle-2.5.1-hnat.exe` |
| Native, `-march=native` (dev default) | `dist/native/sturddle-2.5.1-hnat.exe` |
| Native, AVX2 | `dist/native/sturddle-2.5.1-hnat-avx2.exe` |
| Native, AVX2+VNNI | `dist/native/sturddle-2.5.1-hnat-avx2-vnni.exe` |
| Native, AVX-512+BF16 | `dist/native/sturddle-2.5.1-hnat-avx512-bf16.exe` |

The `dist/native/` subdir disambiguates from the PyInstaller `dist/*.exe`
at the path level — no `-native` suffix in the filename itself.

### Compile/link specifics

- Parallel compile: one `clang-cl /c` per source file via
  `ThreadPoolExecutor(max_workers=os.cpu_count())`. `/MP` flag is
  **omitted** (clang-cl silently ignores it; the per-file invocation is
  the only way to actually parallelize).
- `.obj` files go to a `tempfile.TemporaryDirectory` — auto-cleaned on
  exit, no cross-arch stale-reuse risk by construction.
- MSVC env bootstrap: if `%INCLUDE%` isn't set, the script runs
  `vcvars64.bat` and inherits its env before invoking `clang-cl`.
- **Critical linker flag: `/STACK:33554432`** (32 MB). Default 1 MB
  overflows during search — surfaces as exit code `-1073741571`
  (`0xC00000FD` = `STATUS_STACK_OVERFLOW`) and silent empty stdout
  under MSYS bash. Matches the `editbin /STACK:33554432` post-process
  in `tools/build.py:195`.

### Non-obvious Cython-side include

`uci_native.cpp` uses `_isatty`/`_fileno` on Windows. These come from
`<io.h>`. Previously satisfied transitively via `Python.h` → native
build broke once that transitive path was gated out. Added explicit
`#include <io.h>` on Windows — harmless in the Cython build, required
for native.

## Step 6 — Runtime assets ✓

Copied next to the exe by `make-native.py`:

- `weights.bin` — NNUE weights (required for `SHARED_WEIGHTS=1`)
- `book.bin` — Polyglot opening book (consumed by `NATIVE_BOOK`)
- Syzygy TBs — optional, `setoption name SyzygyPath value <path>`

## Step 7 — Validation ✓

Per-edit loop used throughout:
1. `tools\make-win.bat` — Cython build green (`-Werror` gate).
2. `python tools\make-native.py` — native target green.

End-to-end validation of native `sturddle-2.5.1-hnat.exe`:

| Test | Result |
|------|--------|
| Perft 6 from startpos | **119,060,324** ✓ (matches reference) |
| `go depth 10` — native vs. `.pyd` | cp 26, 105,979 nodes, same PV ✓ |
| `go depth 11` — native vs. `.pyd` | cp 26, 145,598 nodes, same PV ✓ |
| `go depth 12` — native vs. `.pyd` | cp 18, 261,747 nodes, same PV ✓ |
| `test/unit_test.py` (EPD + all unit tests) | all passed ✓ |
| Cython `make-win.bat` | still green ✓ |

Bit-for-bit identical search output at matching depths. Engine is ready
for GUI integration and tournament play.

**Lesson from validation:** `main.py`'s `load_engine()` auto-selects the
best CPU-arch `chess_engine_avx*.pyd`. Rebuilding only the baseline via
`make-win.bat` leaves arch-specific variants stale — `python main.py`
silently loads the old one, leading to spurious "divergence" vs. the
native exe. For parity tests against the current source, bypass
`main.py`:

```
python -c "import sys; sys.path.insert(0,'.'); import chess_engine; chess_engine.uci('Sturddle', debug=False, dev_mode=False)"
```

## Step 8 — Linux build ✓

`build_linux()` in `make-native.py` mirrors `setup.py:230–295`:

- **Compiler via `CXX` env var**, no auto-detection. Required.
  Smoke-tested with `CXX=clang++-20`; gcc ≥ 13 branch wired but untested.
- **Version gate.** Parses the `N.N.N` from `$CXX --version`; rejects
  clang < 16 / gcc < 13, matching `MIN_CLANG_VER` / `MIN_GCC_VER`.
- **Target triplet via `$CXX -dumpmachine`.** Feeds the `-L/usr/lib/llvm-<ver>/lib/<triplet>`
  secondary libdir. Previously hardcoded to `x86_64-pc-linux-gnu` in both
  `make-native.py` and `setup.py`; fixed in both files so ARM/other
  Linux targets work without editing the scripts.
- **Clang path adds:** `-stdlib=libc++ -fexperimental-library`
  + `-fuse-ld=lld -L/usr/lib/llvm-<ver>/lib[/<triplet>]`
  `-L/usr/local/opt/llvm/lib/c++ -lc++ -lc++experimental`.
- **`-ffast-math` instead of `-Ofast`.** Clang 20 errors on `-Ofast`
  under `-Werror`; `-O3 -ffast-math` is the documented replacement and
  matches the Windows `/fp:fast` intent. **Caveat:** the Cython Linux
  build does *not* enable `-ffast-math`, so Linux-native eval may diverge
  bit-for-bit from the Cython Linux `.so`; Windows native remains parity
  against Windows `.pyd`.
- **`context.cpp:segv_handler` gated.** The linux-only `PyErr_SetString`
  call needed `#if !NATIVE_BUILD`; the `dump_backtrace` + `sigaction`
  wiring stays active in both builds (a native crash still dumps).
- **Output** `dist/native/sturddle-<version>[-arch]` — no `.exe`.
- Stack size: default 8 MB main was sufficient for depth 10 parity runs;
  no `pthread_attr_setstacksize` needed so far.

Smoke-tested: `uci`, `isready`, `position startpos`, `go depth 10` all
green; weights.bin loads from exe dir, AVX512/FMA path selected on this
host.

## Step 9 — Native log-level filter ✓

Before this step, `native_log_message()` in `context.cpp` printed every
level unconditionally — `[DEBUG]` traces (init banners, move-ordering
dumps, etc.) leaked to stderr even in non-verbose runs.

New state:

- **`search::native_log_level`** (new `extern` in `context.h`, default
  `LogLevel::INFO`). `native_log_message()` drops messages below
  threshold unless the existing `force` flag is set — same semantics
  as the Cython `forceLevel` param.
- **Single sync point:** `uci_native.cpp::sync_native_log_level()`
  mirrors the file-static `_debug` bool into `native_log_level`
  (`DEBUG` ↔ `INFO`). Called from `uci_loop` startup (reflects
  `params["debug"]` / `-v`) and from the `Debug` UCI option via a
  thin `OptionDebug` subclass of `OptionBool` that chains
  `OptionBool::set()` + `sync_native_log_level()`.
- **Cython build unaffected.** The sync helper is `#if NATIVE_BUILD`
  internally; `OptionDebug` compiles in both builds and reduces to a
  no-op wrapper when `NATIVE_BUILD` is undefined. Python-side logging
  continues to manage levels via `logging.getLogger().setLevel()`.

Validated: `go depth 6` without `-v` emits only `[INFO]/[WARN]/[ERROR]`;
`-v` or `setoption name Debug value true` (under `-D`) re-enables
`[DEBUG]` lines; toggling back to `false` immediately silences them.

## Remaining work (not in this milestone)

- **Per-arch variants tested.** Infrastructure is in place
  (`make-native.py AVX2` etc.) but only the `native` arch has been
  smoke-tested end-to-end on this machine.
- **Embed via `#embed` ✓.** `make-native.py --embed` now uses C23/C++26
  `#embed "weights.bin"` in `context.cpp`; no `weights.h`, no TF
  dependency at build time. Requires GCC 15+ / Clang 19+ / MSVC 17.15+.
  Legacy `weights.h` path preserved under `-DUSE_WEIGHTS_H` for debug.
- **Cleanup of unused callbacks in `Context`.** `_pgn`, `_print_state`,
  `_report`, `_engine`, `_book_init`, `_book_lookup`, `_on_next` could
  be gated out entirely once the Cython path is retired (if ever).
- **`sturddle.cfg` replacement.** Currently loaded by Python `exec()`
  in `__init__.pyx:1231`. Every tunable is already exposed via UCI
  `setoption` so no C++ parser is strictly needed for native operation.
  Low priority.

## Actual scope

| Task                                        | Commit     |
|---------------------------------------------|------------|
| Native EPD serializer + cross-check tests   | `fe57881`  |
| `cython_wrapper` shim                       | `018943f`  |
| `Python.h` gating + default callback sinks  | `dce7a35`  |
| `version.h` + `main_native.cpp`             | `ab5accb`  |
| `make-native.py` parallel build + io.h fix  | `12771ce`  |
| Linux build + `segv_handler` gating + triplet fix | `1e24a89` |
| Native log-level filter                     | `1e24a89`  |
| Plan doc                                    | `19231ae`  |

~420 lines added across C++ (epd serializer, shim, main, defaults),
Cython (version bindings), and Python (build script + tests). Core
search/eval/move-gen untouched.
