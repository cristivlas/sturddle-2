# Native Executable Build — Plan of Action

## Goal

Build Sturddle as a standalone native executable (no Python runtime, no Cython)
on both Linux and Windows, reusing the existing `uci_native.cpp` UCI loop.

## Guiding decisions

- **Windows first, clang-cl only.** The `sturddle.exe` target is built with
  `clang-cl.exe` (LLVM), the same toolchain `tools/make-win.bat` already uses
  for the Cython `.pyd`. MSVC `cl.exe` is **not** a supported compiler for the
  native build — the codebase uses clang-specific flags (`-march=native`,
  `-Ofast`, `-Wno-nan-infinity-disabled`) and libc++-style constructs that
  pre-date MSVC's C++20 `<format>` maturity. Linux and macOS follow once the
  Windows binary is validated.
- **Keep the Python/Cython build path working.** All changes are gated by a new
  `NATIVE_BUILD` define. `setup.py` continues to produce `chess_engine*.pyd`
  from the same source tree, and `tools/make-win.bat` is the primary validator
  after each edit (`-Werror` catches breakage in both build modes).
- **Bootstrap with `SHARED_WEIGHTS=1`.** This avoids the dependency on a
  generated `weights.h`. The executable loads `weights.bin` from its own
  directory at startup. Regenerating `weights.h` from the TF model via
  `tools/nnue/train.py` is a later, separate milestone.
- **Minimum viable UCI first.** Drop everything that's not on the native UCI
  hot path (PGN export, SMP thread reports, `vmem_avail`, Python-side book
  lookup). `NATIVE_BOOK` is already the default.

## Prerequisite: weights.bin

`SHARED_WEIGHTS=1` makes `Context::init()` call `model.load_weights(exe_dir / "weights.bin")`.
A `weights.bin` already ships in the repo root; for the initial build, copy it next
to the executable. Long-term, `tools/nnue/train.py` exports this file from the
TF model.

Compiled-in path (`SHARED_WEIGHTS=0`) is a follow-up that requires regenerating
`weights.h` from the current TF model — out of scope for this first milestone.

## Step 1 — Native EPD serializer

`Context::epd()` currently delegates to Python (`context.cpp:1521`). Used by the
native UCI loop for `debug` output, `LOG_DEBUG` traces, and book-move logging.

- Add `chess::epd::to_string(const State&)` in `chess.h` next to the existing
  `parse_pos`/`parse_side_to_move`/`parse_castling`/`parse_en_passant_target`
  (chess.h:2092+).
- FEN fields: piece placement, side to move, castling rights, en-passant
  square, halfmove clock, fullmove number. Halfmove/fullmove come from
  `Context::_history->_fifty` and `_history->size()` respectively (the Python
  version reads them off `chess.Board`).
- Under `NATIVE_BUILD`, `Context::epd()` calls the native serializer directly
  instead of `cython_wrapper::call(_epd, state)`.

## Step 2 — No-op `cython_wrapper`

`utility.h` currently includes `Python.h` unconditionally. Under `NATIVE_BUILD`:

- Skip `#include "Python.h"`.
- `GIL_State` becomes an empty struct.
- `call(fn, args...)` becomes a direct invocation (null-check on `fn`).
- `call_nogil(fn, args...)` becomes a `try/catch` that logs to stderr on
  exception (no `PyErr_SetString`).
- `cancel_search(CancelReason)` — leave the signature, but the
  `PY_ERROR`/`PY_SIGNAL` reasons will never fire in native mode.

Affected call sites (unchanged source, just swap the wrapper):
`context.h:1152,1684`; `context.cpp:69,433,716,1521`; `search.cpp:85,1362,1387`;
`search.h:80,84,88`.

## Step 3 — Gate Python includes and Python-only callbacks

- `context.h:30` — wrap `#include "Python.h"` in `#ifndef NATIVE_BUILD`. Add a
  forward-declaration of `PyObject` (as an opaque struct) for native builds, so
  the existing `static PyObject* _engine` and callback signatures still compile.
- `context.h:478–490` — keep the callback table as-is. In native mode the
  unused pointers (`_book_init`, `_book_lookup`, `_pgn`, `_print_state`,
  `_report`, `_on_next`, `_vmem_avail`) stay `nullptr` and the call sites
  already null-check them (verify `context.h:1151` and `search.cpp:1362`).
- `context.cpp:66–84` — replace `PyErr_SetString` in the SEGV handler with
  `_exit(-1)` after `dump_backtrace(std::cerr)`.

## Step 4 — Logger

`Context::log_message` (`context.cpp:714`) routes to Python `logging`. Under
`NATIVE_BUILD`:

- Write directly to `stderr` with a level prefix, or
- Append to a log file opened once (path via env var or `--log` arg).

The `_log_message` function-pointer indirection can go away in native mode,
but it's simpler to set it to an internal C++ sink during `Context::init`.

## Step 5 — `main()` entry point

New file `main_native.cpp`:

- Parse argv: `--name`, `--version`, `--debug`, `--dev`, optional
  `--weights=<path>`, `--log=<path>`.
- Populate the `params` `unordered_map` that `uci_loop()` expects
  (`uci_native.cpp:1470`).
- Call `search::Context::init(exe_dir)` before `uci_loop(params)`.
- Return the exit code from `uci_loop`.

Version string: reuse the build-stamp approach from `setup.py`
(`-DBUILD_STAMP=MMDDYY`), drop the Cython-side version assembly.

## Step 6 — Build system

**No CMake.** Mirror the existing `tools/make-win.bat` / `tools/make` idiom —
a thin shell script that invokes the compiler directly. The native target has
six translation units; a build-system dependency would be out of character for
this repo and adds an install requirement developers don't currently have.

### `tools/make-win-native.bat` (Windows, first milestone)

Invokes `clang-cl.exe` directly — no MSVC fallback, no `vcvarsall.bat` dance.
If LLVM isn't on the developer's machine at the expected path, the build
fails fast with a clear error, matching the existing `make-win.bat` behavior.

```bat
@setlocal
set CL_EXE=C:\Program Files\LLVM\bin\clang-cl.exe
set OUT=dist\native
if not exist %OUT% mkdir %OUT%

set SOURCES=chess.cpp context.cpp search.cpp uci_native.cpp tbprobe.cpp main_native.cpp

set DEFINES=-DNATIVE_BUILD=1 -DNATIVE_UCI=1 -DNATIVE_BOOK=1 -DWITH_NNUE ^
            -DSHARED_WEIGHTS -DCALLBACK_PERIOD=8192 -DNO_ASSERT

set INCLUDES=-I. -I./libpopcnt -I./magic-bits/include -I./version2 -I./Fathom/src

set CXXFLAGS=/std:c++20 /fp:fast /O2 /MP -march=native -Werror ^
             -Wno-deprecated-declarations -Wno-unused-command-line-argument ^
             -Wno-unused-label -Wno-unused-variable -Wno-nan-infinity-disabled ^
             /D_FORTIFY_SOURCE=0 /GS-

"%CL_EXE%" %CXXFLAGS% %DEFINES% %INCLUDES% %SOURCES% ^
    /link /OUT:%OUT%\sturddle.exe /LTCG:OFF
if errorlevel 1 exit /b 1

copy /Y weights.bin %OUT%\
copy /Y book.bin %OUT%\
@endlocal
```

`/MP` gives clang-cl intra-invocation parallelism. Full rebuild; incremental
builds can be added if compile time becomes painful.

### `tools/make-native` (Linux/Mac, follow-up milestone)

Equivalent shell script, mirroring `tools/make` — same define set, `-std=c++20`,
`-O3`, `-march=native`, clang-18 or gcc-13, `-stdlib=libc++` + `-fuse-ld=lld`
to match `setup.py:274–282`.

### Flag parity with `setup.py`

Risk: native script and `setup.py` drift over time. Accepted for v1 — the
native subset is much smaller (no Cython flags, no Python include dirs, no
`PyMODINIT_FUNC` macros). Drift is caught by running both `make-win.bat` and
`make-win-native.bat` after changes; `-Werror` is on in both.

If duplication becomes a maintenance issue, factor the flags into
`tools/compile_flags.py` that both scripts source. Deferred.

### Multi-arch variants

Cython build ships `chess_engine_avx*.pyd` via `tools/build.py`. Equivalent
native variants (`sturddle-avx2.exe`, etc.) follow the same pattern — loop over
arch names in the bat file. Not in this milestone. First target is a single
`sturddle.exe` built with `-march=native`.

## Step 7 — Runtime assets

Next to the executable:

- `weights.bin` — NNUE weights (required for `SHARED_WEIGHTS=1`).
- `book.bin` — opening book, if `OwnBook` is enabled (`NATIVE_BOOK` reads from
  `params["dir"] / "book.bin"`; set `dir` to `argv[0]`'s parent).
- Syzygy TBs — optional; path configured via UCI `setoption SyzygyPath`.

## Step 8 — Validation

Per-edit loop:

1. `tools\make-win.bat` — confirms Cython build still green (primary gate;
   `-Werror` catches any flag or header breakage).
2. `tools\make-win-native.bat` — confirms the native target also compiles.

End-to-end validation after all steps:

1. `go depth 12` from startpos: bestmove and node count from
   `dist\native\sturddle.exe` should match the current
   `chess_engine.cp312-win_amd64.pyd`-hosted run within noise.
2. UCI options round-trip: `setoption name Hash value 256`,
   `setoption name Threads value 4`, etc.
3. Perft regression: `go perft 6` from startpos against reference
   119,060,324.
4. Cutechess-cli sanity match: 100 games native vs. Python-hosted, same
   binary version — expect ~50% score.
5. Linux build (clang-18) — repeat 1–3.

## Step 9 — Cleanup / follow-ups (not in this milestone)

- Generate `weights.h` from the current TF model (`tools/nnue/train.py`) so
  `SHARED_WEIGHTS=0` is possible and we can ship a single-file executable.
- Replace `sturddle.cfg` (Python `exec()`-based) with either pure UCI
  `setoption` or a small INI parser. Low priority — every `DECLARE_VALUE`
  param is already exposed via `setoption`.
- Strip `PyObject*` from `Context` entirely once the Cython build path is
  retired (if ever).

## Scope estimate

| Task                          | LoC (new) | LoC (modified) |
|-------------------------------|-----------|----------------|
| Native EPD serializer         | ~60       | ~5             |
| No-op `cython_wrapper`        | ~25       | ~10            |
| Python.h gating + callbacks   | ~0        | ~15            |
| Logger                        | ~20       | ~5             |
| `main_native.cpp`             | ~80       | 0              |
| `CMakeLists.txt`              | ~120      | 0              |
| **Total**                     | **~305**  | **~35**        |

Risk: low. Core search/eval/move-gen is untouched; all changes are at the
Cython boundary or the UCI entry point.
