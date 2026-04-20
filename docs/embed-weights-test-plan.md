# Embedding NNUE weights — test plan

How to verify the `--embed` flag of `tools/make-native.py`. `--embed`
relies on C23/C++26 `#embed` — no TensorFlow, no `weights.h`
generation step.

## Prerequisite

- GCC 15+, Clang 19+, or MSVC 17.15+ (for `#embed` support).
- `weights.bin.gz` + `weights.bin.sha256` present at repo root (tracked).
  `tools/fetch_weights.py` decompresses as needed; `make-native.py` and
  `setup.py` call it automatically.

## 1. Argparse sanity (instant)

```
python tools/make-native.py --help
```

`--embed` is a bare boolean flag (no MODEL argument).

## 2. Default path regression check (~10 s)

Non-embed path must not have regressed.

```
python tools/make-native.py
ls dist/native/
```

Expect: `sturddle-<ver>.exe`, `weights.bin`, `book.bin`, the
`*-sha256.txt` digest. Compile commandline should contain
`-DSHARED_WEIGHTS`.

## 3. Embed build (fast)

```
python tools/make-native.py --embed
```

What to expect:

1. `fetch_weights.ensure()` runs first; decompresses `weights.bin.gz`
   if `weights.bin` is missing or sha256 mismatches.
2. Parallel compile of all TUs. `context.cpp` with `#embed "weights.bin"`
   is near-instant — no ~100 MB text header.
3. Link. Final exe size reflects the 36 MB weights blob.

Quick validations after:

```
ls -la dist/native/sturddle-<ver>.exe     # ~35 MB baseline + weights
ls dist/native/weights.bin                # should FAIL — not copied when embedded
```

## 4. Smoke test the embedded exe (seconds)

Run from a directory **without** `weights.bin` next to it:

```
cd /tmp
cp <repo>/dist/native/sturddle-<ver>.exe .
printf "uci\nquit\n" | ./sturddle-<ver>.exe
```

Expect normal UCI banner and `uciok`. A weights-loading error means
embed wiring is broken — check that `-DSHARED_WEIGHTS` was **not** on
the compile line.

## 5. Search parity vs. SHARED_WEIGHTS build

Same `weights.bin` → identical search output.

```
python tools/make-native.py AVX2
mv dist/native/sturddle-<ver>-avx2.exe /tmp/shared-avx2.exe

python tools/make-native.py AVX2 --embed
mv dist/native/sturddle-<ver>-avx2.exe /tmp/embed-avx2.exe

cd /tmp
cp <repo>/weights.bin . 2>/dev/null || true
for exe in shared-avx2 embed-avx2; do
  printf "position startpos\ngo depth 12\nquit\n" | ./$exe.exe 2>/dev/null \
    | grep "^info score.* depth 12 " > $exe.log
done
diff /tmp/shared-avx2.log /tmp/embed-avx2.log
```

Expect empty diff.

## Debug fallback: `weights.h`

`tools/nnue/train.py -o weights.h` still exports a constexpr-arrays
header. Build with `-DUSE_WEIGHTS_H` (and without `-DSHARED_WEIGHTS`)
to bypass `#embed` entirely. Useful for toolchains without `#embed`
or for comparing the embedded blob against named-layer arrays.

## Failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `#error: embedded build requires #embed support` | Toolchain too old | Upgrade to GCC 15+/Clang 19+/MSVC 17.15+, or use `-DUSE_WEIGHTS_H` |
| `#error: weights.bin not found` | `fetch_weights.ensure()` didn't run | Run `python tools/fetch_weights.py` manually |
| `ERROR: decompressed sha256 mismatch` | Corrupted `weights.bin.gz` or wrong `weights.bin.sha256` | Re-export from training, regenerate both |
| Exe size barely grew | `-DSHARED_WEIGHTS` wasn't dropped | Inspect compile commandline |
