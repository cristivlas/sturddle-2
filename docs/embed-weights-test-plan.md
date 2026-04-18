# Embedding NNUE weights — test plan

How to verify the `--embed` flag of `tools/make-native.py`.

## Prerequisite

TensorFlow must be importable by `python tools/nnue/train.py`. Missing-TF
surfaces as an `ImportError` before any compilation starts — safe to
discover early.

## 1. Argparse sanity (instant)

```
python tools/make-native.py --help
```

Expect the `--embed [MODEL]` entry in the help text.

## 2. Default path regression check (~10 s)

Non-embed path must not have regressed.

```
python tools/make-native.py
ls dist/native/
```

Expect: `sturddle-2.5.1-hnat.exe`, `weights.bin`, `book.bin`, the
`*-sha256.txt` digest. The compile commandline should contain
`-DSHARED_WEIGHTS`.

## 3. First embed — triggers weights.h regeneration (multi-minute)

```
python tools/make-native.py --embed
```

What to expect, in order:

1. TensorFlow import + SavedModel load (~10–30 s).
2. `Regenerating weights.h from models/Raptor-III ...`
3. Per-layer shape lines from `train.py`.
4. ~100 MB `weights.h` written at the repo root.
5. Parallel compile of the 5 non-`context.cpp` TUs (seconds).
6. `context.cpp` is the critical path — parsing a ~100 MB header with
   clang-cl can take 30 s to several minutes and spikes RSS hard. Kill
   if it runs > 5 minutes or OOMs — at that point the approach needs
   rethinking (see **Failure modes**).
7. Link. Final exe size should jump from ~1.7 MB to ~35 MB.

Quick validations after:

```
wc -l weights.h                                      # ~300k lines expected
head -3 weights.h                                    # marker: "// Generated from models/Raptor-III"
ls -la dist/native/sturddle-2.5.1-hnat.exe          # ~35 MB
ls dist/native/weights.bin                          # should FAIL — not copied when embedded
```

## 4. Re-run hits the cache (instant regen-check)

```
python tools/make-native.py --embed
```

Expect `weights.h is up-to-date for models/Raptor-III` immediately, no
`train.py` invocation. Compile-only rebuild follows.

## 5. Force regen via different model string (multi-minute)

```
python tools/make-native.py --embed models/Raptor-III/
```

Trailing `/` makes the marker mismatch → regenerates. Exercises the
stale-check path on string equality.

## 6. Smoke test the embedded exe (seconds)

Run from a directory **without** `weights.bin` next to it:

```
cd /tmp
cp /c/Users/crist/Projects/sturddle-dev/dist/native/sturddle-2.5.1-hnat.exe .
printf "uci\nquit\n" | ./sturddle-2.5.1-hnat.exe
```

Expect the normal UCI banner and `uciok`. Crash with a weights-loading
error means embed wiring is broken — check that `-DSHARED_WEIGHTS` was
**not** on the compile line in step 3.

## 7. Search parity vs. SHARED_WEIGHTS build

Same model → identical search output. Any divergence at matched depth
means the two loaders produced different in-memory weights.

```
python tools/make-native.py AVX2
mv dist/native/sturddle-2.5.1-hnat-avx2.exe /tmp/shared-avx2.exe

python tools/make-native.py AVX2 --embed
mv dist/native/sturddle-2.5.1-hnat-avx2.exe /tmp/embed-avx2.exe

cd /tmp
cp /c/Users/crist/Projects/sturddle-dev/weights.bin . 2>/dev/null || true
for exe in shared-avx2 embed-avx2; do
  printf "position startpos\ngo depth 12\nquit\n" | ./$exe.exe 2>/dev/null \
    | grep "^info score.* depth 12 " > $exe.log
done
diff /tmp/shared-avx2.log /tmp/embed-avx2.log
```

Expect empty diff.

## Failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `ImportError: No module named tensorflow` | TF not installed | Install TF, or stay on SHARED_WEIGHTS |
| Model path not found | Bad `--embed <path>` | Check the directory exists |
| clang-cl hangs or RSS > 8 GB on `context.cpp` | Header too large for the toolchain | Kill; embed-weights approach not viable at this model size — consider binary embed (`.rc` on Windows, `.incbin` on GCC) |
| `fatal error C1060: compiler is out of heap space` | Same as above | Same as above |
| Exe size barely grew | `-DSHARED_WEIGHTS` wasn't dropped | Inspect compile commandline; should be absent when `--embed` is set |
