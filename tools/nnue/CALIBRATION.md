# NNUE Data Calibration Manual

Two independent calibrations, both per-bucket, both stored in `<member>.h5.profile.json`
sidecars and applied per-row by the trainers:

1. **eval label scale** (`ratios`) — normalizes each source's centipawn labels to engine scale.
2. **outcome scale** (`outcome_scale`) — the S in `P(win) = sigmoid(cp / S)` used by the blend loss.

16 buckets = 4 pawn bands (0-4 / 5-8 / 9-12 / 13-16) x 4 king-file configs (QQ/QK/KQ/KK).

---

## Tools

| tool | purpose |
|------|---------|
| `label_check.py`   | UCI-engine referee: per-bucket ratio (label/engine) + corr, per file |
| `scale_groups.py`  | split a file list into scale-consistent groups from label_check reports |
| `bucket_report.py` | per-bucket diet: zero% / mid% / ext% / draw% (label quality, not scale) |
| `optmix.py`        | greedy mix builder (`--density` = gain-per-position) under a row budget |
| `mkprofile.py`     | build `ratios` from label_check reports |
| `outcome_scale.py` | fit S; `--update-profile` writes `outcome_scale` into each member sidecar |

Sidecar precedence (both keys, per member): own sidecar -> container sidecar -> default
(`ratios`=all-ones, `outcome_scale`=`--outcome-scale`). Legacy sidecars lacking a key fall back.
VDS member paths are relative — run trainers from the data dir.

---

## Section A — eval label scale (`ratios`)

**Fact:** sources label centipawns on incompatible scales. Uncorrected, eval targets fight.

**Measure** (depth >= 16; shallow referee inflates endgame ratios ~2x):
```
./label_check.py all_files.txt -e <engine> --threads 8 --depth 20 > report.txt
```
Per-file table gives ratio = |label|/|engine| and corr per bucket.

**Verdicts observed:**
- ratio ~1, corr >0.95: hnat, ppst, guid sets, self-play, eval_before openings/elo-2400
- ratio 2-3.5, corr >0.9: leela pieces, test80/91, T80, farseer, OLD lichess
- corr < 0.7: BROKEN (ccrl, raw lichess A/B/elo-*, eval_before root) — exclude, no S fixes low corr
- test90: scale drifts BY DATE (<Oct13'25 ~0.1 junk; Oct14-19 ~1x; Oct22+ ~3x) — filter by filename

**Group + build:**
```
./scale_groups.py all_files.txt -r report.txt --split 1.5 --min-corr 0.8 --out-prefix scale
./optmix.py scale_1.txt --max-positions 2e9 --density --out mix_lo.txt
./mkprofile.py report.txt --match leela T80 farseer --out mix_hi.h5.profile.json
```
Rule: lo group (~1x) trains at ratios=1 (no sidecar needed). Hi group gets a `ratios` sidecar.

### Chesster (data machine games; /mnt/nvme/h5)
- `mix_new.h5` = `mix_hi_new` (3B, ratios 2.2-3.4) + `mix_lo_new` (2B, ratios=1)

### Dragon (mixed sources; /mnt/b/Projects/train-2, list in all_files.txt)
- `dmix.h5` = `dmix_lo` (1.8B, ratios=1) + `dmix_leela` 2.5B + `dmix_farseer` 1B + `dmix_t80` 1B + `ppst4`
- each hi member has its own `ratios` sidecar (per-family; ratios 1.8-3.7)
- excluded: h5-plus-200 (corr 0.65-0.8), ccrl, raw lichess, eval_before root

---

## Section B — outcome scale (S)

**Fact:** S is a per-source property (outcome noise), NOT a universal 400.
`P(win) = sigmoid(cp / S)`. Small S = steep = small eval already implies near-certain result.

- engine self-play converts reliably -> S ~ 110-140
- human games (lichess) blunder -> S ~ 390 (this is where the 400 convention came from)

**How the fit works:** logistic regression of outcome (0/0.5/1) on profile-corrected cp.
Minimizes binary cross-entropy over beta = 1/S; Newton/IRLS, convex, few iterations.
Scale-equivariant per bucket, so ratios must be applied first only for the pooled/overall stats.

**Measure + write** (fits each VDS member separately, writes its sidecar):
```
./outcome_scale.py dmix.h5 --sample 0.05 --update-profile
```
`--raw` ignores ratios (shows the human ~400 vs engine ~120 gap). `--min-n` large -> per-member scalar.
Trust the empirical calibration table (actual score% per eval bin vs fitted) over the raw number.

### Chesster
- all machine games -> S ~ 110-120 uniformly. `--outcome-scale 115` was fine even without sidecar.

### Dragon (measured, corrected domain)
```
member          overall S
dmix_lo            130.8
dmix_leela         114.4
dmix_farseer       121.8
dmix_t80           104.6
ppst4              136.7
overall            120.8
```
Per-bucket within a member spans ~95-145 (full-pawn buckets need larger S than endgames).
Empirical curve tracked the S=120 fit within ~2 points at every eval bin — the number is real.

---

## Loss functions

Eval target = profile-corrected cp. Outcome target = 0/0.5/1. The flag decides how each
is compared to the prediction.

| flag | eval term | outcome term | keeps eval magnitude? |
|------|-----------|--------------|-----------------------|
| `--loss-blend` | Huber on raw cp | BCE in prob space | YES (centipawn space) |
| (default) mse  | MSE in prob space | MSE in prob space | no (sigmoid squashes) |
| `--loss-bce`   | BCE in prob space | BCE in prob space | no (sigmoid squashes) |
| `--loss-mae`   | MAE in prob space | MAE in prob space | no |

**blend** — eval loss lives in cp space, so the net learns +300 > +150 (magnitude), not just
"both winning". Huber caps extreme-label pull. Best when eval labels are trusted + graded.
**mse / bce** — squash through sigmoid first, so large evals compress (+800 ~ +1500 ~ 0.99);
loses endgame magnitude resolution. Safe, hard to break; correct when label scale is untrusted.
bce penalizes confident-wrong harder than mse. **mae** — outlier-robust but weak gradient near
target; generally weakest for final strength, occasional noise-robust sanity run.

**Choosing:**
- labels hot / mixed-scale / untrusted -> mse or bce (magnitude is garbage, don't fit it)
- labels clean + graded (post-calibration state) -> blend, to exploit magnitude
- endgame drift after a blend run -> rerun mse to isolate loss vs data
- outcome-only data (no evals) -> bce with `--outcome-weight 1.0`

blend is the recommendation *because* the calibration work is done; before it, mse was the safer
pick. Ultimate answer is Elo.

**Scale mechanics per regime** (pred side of the sigmoid):
- blend: per-row sidecar S — its outcome nudge must land on the same cp scale as the Huber term.
- mse/mae/bce: global `--outcome-scale` for the prediction (defines the net's output cp units,
  keep ~120 = engine scale), per-row sidecar S for the eval target (each source's own
  eval->prob dictionary). One scale-fixing mechanism per regime: blend uses ratios,
  prob-space losses use per-member S.

---

## Alternative regime: S-only (no eval ratios)

Idea: skip ratio calibration entirely; fit per-member S on RAW evals (all-ones ratios) and train
mse. `sigmoid(eval/S)` translates every source into probability space directly — a hot member
just gets a proportionally bigger S. Setup: `profiles.py backup`, move sidecars away,
`outcome_scale.py --update-profile` (writes ratios=1 + raw-domain S), train mse with
`--outcome-scale` set to the net's intended output scale.

Pros:
- no referee engine in the scale estimate (S fits from the data's own eval/outcome pairs);
  referee depth/bias errors vanish — the referee survives only in the binary corr-exclusion gate
- inter-bucket ratio differences that are really probability-evolution-by-phase stop being
  "corrected" away

Cons:
- magnitude blindness: +400 and +900 both map to ~1.0 — loses the winning-vs-crushing signal
  blend's Huber preserves (endgame technique)
- S conflates label scale with outcome noise: good labels + blundery outcomes (human data,
  fishnet endgames) -> big S -> flattened eval targets; good labels get diluted
- new failure mode: ratios needed only labels, S needs outcomes that correlate — a source with
  broken outcomes gets a garbage S even when labels are fine
- near-saturation rows contribute ~zero gradient (mild data waste)

Net trade: swaps referee error for outcome-noise sensitivity, paid in magnitude resolution.
Untested as of July 2026; A/B against ratio+blend by match.

---

## Training settings (post-calibration)

- `--loss-blend` (Huber eval + BCE outcome) — safe now that scales are corrected; huber delta 2.0
- `--outcome-weight 0.4` (was masking miscalibration at 0.35; 0.5 defensible now, ceiling ~0.6)
- `--outcome-scale` becomes the fallback only; sidecar `outcome_scale` takes precedence per row
- `--filter 1260` masks RAW (pre-scaling) labels — hi members effectively cut at ~450 engine-cp
- restart from checkpoint (`--import-file`) after writing sidecars — no CLI change needed

---

## Move-prediction head (TF trainer, experimental)

Facts (source of truth: context.cpp, nnue.h):
- `USE_MOVE_PREDICTION` gates it; 2.5.0 ships it, 2.6.0 dev does not. Cost is negligible.
- Used only at early iterations (`iteration() <= MOVE_PREDICTION_MAX_ITER`) to seed move
  ordering before history stabilizes; deeper plies use history+eval.
- Inference `score_move` (nnue.h) is ALREADY sparse: per move it gathers one column
  `W[:, from*64+to]` and dots with active inputs. Never materializes 4096. So output width
  is NOT an inference cost — shrinking it saves only training + capacity.
- Head is a single linear `Dense(4096)` off `stop_gradient(raw inputs)`. Coupling it to the
  eval trunk was tried and back-spilled into eval badly — hence the stop_gradient. KEEP DECOUPLED.

Improvement direction (architecture): give the head its own incrementally-updated accumulator,
nested inside the eval `Accumulator` struct under `#if USE_MOVE_PREDICTION` (shares add/sub call
sites, independent weights, no eval back-spill). Per-move score stays a cheap dot vs that
sub-accumulator (sparsity is move-specific: ~35 real moves out of 4096; eval can't use the trick,
its output is width-1 dense).

Improvement direction (training speed): loss today is full 4096 softmax
(`sparse_categorical_crossentropy`, "sparse" = integer label only, still normalizes over all 4096)
— that's the expense, even with `--freeze-eval` (freezing removes trunk cost, not head cost).
Fix = sampled loss over played move + K negatives (random or in-batch; engine pseudolegals are a
wash since data has only the played move). ~64 logits vs 4096.

Plan: do this in the TF trainer (head wiring + `--alt-model`/`--import` machinery already there),
import eval weights.bin from the torch side, `--freeze-eval`, train head only. Port to torch only
if the head graduates to permanent.

**BEFORE TESTING — round-trip sanity check** (torch->TF weight import was hardened but not run):
1. torch `train_torch.py -o weights.bin` (export eval)
2. TF `train.py <data> --import-file weights.bin -o roundtrip.bin` (no `--predict-moves`)
3. diff the two .bin files (or compare eval output on a few FENs) — must match.
`load_binary_weights` now matches eval layers by name (hidden_1a..out) and validates total length,
so a `--predict-moves` graph importing an eval-only file leaves the head untouched. Verify anyway.

---

## End-to-end recipe (fresh dataset)

```
1. label_check.py all_files.txt -e <engine> -d 20 > report.txt
2. scale_groups.py all_files.txt -r report.txt --split 1.5 --min-corr 0.8 --out-prefix scale
3. (date-filter test90; drop corr<0.8 bins)
4. bucket_report.py --mid 50 500 --extreme 1260 --sample 0.02 $(cat scale_1.txt)   # diet sanity
5. optmix.py scale_N.txt --max-positions <budget> --density --out mix_N.txt   # per family for hi
6. mkprofile.py report.txt --match <family> --out mix_N.h5.profile.json          # ratios
7. h5cat.py <members> -o mix.h5                                                   # nested VDS
8. outcome_scale.py mix.h5 --sample 0.05 --update-profile                         # outcome_scale
9. train_torch.py mix.h5 ... --loss-blend --outcome-weight 0.4                    # from data dir
```
