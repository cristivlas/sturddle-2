#!/usr/bin/env bash
set -euo pipefail

die() { echo "$(basename "$0"): $*" >&2; exit 1; }

for v in WEIGHTS_HOST WEIGHTS_PATH WEIGHTS_FILE; do
    [[ -n ${!v:-} ]] || die "$v not set"
done
[[ $WEIGHTS_FILE != weights.bin ]] || die "WEIGHTS_FILE cannot be weights.bin (export target)"

trap 'die "failed at line $LINENO: $BASH_COMMAND"' ERR

# repo root is two levels up from this script; outputs land there
cd "$(dirname "$(realpath "$0")")/../.."

command -v python >/dev/null || die "python not found (venv not active?)"
python -c 'import torch, numpy, h5py, chess' 2>/dev/null \
    || die "missing python deps: pip install torch numpy h5py chess tqdm"

scp ${WEIGHTS_PORT:+-P"$WEIGHTS_PORT"} "$WEIGHTS_HOST:$WEIGHTS_PATH/$WEIGHTS_FILE" "$WEIGHTS_FILE"
[[ -s $WEIGHTS_FILE ]] || die "copy of $WEIGHTS_HOST:$WEIGHTS_PATH/$WEIGHTS_FILE missing or empty"

python tools/nnue/train_torch.py --import "$WEIGHTS_FILE" --export weights.bin --quant
[[ -s weights.bin ]] || die "weights.bin not produced"

python tools/nnue/pack_weights.py
python tools/nnue/golds.py weights.bin

echo "$(basename "$0"): done"
