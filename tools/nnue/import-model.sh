#!/usr/bin/env bash
set -euo pipefail

# --- argument validation ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_path>" >&2
    exit 1
fi

model_path=$1

if [[ ! -d "${model_path}" ]]; then
    echo "Error: model path '${model_path}' does not exist or is not a directory." >&2
    exit 1
fi

model=$(basename "${model_path}")
target_path=__models__/${model}
weights=${model}.bin

echo "Importing: ${model_path}"
echo "       to: ${target_path}"
echo "  weights: ${weights}"

# --- dependency checks ---
for tool in ./tools/nnue/train.py ./tools/nnue/pack_weights.py; do
    if [[ ! -x "${tool}" ]]; then
        echo "Error: required tool '${tool}' not found or not executable." >&2
        exit 1
    fi
done

# --- remove existing target (note: was missing $ in original) ---
if [[ -e "${target_path}" ]]; then
    rm -rI "${target_path}" || { echo "Error: failed to remove '${target_path}'." >&2; exit 1; }
fi

# --- copy model ---
cp -r "${model_path}" "${target_path}" || { echo "Error: failed to copy '${model_path}' to '${target_path}'." >&2; exit 1; }

# --- verify ---
echo "Diffing models..."
diff -r "${model_path}" "${target_path}"
echo

# --- export quantized weights ---
echo "Exporting quantized weights..."
CUDA_VISIBLE_DEVICES=-1 ./tools/nnue/train.py export -m "${target_path}" --quantize --bin -o "${weights}" \
    || { echo "Error: weight export failed." >&2; exit 1; }

if [[ ! -f "${weights}" ]]; then
    echo "Error: expected weights file '${weights}' was not created." >&2
    exit 1
fi

# --- quantize model and re-export ---
echo
echo "Re-exporting with imported weights..."
CUDA_VISIBLE_DEVICES=-1 ./tools/nnue/train.py export -m "${target_path}" --import "${weights}" --bin -o weights.bin --save-model \
    || { echo "Error: quantized re-export failed." >&2; exit 1; }

# --- verify weights match ---
echo "Verifying weights..."
if ! diff weights.bin "${weights}"; then
    echo "Error: weights.bin differs from '${weights}' — outputs do not match." >&2
    exit 1
fi

# --- pack weights ---
echo "Packing weights..."
./tools/nnue/pack_weights.py || { echo "Error: pack_weights.py failed." >&2; exit 1; }

echo "Done."
