#!/usr/bin/env python3
"""
Verify NNUE binary weights for proper clipping and rounding (on-grid, i.e. exported with -q).
Architecture: perspective accumulator (16 buckets x 768 -> 1024, shared by both views),
[black, white] -> 2048, two side-to-move stacks 2048 -> 32 -> 32 -> 1.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fetch_weights

Q_SCALE = 1024
Q_MAX_A = 992 / Q_SCALE  # accumulator: 32 pieces + bias == 33 terms fit int16

Q_W2 = 64  # hidden_2 s8 weights
Q_MAX_W2 = 127 / Q_W2
Q_B2 = 128 * Q_W2  # hidden_2 int32 bias, at activation scale x weight scale
Q_MAX_B2 = (2**31 - 1) / Q_B2

Q_MAX_MOVE = 32767 / Q_SCALE / 34  # move head: 32 pieces + side-to-move + bias == 34 terms

ACTIVE_INPUTS = 768
ACCUMULATOR_SIZE = 1024
MAIN_BUCKETS = 16
STACKS = 2  # black to move first
HIDDEN_2 = 32
HIDDEN_3 = 32
MOVE_INPUTS = 769
MOVE_ACCUMULATOR_SIZE = 256
MOVE_OUTPUTS = 4096  # 64x64 (from, to)

# constraint: ((kernel scale, kernel max), (bias scale, bias max)), or None for float layers
ACC = ((Q_SCALE, Q_MAX_A), (Q_SCALE, Q_MAX_A))
L2 = ((Q_W2, Q_MAX_W2), (Q_B2, Q_MAX_B2))
MOVE = ((Q_SCALE, Q_MAX_MOVE), (Q_SCALE, Q_MAX_MOVE))

# Layer definitions: (name, kernel_shape, bias_shape, constraint)
# ORDER MATTERS - must match export order from trainer
LAYERS = [('hidden_1a', (ACTIVE_INPUTS * MAIN_BUCKETS, ACCUMULATOR_SIZE), (ACCUMULATOR_SIZE,), ACC)] + [
    layer
    for s in range(STACKS)
    for layer in (
        (f'hidden_2_{s}', (2 * ACCUMULATOR_SIZE, HIDDEN_2), (HIDDEN_2,), L2),
        (f'hidden_3_{s}', (HIDDEN_2, HIDDEN_3), (HIDDEN_3,), None),
        (f'out_{s}', (HIDDEN_3, 1), (1,), None),
    )
]

# Optional move prediction head: own sub-accumulator, decoupled from eval
MOVE_LAYERS = [
    ('move_acc', (MOVE_INPUTS, MOVE_ACCUMULATOR_SIZE), (MOVE_ACCUMULATOR_SIZE,), MOVE),
    ('move', (MOVE_ACCUMULATOR_SIZE, MOVE_OUTPUTS), (MOVE_OUTPUTS,), MOVE),
]


def check_clipping(weights, qmax, layer_name, weight_type):
    """Check if all values are within [-qmax, qmax]."""
    violations = np.abs(weights) > qmax
    count = np.sum(violations)
    if count > 0:
        worst = np.max(np.abs(weights))
        idx = np.where(violations.flat)[0][:5]
        print(f"  CLIP VIOLATION in {layer_name} {weight_type}: {count} values outside [-{qmax:.10f}, {qmax:.10f}]")
        print(f"    Worst value: {worst:.10f}")
        print(f"    Examples at indices {idx}: {[weights.flat[i] for i in idx]}")
        return count
    return 0


def check_rounding(weights, qscale, qmax, layer_name, weight_type):
    """Check if all values are multiples of 1/qscale (except clamp boundaries that are off the grid)."""
    scaled = weights.astype(np.float64) * qscale
    violations = scaled != np.round(scaled)
    # an off-grid clamp (move head) leaves boundary values unrounded
    at_boundary = np.abs(weights) >= qmax * 0.9999
    violations = violations & ~at_boundary

    count = np.sum(violations)
    if count > 0:
        idx = np.where(violations.flat)[0][:5]
        print(f"  ROUND VIOLATION in {layer_name} {weight_type}: {count} values not multiples of 1/{qscale}")
        for i in idx:
            w = weights.flat[i]
            r = np.round(scaled.flat[i]) / qscale
            print(f"    [{i}] {w:.10f} nearest {r:.10f}, diff={w-r:.2e}")
        return count

    boundary_count = np.sum(at_boundary & (scaled != np.round(scaled)))
    if boundary_count > 0:
        print(f"  (note: {boundary_count} values at off-grid boundary ±{qmax:.10f}, rounding not checked)")

    return 0


def verify_layers(data, layers, offset=0):
    """Verify a list of layers starting at given offset. Returns new offset and violation counts."""
    total_clip_violations = 0
    total_round_violations = 0

    for layer_name, kernel_shape, bias_shape, constraint in layers:
        kernel_size = np.prod(kernel_shape)
        bias_size = np.prod(bias_shape)
        total_size = kernel_size + bias_size

        # Check if we have enough data
        if offset + total_size > len(data):
            return offset, total_clip_violations, total_round_violations, False

        kernel = data[offset:offset + kernel_size].reshape(kernel_shape)
        offset += kernel_size

        bias = data[offset:offset + bias_size].reshape(bias_shape)
        offset += bias_size

        print(f"{layer_name}: kernel {kernel_shape}, bias {bias_shape}")

        if constraint is None:
            print(f"  (float, no constraint)")
            continue

        for values, (qscale, qmax), weight_type in zip((kernel, bias), constraint, ("kernel", "bias")):
            total_clip_violations += check_clipping(values, qmax, layer_name, weight_type)
            total_round_violations += check_rounding(values, qscale, qmax, layer_name, weight_type)

    return offset, total_clip_violations, total_round_violations, True


def main():
    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]} [weights.bin]")
        sys.exit(1)

    filepath = sys.argv[1] if len(sys.argv) == 2 else str(fetch_weights.ensure())
    print(f"Loading: {filepath}")
    print(f"hidden_1a: 1/{Q_SCALE}, max {Q_MAX_A:.10f}")
    print(f"hidden_2:  1/{Q_W2} weights, max {Q_MAX_W2:.10f}; 1/{Q_B2} biases")
    print(f"move head: 1/{Q_SCALE}, max {Q_MAX_MOVE:.10f}")
    print()

    data = np.fromfile(filepath, dtype=np.float32)
    print(f"Total values: {len(data)}")

    base_total = sum(np.prod(k) + np.prod(b) for _, k, b, _ in LAYERS)
    move_total = sum(np.prod(k) + np.prod(b) for _, k, b, _ in MOVE_LAYERS)

    print(f"Expected (without move): {base_total}")
    print(f"Expected (with move): {base_total + move_total}")

    has_move_layer = len(data) == base_total + move_total

    if len(data) == base_total:
        print("Detected: model WITHOUT move prediction")
    elif has_move_layer:
        print("Detected: model WITH move prediction")
    else:
        print(f"ERROR: Size mismatch! Got {len(data)}, expected {base_total} or {base_total + move_total}")
        sys.exit(1)

    print()

    # Verify base layers
    offset, total_clip_violations, total_round_violations, success = verify_layers(data, LAYERS)

    if not success:
        print("ERROR: Unexpected end of data while reading base layers")
        sys.exit(1)

    # Verify move head if present
    if has_move_layer:
        offset, clip_v, round_v, success = verify_layers(data, MOVE_LAYERS, offset)
        total_clip_violations += clip_v
        total_round_violations += round_v

        if not success:
            print("ERROR: Unexpected end of data while reading move head")
            sys.exit(1)

    # Verify we consumed all data
    if offset != len(data):
        print(f"WARNING: {len(data) - offset} values remaining after parsing")

    print()
    print("=" * 60)
    if total_clip_violations == 0 and total_round_violations == 0:
        print("All constraints satisfied!")
    else:
        print(f"Total clipping violations: {total_clip_violations}")
        print(f"Total rounding violations: {total_round_violations}")
        sys.exit(1)


if __name__ == '__main__':
    main()
