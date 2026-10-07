#!/usr/bin/env python3
"""
Verify NNUE binary weights for proper clipping and rounding.
Architecture: 2048-accumulator, hidden_1b + hidden_1c (linear) modulate pooled 1:1, 16-way bucketing.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fetch_weights

Q_SCALE = 1024

# Constraint A: for hidden_1a layer
Q_MAX_A = 32767 / Q_SCALE / 34

# Constraint B: for hidden_1b layer
Q_MAX_B = 32767 / Q_SCALE / 19

# Constraint C: for hidden_1c layer (32 occupied + up to 20 bishops + bias)
Q_MAX_C = 32767 / Q_SCALE / 53

ACTIVE_INPUTS = 769
ACCUMULATOR_SIZE = 2048
POOL_SIZE = 8
POOLED = ACCUMULATOR_SIZE // POOL_SIZE  # hidden_1b output width (modulates pooled 1:1)
MAIN_BUCKETS = 16

# Layer definitions: (name, kernel_shape, bias_shape, constraint_type)
# constraint_type: 'A', 'B', 'C', or None
# ORDER MATTERS - must match export order from trainer

LAYERS = [
    ('hidden_1a', (ACTIVE_INPUTS * MAIN_BUCKETS, ACCUMULATOR_SIZE), (ACCUMULATOR_SIZE,), 'A'),
    ('hidden_1b', (256, POOLED), (POOLED,), 'B'),
    ('hidden_1c', (256, POOLED), (POOLED,), 'C'),  # bishops + occupancy, adds to hidden_1b
    ('pool', (2 * ACCUMULATOR_SIZE, 1), (0,), None),  # stm-major, kernel-only, float; (0,) -> np.prod == 0, no bias read
    ('hidden_2', (POOLED, 16), (16,), None),
    ('hidden_3', (16, 16), (16,), None),
    ('out', (16, 1), (1,), None),
]


def get_constraint_params(constraint_type):
    if constraint_type == 'A':
        return Q_MAX_A, Q_SCALE
    elif constraint_type == 'B':
        return Q_MAX_B, Q_SCALE
    elif constraint_type == 'C':
        return Q_MAX_C, Q_SCALE
    else:
        return None, None


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
    """Check if all values are multiples of 1/qscale (except clamped edge values)."""
    rounded = np.round(weights * qscale) / qscale
    rounded = rounded.astype(np.float32)
    
    violations = weights != rounded
    # Exclude values at the clamp boundaries - these are expected to not be rounded
    at_boundary = np.abs(weights) >= qmax * 0.9999
    violations = violations & ~at_boundary
    
    count = np.sum(violations)
    if count > 0:
        idx = np.where(violations.flat)[0][:5]
        print(f"  ROUND VIOLATION in {layer_name} {weight_type}: {count} values not rounded to 1/{qscale}")
        for i in idx:
            w = weights.flat[i]
            r = rounded.flat[i]
            print(f"    [{i}] {w:.10f} should be {r:.10f}, diff={w-r:.2e}")
        return count
    
    # Report boundary values for info
    boundary_count = np.sum(at_boundary)
    if boundary_count > 0:
        print(f"  (note: {boundary_count} values at boundary ±{qmax:.10f}, rounding not checked)")
    
    return 0


def verify_layers(data, layers, offset=0):
    """Verify a list of layers starting at given offset. Returns new offset and violation counts."""
    total_clip_violations = 0
    total_round_violations = 0
    
    for layer_name, kernel_shape, bias_shape, constraint_type in layers:
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
        
        print(f"{layer_name}: kernel {kernel_shape}, bias {bias_shape}, constraint: {constraint_type}")
        
        if constraint_type is None:
            print(f"  (no constraint)")
            continue
        
        qmax, qscale = get_constraint_params(constraint_type)
        
        # Check clipping
        total_clip_violations += check_clipping(kernel, qmax, layer_name, "kernel")
        total_clip_violations += check_clipping(bias, qmax, layer_name, "bias")
        
        # Check rounding (pass qmax to exclude boundary values)
        total_round_violations += check_rounding(kernel, qscale, qmax, layer_name, "kernel")
        total_round_violations += check_rounding(bias, qscale, qmax, layer_name, "bias")
    
    return offset, total_clip_violations, total_round_violations, True


def main():
    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]} [weights.bin]")
        sys.exit(1)

    filepath = sys.argv[1] if len(sys.argv) == 2 else str(fetch_weights.ensure())
    print(f"Loading: {filepath}")
    print(f"Q_SCALE = {Q_SCALE}")
    print(f"Q_MAX_A = {Q_MAX_A:.10f} (hidden_1a)")
    print(f"Q_MAX_B = {Q_MAX_B:.10f} (hidden_1b)")
    print(f"Q_MAX_C = {Q_MAX_C:.10f} (hidden_1c)")
    print()
    
    data = np.fromfile(filepath, dtype=np.float32)
    print(f"Total values: {len(data)}")

    # Without hidden_1c (NNUE_HIDDEN_1C off)?
    layers = LAYERS
    no_1c = [layer for layer in LAYERS if layer[0] != 'hidden_1c']
    no_1c_size = sum(np.prod(k) + np.prod(b) for _, k, b, _ in no_1c)
    if len(data) == no_1c_size:
        layers = no_1c
        print("Detected: model WITHOUT hidden_1c")

    base_total = sum(np.prod(k) + np.prod(b) for _, k, b, _ in layers)
    print(f"Expected: {base_total}")

    if len(data) != base_total:
        print(f"ERROR: Size mismatch! Got {len(data)}, expected {base_total}")
        sys.exit(1)

    print()

    offset, total_clip_violations, total_round_violations, success = verify_layers(data, layers)

    if not success:
        print("ERROR: Unexpected end of data while reading base layers")
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


if __name__ == '__main__':
    main()
