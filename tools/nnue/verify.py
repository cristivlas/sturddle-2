#!/usr/bin/env python3
"""
Verify NNUE binary weights for proper clipping and rounding.
Architecture: 1280-accumulator with attention modulation.
"""
import sys
import numpy as np

Q_SCALE = 1024

# Constraint A: for hidden_1a and move layers
Q_MAX_A = 32767 / Q_SCALE / 66

# Constraint B: for hidden_1b layer
Q_MAX_B = 32767 / Q_SCALE / 19

ACCUMULATOR_SIZE = 1280
POOL_SIZE = 8
ATTN_BUCKETS = 4

# Layer definitions: (name, kernel_shape, bias_shape, constraint_type)
# constraint_type: 'A', 'B', or None
# ORDER MATTERS - must match export order from trainer

LAYERS = [
    ('hidden_1b', (256, 64), (64,), 'B'),
    ('hidden_1a', (3588, ACCUMULATOR_SIZE), (ACCUMULATOR_SIZE,), 'A'),
    ('spatial_attn', (64 * ATTN_BUCKETS, 32), (32,), None),  # bucketed: 64 inputs × 4 buckets
    ('hidden_2', (ACCUMULATOR_SIZE // POOL_SIZE, 16), (16,), None),
    ('hidden_3', (16, 16), (16,), None),
    ('out', (16, 1), (1,), None),
]

# Optional move prediction layer
MOVE_LAYER = ('move', (897, 4096), (4096,), 'A')


def get_constraint_params(constraint_type):
    if constraint_type == 'A':
        return Q_MAX_A, Q_SCALE
    elif constraint_type == 'B':
        return Q_MAX_B, Q_SCALE
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
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <weights.bin>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Loading: {filepath}")
    print(f"Q_SCALE = {Q_SCALE}")
    print(f"Q_MAX_A = {Q_MAX_A:.10f} (hidden_1a, move)")
    print(f"Q_MAX_B = {Q_MAX_B:.10f} (hidden_1b)")
    print()
    
    data = np.fromfile(filepath, dtype=np.float32)
    print(f"Total values: {len(data)}")
    
    # Calculate expected sizes
    base_total = sum(np.prod(k) + np.prod(b) for _, k, b, _ in LAYERS)
    move_total = np.prod(MOVE_LAYER[1]) + np.prod(MOVE_LAYER[2])
    
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
    
    # Verify move layer if present
    if has_move_layer:
        offset, clip_v, round_v, success = verify_layers(data, [MOVE_LAYER], offset)
        total_clip_violations += clip_v
        total_round_violations += round_v
        
        if not success:
            print("ERROR: Unexpected end of data while reading move layer")
            sys.exit(1)
    
    # Verify we consumed all data
    if offset != len(data):
        print(f"WARNING: {len(data) - offset} bytes remaining after parsing")
    
    print()
    print("=" * 60)
    if total_clip_violations == 0 and total_round_violations == 0:
        print("All constraints satisfied!")
    else:
        print(f"Total clipping violations: {total_clip_violations}")
        print(f"Total rounding violations: {total_round_violations}")


if __name__ == '__main__':
    main()
