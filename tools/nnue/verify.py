#!/usr/bin/env python3
"""
Verify NNUE-256 binary weights for proper clipping and rounding.
"""
import sys
import numpy as np

Q16_SCALE = 255
Q16_MAX = 32767 / Q16_SCALE / 33

Q8_SCALE = 64
Q8_MAX = 127 / Q8_SCALE

# Layer definitions: (name, kernel_shape, bias_shape, constraint_type)
# constraint_type: 'Q16', 'Q8', or None
LAYERS = [
    ('black_perspective', (32 * 768, 256), (256,), 'Q16'),
    ('white_perspective', (32 * 768, 256), (256,), 'Q16'),
    ('hidden_2', (512, 32), (32,), 'Q8'),
    ('hidden_3', (32, 8), (8,), None),
    ('eval', (8, 1), (1,), None),
]

def get_constraint_params(constraint_type):
    if constraint_type == 'Q16':
        return Q16_MAX, Q16_SCALE
    elif constraint_type == 'Q8':
        return Q8_MAX, Q8_SCALE
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

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <weights.bin>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Loading: {filepath}")
    print(f"Q16_MAX = {Q16_MAX:.10f}, Q16_SCALE = {Q16_SCALE}")
    print(f"Q8_MAX = {Q8_MAX:.10f}, Q8_SCALE = {Q8_SCALE}")
    print()
    
    data = np.fromfile(filepath, dtype=np.float32)
    print(f"Total values: {len(data)}")
    
    expected_total = sum(np.prod(k) + np.prod(b) for _, k, b, _ in LAYERS)
    print(f"Expected total: {expected_total}")
    
    if len(data) != expected_total:
        print(f"ERROR: Size mismatch!")
        sys.exit(1)
    
    print()
    
    offset = 0
    total_clip_violations = 0
    total_round_violations = 0
    
    for layer_name, kernel_shape, bias_shape, constraint_type in LAYERS:
        kernel_size = np.prod(kernel_shape)
        bias_size = np.prod(bias_shape)
        
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
    
    print()
    print("=" * 60)
    if total_clip_violations == 0 and total_round_violations == 0:
        print("All constraints satisfied!")
    else:
        print(f"Total clipping violations: {total_clip_violations}")
        print(f"Total rounding violations: {total_round_violations}")

if __name__ == '__main__':
    main()
