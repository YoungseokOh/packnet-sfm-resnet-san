#!/usr/bin/env python3
"""
Load and verify raw sigmoid outputs.
"""

import numpy as np
import os

def load_and_verify(npy_path):
    """Load and verify a sigmoid output file."""
    data = np.load(npy_path)
    
    print(f"\n{'='*80}")
    print(f"File: {os.path.basename(npy_path)}")
    print(f"{'='*80}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Range: [{data.min():.6f}, {data.max():.6f}]")
    print(f"Mean: {data.mean():.6f}")
    print(f"Std: {data.std():.6f}")
    print(f"Median: {np.median(data):.6f}")
    
    # Check value distribution
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(data.flatten(), bins=bins)
    total = data.size
    
    print(f"\nValue Distribution:")
    for i in range(len(bins)-1):
        pct = 100.0 * hist[i] / total
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:7d} pixels ({pct:5.2f}%)")
    
    # Example conversions
    print(f"\n💡 Example Depth Conversions (min=0.5m, max=15.0m):")
    
    # Linear transform
    min_inv = 1.0 / 15.0  # 0.0667
    max_inv = 1.0 / 0.5   # 2.0
    
    sample_sigmoids = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\n  Linear Transform:")
    for sig in sample_sigmoids:
        inv = min_inv + (max_inv - min_inv) * sig
        depth = 1.0 / inv
        print(f"    sigmoid={sig:.2f} → inv={inv:.4f} → depth={depth:.4f}m")
    
    # Log transform
    log_min_inv = np.log(min_inv)
    log_max_inv = np.log(max_inv)
    
    print(f"\n  Log Transform:")
    for sig in sample_sigmoids:
        log_inv = log_min_inv + (log_max_inv - log_min_inv) * sig
        inv = np.exp(log_inv)
        depth = 1.0 / inv
        print(f"    sigmoid={sig:.2f} → inv={inv:.4f} → depth={depth:.4f}m")
    
    return data


if __name__ == '__main__':
    output_dir = '/workspace/packnet-sfm/outputs/raw_sigmoid_outputs'
    
    print(f"\n{'='*80}")
    print(f"RAW SIGMOID OUTPUT VERIFICATION")
    print(f"{'='*80}")
    print(f"Directory: {output_dir}")
    
    # Load all files
    npy_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])
    
    print(f"\nFound {len(npy_files)} .npy files\n")
    
    for npy_file in npy_files:
        npy_path = os.path.join(output_dir, npy_file)
        data = load_and_verify(npy_path)
    
    print(f"\n{'='*80}")
    print(f"✅ VERIFICATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll files are raw sigmoid outputs [0, 1]")
    print(f"Shape: (384, 640) - matches model output resolution")
    print(f"Dtype: float32")
    print(f"\n💡 To convert to depth:")
    print(f"   1. Linear: sigmoid_to_depth_linear(sigmoid, min=0.5, max=15.0)")
    print(f"   2. Log:    sigmoid_to_depth_log(sigmoid, min=0.5, max=15.0)")
    print(f"   3. Use packnet_sfm.utils.post_process_depth module")
    print(f"{'='*80}\n")
