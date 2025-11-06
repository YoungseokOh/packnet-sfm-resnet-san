#!/usr/bin/env python3
"""
Debug Direct Depth Model Output

Check what depth values the model is actually producing.
"""

import sys
sys.path.append('/workspace/packnet-sfm')

import torch
import numpy as np
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01

def debug_model_output():
    """Debug what the model outputs"""
    
    print("=" * 80)
    print("🔍 Debugging Direct Depth Model Output")
    print("=" * 80)
    
    min_depth, max_depth = 0.5, 15.0
    
    # Create model
    model = ResNetSAN01(
        depth_output_mode='direct',
        min_depth=min_depth,
        max_depth=max_depth
    )
    model.eval()
    
    # Create test input
    rgb = torch.randn(1, 3, 384, 640)
    
    # Forward pass
    with torch.no_grad():
        output = model(rgb)
    
    # Get depth output
    depth = output['inv_depths'][0]  # Scale 0
    
    print(f"\n✅ Model Output Statistics:")
    print(f"   Shape: {depth.shape}")
    print(f"   Min: {depth.min().item():.4f}m")
    print(f"   Max: {depth.max().item():.4f}m")
    print(f"   Mean: {depth.mean().item():.4f}m")
    print(f"   Std: {depth.std().item():.4f}m")
    print(f"   Median: {depth.median().item():.4f}m")
    
    print(f"\n📐 Expected Range:")
    print(f"   Min: {min_depth}m")
    print(f"   Max: {max_depth}m")
    print(f"   Mean (expected): ~7.75m (midpoint)")
    
    # Check if values are in valid range
    in_range = (depth >= min_depth) & (depth <= max_depth)
    print(f"\n🔍 Range Check:")
    print(f"   Values in range [{min_depth}, {max_depth}]: {in_range.float().mean()*100:.2f}%")
    
    if in_range.float().mean() < 0.99:
        print(f"   ⚠️  Some values are OUT OF RANGE!")
        print(f"   Values < {min_depth}: {(depth < min_depth).float().mean()*100:.2f}%")
        print(f"   Values > {max_depth}: {(depth > max_depth).float().mean()*100:.2f}%")
    
    # Check sigmoid values (internal)
    print(f"\n🔍 Checking internal sigmoid values...")
    
    # Access decoder to get sigmoid before transformation
    with torch.no_grad():
        rgb_features = model.encoder(rgb)
        skip_features = [feat.detach() for feat in rgb_features]
        decoder_output = model.decoder(skip_features)
        sigmoid = decoder_output[("disp", 0)]
    
    print(f"\n📊 Sigmoid Output (before linear transform):")
    print(f"   Shape: {sigmoid.shape}")
    print(f"   Min: {sigmoid.min().item():.6f}")
    print(f"   Max: {sigmoid.max().item():.6f}")
    print(f"   Mean: {sigmoid.mean().item():.6f}")
    print(f"   Expected range: [0, 1]")
    
    if sigmoid.min() < 0 or sigmoid.max() > 1:
        print(f"\n❌ PROBLEM FOUND: Sigmoid values are NOT in [0,1] range!")
        print(f"   This suggests the decoder is not using sigmoid activation properly")
    
    # Manually compute what depth should be
    expected_depth = min_depth + (max_depth - min_depth) * sigmoid
    
    print(f"\n🔍 Manual Depth Calculation:")
    print(f"   Expected depth min: {expected_depth.min().item():.4f}m")
    print(f"   Expected depth max: {expected_depth.max().item():.4f}m")
    print(f"   Expected depth mean: {expected_depth.mean().item():.4f}m")
    
    # Compare with actual output
    diff = (depth - expected_depth).abs()
    print(f"\n🔍 Comparison with model output:")
    print(f"   Difference (mean): {diff.mean().item():.6f}m")
    print(f"   Difference (max): {diff.max().item():.6f}m")
    
    if diff.max() > 0.001:
        print(f"\n❌ MISMATCH: Model output != Expected linear transformation!")
        print(f"   Model might be using wrong transformation")
    else:
        print(f"\n✅ Model output matches expected linear transformation")
    
    # Check percentiles
    depth_np = depth.cpu().numpy().flatten()
    print(f"\n📊 Depth Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(depth_np, p)
        print(f"   p{p:2d}: {val:.4f}m")
    
    return depth, sigmoid


if __name__ == '__main__':
    depth, sigmoid = debug_model_output()
    
    print("\n" + "=" * 80)
    print("🔍 Diagnosis Complete")
    print("=" * 80)
    print("\nNext: Check if the model's checkpoint is loading correctly")
    print("or if it's starting from random initialization.")
