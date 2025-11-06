#!/usr/bin/env python3
"""
왜 Bounded Inverse와 Direct Linear가 똑같은 결과를 내는가?

문제: sigmoid_to_depth_linear()이 사실상 Bounded Inverse와 동일한 공식을 사용!
"""

import torch
import numpy as np

def bounded_inverse_transform(sigmoid, min_depth=0.5, max_depth=15.0):
    """기존 방법: Bounded Inverse"""
    inv_min = 1.0 / max_depth
    inv_max = 1.0 / min_depth
    inv_depth = inv_min + (inv_max - inv_min) * sigmoid
    depth = 1.0 / inv_depth
    return depth


def linear_transform_wrong(sigmoid, min_depth=0.5, max_depth=15.0):
    """
    packnet_sfm/utils/post_process_depth.py의 sigmoid_to_depth_linear()
    
    이름은 'linear'지만 실제로는 Bounded Inverse!
    """
    min_inv = 1.0 / max_depth
    max_inv = 1.0 / min_depth
    inv_depth = min_inv + (max_inv - min_inv) * sigmoid
    depth = 1.0 / inv_depth
    return depth


def linear_transform_correct(sigmoid, min_depth=0.5, max_depth=15.0):
    """
    진짜 Linear 변환 (Direct Depth)
    
    depth = min_depth + (max_depth - min_depth) * sigmoid
    """
    depth = min_depth + (max_depth - min_depth) * sigmoid
    return depth


print("="*80)
print("Sigmoid → Depth 변환 방법 비교")
print("="*80)
print()

# Test values
sigmoid_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
min_depth = 0.5
max_depth = 15.0

print(f"Sigmoid values: {sigmoid_values.tolist()}")
print(f"Depth range: [{min_depth}, {max_depth}]m")
print()

# Method 1: Bounded Inverse
depth_bounded = bounded_inverse_transform(sigmoid_values, min_depth, max_depth)
print("1️⃣  Bounded Inverse (기존):")
print(f"   inv_depth = 1/{max_depth} + (1/{min_depth} - 1/{max_depth}) × sigmoid")
print(f"   depth = 1 / inv_depth")
print(f"   Result: {depth_bounded.tolist()}")
print()

# Method 2: sigmoid_to_depth_linear (현재 코드)
depth_linear_wrong = linear_transform_wrong(sigmoid_values, min_depth, max_depth)
print("2️⃣  sigmoid_to_depth_linear() [현재 코드]:")
print(f"   ❌ 이름은 'linear'지만 실제로는 Bounded Inverse!")
print(f"   Result: {depth_linear_wrong.tolist()}")
print()

# Method 3: True Linear (Direct Depth)
depth_linear_correct = linear_transform_correct(sigmoid_values, min_depth, max_depth)
print("3️⃣  True Linear (Direct Depth) [원하는 방법]:")
print(f"   depth = {min_depth} + ({max_depth} - {min_depth}) × sigmoid")
print(f"   Result: {depth_linear_correct.tolist()}")
print()

# Comparison
print("="*80)
print("🔍 비교 분석")
print("="*80)
print()

print("Bounded Inverse == sigmoid_to_depth_linear():")
print(f"   {torch.allclose(depth_bounded, depth_linear_wrong)}")
print()

print("Bounded Inverse == True Linear:")
print(f"   {torch.allclose(depth_bounded, depth_linear_correct)}")
print()

print("sigmoid_to_depth_linear() == True Linear:")
print(f"   {torch.allclose(depth_linear_wrong, depth_linear_correct)}")
print()

# Detailed comparison
print("="*80)
print("📊 Detailed Comparison @ Sigmoid=0.5")
print("="*80)
print()

idx = 2  # sigmoid=0.5
print(f"Sigmoid: {sigmoid_values[idx]:.2f}")
print(f"   Bounded Inverse:       {depth_bounded[idx]:.4f}m")
print(f"   sigmoid_to_depth_linear(): {depth_linear_wrong[idx]:.4f}m")
print(f"   True Linear (Direct):  {depth_linear_correct[idx]:.4f}m")
print()

# INT8 Error Analysis
print("="*80)
print("🎯 INT8 Quantization Error @ 15m")
print("="*80)
print()

# Sigmoid step for INT8
sigmoid_step = 1.0 / 255
print(f"INT8 sigmoid step: {sigmoid_step:.6f}")
print()

# Bounded Inverse error
sigmoid_at_15m = 0.0  # sigmoid=0 → depth=15m
d_inv_d_sigmoid = -(1.9333) / ((1/15 + 1.9333 * sigmoid_at_15m) ** 2)
error_bounded = abs(d_inv_d_sigmoid * sigmoid_step) * 1000  # mm
print(f"Bounded Inverse @ 15m:")
print(f"   |∂depth/∂sigmoid| = {abs(d_inv_d_sigmoid):.2f}")
print(f"   Error = {error_bounded:.1f}mm ❌")
print()

# True Linear error
depth_range = max_depth - min_depth
error_linear = (depth_range / 255 / 2) * 1000  # mm
print(f"True Linear (Direct) @ 15m:")
print(f"   |∂depth/∂sigmoid| = {depth_range:.2f} (constant)")
print(f"   Error = ±{error_linear:.1f}mm ✅")
print()

print("="*80)
print("🚨 CRITICAL FINDING")
print("="*80)
print()
print("packnet_sfm/utils/post_process_depth.py의 sigmoid_to_depth_linear()은")
print("이름과 달리 Bounded Inverse 변환을 사용하고 있습니다!")
print()
print("❌ 현재 코드:")
print("   inv_depth = min_inv + (max_inv - min_inv) * sigmoid")
print("   depth = 1 / inv_depth")
print()
print("✅ 올바른 Linear:")
print("   depth = min_depth + (max_depth - min_depth) * sigmoid")
print()
print("→ sigmoid_to_depth_linear()을 수정하거나")
print("→ 새로운 함수 sigmoid_to_depth_direct()를 추가해야 합니다!")
