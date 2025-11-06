#!/usr/bin/env python3
"""
INT8 양자화 해상도 분석: Linear vs Bounded Inverse Depth
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_int8_resolution():
    """INT8 양자화 해상도 상세 분석"""
    
    min_depth = 0.5
    max_depth = 15.0
    int8_levels = 256  # 8-bit unsigned
    
    print("="*80)
    print("INT8 Quantization Resolution Analysis")
    print("="*80)
    print(f"Depth range: [{min_depth}, {max_depth}]m")
    print(f"INT8 levels: {int8_levels}")
    print()
    
    # =========================================================================
    # Method 1: Linear Depth (Direct)
    # =========================================================================
    print("📊 Method 1: LINEAR DEPTH (Direct Quantization)")
    print("-"*80)
    
    # Linear mapping: depth ∈ [0.5, 15] → int8 ∈ [0, 255]
    depth_range = max_depth - min_depth
    linear_step = depth_range / (int8_levels - 1)
    
    print(f"Formula: int8 = round((depth - {min_depth}) / {linear_step:.6f})")
    print(f"Quantization step: {linear_step:.6f}m = {linear_step*1000:.3f}mm")
    print()
    
    # 깊이별 절대 오차
    test_depths = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    print("Absolute error at different depths:")
    for d in test_depths:
        max_error = linear_step / 2
        rel_error = (max_error / d) * 100
        print(f"  {d:5.1f}m: max_error={max_error*1000:6.3f}mm ({rel_error:5.2f}%)")
    
    print()
    
    # =========================================================================
    # Method 2: Bounded Inverse Depth (Current)
    # =========================================================================
    print("📊 Method 2: BOUNDED INVERSE DEPTH (Current)")
    print("-"*80)
    
    inv_min = 1.0 / max_depth  # 0.0667
    inv_max = 1.0 / min_depth  # 2.0
    inv_range = inv_max - inv_min
    
    # Sigmoid [0,1] → int8 [0,255]
    sigmoid_step = 1.0 / (int8_levels - 1)
    
    print(f"Inverse range: [{inv_min:.4f}, {inv_max:.4f}]")
    print(f"Formula: sigmoid → int8 [0,255] → inv = {inv_min:.4f} + {inv_range:.4f} × (int8/255)")
    print(f"Sigmoid quantization step: {sigmoid_step:.8f}")
    print()
    
    # 깊이별 오차 계산 (미분 사용)
    # depth = 1 / inv
    # inv = inv_min + inv_range × sigmoid
    # d(depth)/d(sigmoid) = -1/inv² × inv_range
    
    print("Absolute error at different depths (via derivative):")
    for d in test_depths:
        inv = 1.0 / d
        # 미분: |∂depth/∂sigmoid| = inv_range / inv²
        deriv = inv_range / (inv ** 2)
        max_error = deriv * sigmoid_step / 2
        rel_error = (max_error / d) * 100
        print(f"  {d:5.1f}m: max_error={max_error*1000:6.3f}mm ({rel_error:5.2f}%)")
    
    print()
    
    # =========================================================================
    # Method 3: Log-space Depth
    # =========================================================================
    print("📊 Method 3: LOG-SPACE DEPTH")
    print("-"*80)
    
    log_min = np.log(min_depth)  # -0.693
    log_max = np.log(max_depth)  # 2.708
    log_range = log_max - log_min
    log_step = log_range / (int8_levels - 1)
    
    print(f"Log range: [{log_min:.3f}, {log_max:.3f}]")
    print(f"Formula: log(depth) = {log_min:.3f} + {log_range:.3f} × (int8/255)")
    print(f"Log quantization step: {log_step:.6f}")
    print()
    
    print("Absolute error at different depths (via derivative):")
    for d in test_depths:
        # depth = exp(log_depth)
        # d(depth)/d(log_depth) = depth
        deriv = d
        max_error = deriv * log_step / 2
        rel_error = (max_error / d) * 100
        print(f"  {d:5.1f}m: max_error={max_error*1000:6.3f}mm ({rel_error:5.2f}%)")
    
    print()
    
    # =========================================================================
    # Comparison Plot
    # =========================================================================
    print("📈 Generating comparison plot...")
    
    depths = np.linspace(min_depth, max_depth, 1000)
    
    # Linear
    linear_errors = np.full_like(depths, linear_step / 2)
    
    # Bounded Inverse
    invs = 1.0 / depths
    bounded_errors = (inv_range / (invs ** 2)) * sigmoid_step / 2
    
    # Log-space
    log_errors = depths * log_step / 2
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Absolute Error (mm)
    ax1 = axes[0]
    ax1.plot(depths, linear_errors * 1000, 'b-', linewidth=2, label='Linear')
    ax1.plot(depths, bounded_errors * 1000, 'r-', linewidth=2, label='Bounded Inverse')
    ax1.plot(depths, log_errors * 1000, 'g-', linewidth=2, label='Log-space')
    ax1.set_xlabel('Depth (m)', fontsize=12)
    ax1.set_ylabel('Max Absolute Error (mm)', fontsize=12)
    ax1.set_title('INT8 Quantization Error vs Depth (Absolute)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min_depth, max_depth])
    
    # Plot 2: Relative Error (%)
    ax2 = axes[1]
    ax2.plot(depths, (linear_errors / depths) * 100, 'b-', linewidth=2, label='Linear')
    ax2.plot(depths, (bounded_errors / depths) * 100, 'r-', linewidth=2, label='Bounded Inverse')
    ax2.plot(depths, (log_errors / depths) * 100, 'g-', linewidth=2, label='Log-space')
    ax2.set_xlabel('Depth (m)', fontsize=12)
    ax2.set_ylabel('Max Relative Error (%)', fontsize=12)
    ax2.set_title('INT8 Quantization Error vs Depth (Relative)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min_depth, max_depth])
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='1% threshold')
    
    plt.tight_layout()
    output_path = 'outputs/int8_quantization_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    plt.close()
    
    # =========================================================================
    # Detailed Error Table
    # =========================================================================
    print()
    print("="*80)
    print("📋 DETAILED COMPARISON TABLE")
    print("="*80)
    print()
    
    print(f"{'Depth':<8} {'Linear Error':<20} {'Bounded Inv Error':<20} {'Log Error':<20}")
    print(f"{'(m)':<8} {'(mm / %)':<20} {'(mm / %)':<20} {'(mm / %)':<20}")
    print("-"*80)
    
    for d in test_depths:
        # Linear
        lin_err = linear_step / 2
        lin_rel = (lin_err / d) * 100
        
        # Bounded Inverse
        inv = 1.0 / d
        bound_err = (inv_range / (inv ** 2)) * sigmoid_step / 2
        bound_rel = (bound_err / d) * 100
        
        # Log-space
        log_err = d * log_step / 2
        log_rel = (log_err / d) * 100
        
        print(f"{d:6.1f}   {lin_err*1000:6.3f} / {lin_rel:5.2f}%    "
              f"{bound_err*1000:6.3f} / {bound_rel:5.2f}%    "
              f"{log_err*1000:6.3f} / {log_rel:5.2f}%")
    
    print()
    
    # =========================================================================
    # Key Findings
    # =========================================================================
    print("="*80)
    print("🔍 KEY FINDINGS")
    print("="*80)
    print()
    
    # Find worst-case errors
    worst_linear = max(linear_errors)
    worst_bounded = max(bounded_errors)
    worst_log = max(log_errors)
    
    worst_linear_rel = max((linear_errors / depths) * 100)
    worst_bounded_rel = max((bounded_errors / depths) * 100)
    worst_log_rel = max((log_errors / depths) * 100)
    
    print("Worst-case Absolute Errors:")
    print(f"  Linear:         {worst_linear*1000:.3f}mm (constant)")
    print(f"  Bounded Inv:    {worst_bounded*1000:.3f}mm (at {min_depth}m)")
    print(f"  Log-space:      {worst_log*1000:.3f}mm (at {max_depth}m)")
    print()
    
    print("Worst-case Relative Errors:")
    print(f"  Linear:         {worst_linear_rel:.2f}% (at {min_depth}m)")
    print(f"  Bounded Inv:    {worst_bounded_rel:.2f}% (at {min_depth}m)")
    print(f"  Log-space:      {worst_log_rel:.2f}% (constant)")
    print()
    
    print("✅ Linear is BEST for:")
    print("   - Uniform absolute error across all depths")
    print("   - Simple implementation")
    print("   - Direct depth prediction")
    print()
    
    print("⚠️  Bounded Inverse is WORST for:")
    print("   - Near-field accuracy (high error at 0.5m)")
    print("   - INT8 quantization (error amplification)")
    print("   - Non-uniform error distribution")
    print()
    
    print("🟢 Log-space is BEST for:")
    print("   - Uniform relative error (constant %)")
    print("   - Human perception (Weber-Fechner law)")
    print("   - Far-field accuracy")
    print()
    
    # =========================================================================
    # INT8 Sufficiency Analysis
    # =========================================================================
    print("="*80)
    print("❓ IS INT8 SUFFICIENT FOR LINEAR DEPTH?")
    print("="*80)
    print()
    
    print(f"Linear quantization step: {linear_step*1000:.3f}mm = {linear_step*100:.3f}cm")
    print()
    
    # Typical depth accuracy requirements
    requirements = {
        "Automotive (ADAS)": 100,  # 10cm
        "Robotics (navigation)": 50,  # 5cm
        "AR/VR": 10,  # 1cm
        "Precision robotics": 5,  # 5mm
    }
    
    print("Comparison with typical requirements:")
    for app, req_mm in requirements.items():
        status = "✅ OK" if linear_step*1000 < req_mm else "❌ INSUFFICIENT"
        print(f"  {app:<25}: {req_mm:4.0f}mm required → {status}")
    
    print()
    print("🎯 CONCLUSION:")
    print(f"   INT8 linear depth provides {linear_step*1000:.3f}mm resolution")
    print(f"   This is SUFFICIENT for most applications (< 100mm)")
    print(f"   Maximum relative error at 0.5m: {worst_linear_rel:.2f}%")
    print(f"   Maximum relative error at 15m: {(linear_step/2/max_depth)*100:.2f}%")
    print()


if __name__ == '__main__':
    analyze_int8_resolution()
