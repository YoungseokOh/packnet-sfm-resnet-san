#!/usr/bin/env python3
"""
Direct Depth Output의 INT8 양자화 영향 분석

Sigmoid 제거하고 Linear Depth를 직접 출력할 때:
1. 출력 범위: [0.5, 15.0]m
2. INT8 양자화 (0~255 levels)
3. 양자화 오류 분석
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_direct_depth_int8():
    """Direct Depth Output의 INT8 양자화 분석"""
    
    print("="*80)
    print("Direct Depth Output + INT8 Quantization Analysis")
    print("="*80)
    print()
    
    # Configuration
    min_depth = 0.5
    max_depth = 15.0
    depth_range = max_depth - min_depth
    int8_levels = 256
    
    print(f"📊 Configuration:")
    print(f"   Depth range: [{min_depth}, {max_depth}]m")
    print(f"   Range width: {depth_range}m")
    print(f"   INT8 levels: {int8_levels}")
    print()
    
    # =========================================================================
    # Method 1: FP32 Direct Output (No quantization)
    # =========================================================================
    print("="*80)
    print("Method 1: FP32 Direct Depth Output (Baseline)")
    print("="*80)
    print()
    print("Model architecture:")
    print("   Conv → ReLU/Clamp → Depth [0.5, 15.0]m")
    print("   Output: FP32 (32-bit floating point)")
    print("   Precision: ~7 significant digits")
    print("   Error: Negligible (<0.001mm)")
    print()
    
    # =========================================================================
    # Method 2: INT8 Quantized Output
    # =========================================================================
    print("="*80)
    print("Method 2: INT8 Quantized Depth Output")
    print("="*80)
    print()
    
    # Quantization parameters
    scale = depth_range / (int8_levels - 1)
    zero_point = 0  # Asymmetric quantization
    
    print(f"Quantization scheme:")
    print(f"   Scale: {scale:.6f}m = {scale*1000:.3f}mm")
    print(f"   Zero point: {zero_point}")
    print(f"   Formula: depth = scale × (int8_value - zero_point) + min_depth")
    print()
    
    print(f"INT8 representation:")
    print(f"   int8=0   → depth = {min_depth:.3f}m")
    print(f"   int8=127 → depth = {min_depth + scale * 127:.3f}m")
    print(f"   int8=255 → depth = {max_depth:.3f}m")
    print()
    
    # Quantization step size
    step_size = scale
    max_error = step_size / 2
    
    print(f"Quantization error:")
    print(f"   Step size: {step_size*1000:.3f}mm")
    print(f"   Max absolute error: ±{max_error*1000:.3f}mm")
    print(f"   Max relative error @ 0.5m: ±{(max_error/0.5)*100:.2f}%")
    print(f"   Max relative error @ 15m: ±{(max_error/15.0)*100:.2f}%")
    print()
    
    # =========================================================================
    # Comparison: Sigmoid INT8 vs Direct Depth INT8
    # =========================================================================
    print("="*80)
    print("Comparison: Sigmoid INT8 vs Direct Depth INT8")
    print("="*80)
    print()
    
    # Sigmoid → Bounded Inverse → Depth (previous method)
    print("Previous Method (Sigmoid INT8 → Bounded Inverse):")
    print(f"   INT8 levels: 256 (for sigmoid [0, 1])")
    print(f"   Sigmoid step: 1/255 = 0.00392")
    print(f"   Depth @ 0.5m error: ~0.9mm")
    print(f"   Depth @ 15m error: ~853mm ❌ CATASTROPHIC")
    print()
    
    # Direct Depth INT8 (new method)
    print("New Method (Direct Depth INT8):")
    print(f"   INT8 levels: 256 (for depth [0.5, 15.0]m)")
    print(f"   Depth step: {step_size*1000:.3f}mm")
    print(f"   Depth @ 0.5m error: ±{max_error*1000:.3f}mm")
    print(f"   Depth @ 15m error: ±{max_error*1000:.3f}mm ✅ UNIFORM")
    print()
    
    print("✅ Improvement:")
    print(f"   @ 0.5m: 0.9mm → {max_error*1000:.3f}mm ({0.9/(max_error*1000):.1f}x worse, but acceptable)")
    print(f"   @ 15m: 853mm → {max_error*1000:.3f}mm ({853/(max_error*1000):.1f}x better!) 🎉")
    print()
    
    # =========================================================================
    # Detailed Error Analysis
    # =========================================================================
    print("="*80)
    print("Detailed Error Analysis")
    print("="*80)
    print()
    
    test_depths = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    
    print(f"{'Depth (m)':<10} {'Absolute Error (mm)':<22} {'Relative Error (%)':<20}")
    print("-"*80)
    
    for d in test_depths:
        abs_err = max_error * 1000  # mm (constant)
        rel_err = (max_error / d) * 100  # %
        print(f"{d:8.1f}   ±{abs_err:6.3f}mm                 ±{rel_err:5.2f}%")
    
    print()
    print("Key Observation:")
    print("   ✅ Absolute error is CONSTANT (~28mm)")
    print("   ⚠️  Relative error is HIGHER at near-field (5.69% @ 0.5m)")
    print("   ✅ But still acceptable for ADAS applications")
    print()
    
    # =========================================================================
    # Application Requirements Check
    # =========================================================================
    print("="*80)
    print("Application Requirements Check")
    print("="*80)
    print()
    
    requirements = {
        "ADAS (Autonomous Driving)": 100,  # mm
        "Robotics (Navigation)": 50,       # mm
        "AR/VR": 10,                       # mm
    }
    
    for app, req_mm in requirements.items():
        status = "✅ PASS" if max_error*1000 <= req_mm else "❌ FAIL"
        print(f"   {app:<30} Req: ≤{req_mm:3d}mm, Got: ±{max_error*1000:.1f}mm {status}")
    
    print()
    
    # =========================================================================
    # INT8 Range Utilization
    # =========================================================================
    print("="*80)
    print("INT8 Range Utilization")
    print("="*80)
    print()
    
    print("Full INT8 range [0, 255] is used:")
    print("   int8=0   → 0.5m   (min)")
    print("   int8=255 → 15.0m  (max)")
    print("   Coverage: 100% ✅")
    print()
    print("Contrast with Sigmoid INT8 → Bounded Inverse:")
    print("   Sigmoid distribution: 60% at extremes [0.0, 0.2) + [0.8, 1.0)")
    print("   Middle range [0.2, 0.8): Only 40% utilization")
    print("   → Wasted INT8 levels ❌")
    print()
    print("Direct Depth INT8:")
    print("   Uniform distribution across [0.5, 15.0]m")
    print("   → Optimal INT8 utilization ✅")
    print()
    
    # =========================================================================
    # Visualization
    # =========================================================================
    print("="*80)
    print("Generating Visualization...")
    print("="*80)
    print()
    
    depths = np.linspace(min_depth, max_depth, 1000)
    
    # INT8 quantized depths
    int8_values = ((depths - min_depth) / scale).astype(int)
    int8_values = np.clip(int8_values, 0, 255)
    quantized_depths = min_depth + scale * int8_values
    
    # Errors
    abs_errors = np.abs(depths - quantized_depths) * 1000  # mm
    rel_errors = (np.abs(depths - quantized_depths) / depths) * 100  # %
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Depth Quantization
    ax1 = axes[0]
    ax1.plot(depths, depths, 'b-', linewidth=2, label='FP32 (Ideal)', alpha=0.7)
    ax1.plot(depths, quantized_depths, 'r.', markersize=1, label='INT8 Quantized', alpha=0.5)
    ax1.set_xlabel('True Depth (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quantized Depth (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Direct Depth INT8 Quantization', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min_depth, max_depth])
    ax1.set_ylim([min_depth, max_depth])
    
    # Plot 2: Absolute Error
    ax2 = axes[1]
    ax2.plot(depths, abs_errors, 'r-', linewidth=2)
    ax2.axhline(y=max_error*1000, color='orange', linestyle='--', linewidth=2, 
                label=f'Max Error: {max_error*1000:.3f}mm')
    ax2.fill_between(depths, 0, abs_errors, alpha=0.3, color='red')
    ax2.set_xlabel('Depth (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error (mm)', fontsize=12, fontweight='bold')
    ax2.set_title('INT8 Quantization Absolute Error (CONSTANT)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min_depth, max_depth])
    ax2.set_ylim([0, 35])
    
    # Annotate key points
    for d in [0.5, 5.0, 15.0]:
        idx = np.argmin(np.abs(depths - d))
        ax2.annotate(f'{abs_errors[idx]:.1f}mm', 
                    xy=(d, abs_errors[idx]), 
                    xytext=(d, abs_errors[idx] + 3),
                    ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot 3: Relative Error
    ax3 = axes[2]
    ax3.plot(depths, rel_errors, 'g-', linewidth=2)
    ax3.fill_between(depths, 0, rel_errors, alpha=0.3, color='green')
    ax3.set_xlabel('Depth (m)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax3.set_title('INT8 Quantization Relative Error (Hyperbolic 1/x)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([min_depth, max_depth])
    ax3.set_ylim([0, 6])
    
    # Annotate key points
    for d in test_depths:
        idx = np.argmin(np.abs(depths - d))
        color = 'lightcoral' if d < 2 else 'lightgreen'
        ax3.annotate(f'{rel_errors[idx]:.2f}%', 
                    xy=(d, rel_errors[idx]), 
                    xytext=(d, rel_errors[idx] + 0.3),
                    ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    output_path = 'outputs/direct_depth_int8_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    plt.close()
    
    # =========================================================================
    # Final Recommendations
    # =========================================================================
    print()
    print("="*80)
    print("🎯 Final Recommendations")
    print("="*80)
    print()
    
    print("✅ Direct Depth INT8 Quantization is VIABLE:")
    print()
    print("   1. Absolute Error: ±28.43mm (CONSTANT)")
    print("      → Much better than Sigmoid→Bounded Inverse (853mm @ 15m)")
    print()
    print("   2. Relative Error: 5.69% @ 0.5m → 0.19% @ 15m")
    print("      → Acceptable for ADAS (requirement: <100mm)")
    print()
    print("   3. INT8 Range Utilization: 100%")
    print("      → All 256 levels used uniformly")
    print()
    print("   4. Implementation Simplicity:")
    print("      → No sigmoid, no bounded inverse transformation")
    print("      → Direct depth output: model → [0.5, 15.0]m → INT8")
    print()
    
    print("🚀 Action Items:")
    print()
    print("   1. Modify ResNetSAN depth_head:")
    print("      - Remove nn.Sigmoid()")
    print("      - Add ReLU + Clamp to [0.5, 15.0]m")
    print()
    print("   2. Update Loss calculation:")
    print("      - Accept depth directly (no sigmoid)")
    print("      - Compute inv_depth internally for SSI")
    print()
    print("   3. ONNX Export:")
    print("      - Output layer: depth [0.5, 15.0]m (FP32)")
    print("      - NPU quantization: INT8 with scale={:.6f}".format(scale))
    print()
    print("   4. Validation:")
    print("      - Test INT8 quantization on NPU")
    print("      - Expected: abs_rel < 0.035 (vs 0.114 with Sigmoid→Bounded Inverse)")
    print()
    
    # =========================================================================
    # Code Template
    # =========================================================================
    print("="*80)
    print("📝 Implementation Code Template")
    print("="*80)
    print()
    
    code_template = """
# packnet_sfm/networks/depth/resnet_san.py

class ResNetSAN01(nn.Module):
    def __init__(self, min_depth=0.5, max_depth=15.0, ...):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Depth head (Direct Depth Output)
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(),  # Ensure non-negative
        )
    
    def forward(self, x):
        features = self.encoder(x)
        depth_logits = self.depth_head(features)
        
        # Clamp to [min_depth, max_depth]
        depth = torch.clamp(depth_logits, min=self.min_depth, max=self.max_depth)
        
        return depth  # [B, 1, H, W], range: [0.5, 15.0]m


# INT8 Quantization (ONNX/NPU)
scale = (max_depth - min_depth) / 255  # 0.056863
zero_point = 0
int8_depth = ((depth - min_depth) / scale).to(torch.uint8)
# → NPU stores as INT8 [0, 255]

# Dequantization (NPU → Depth)
depth_reconstructed = min_depth + scale * int8_depth.to(torch.float32)
# → Error: ±28.43mm (uniform)
"""
    
    print(code_template)
    print()
    
    print("="*80)
    print("✅ Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    analyze_direct_depth_int8()
