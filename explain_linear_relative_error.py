#!/usr/bin/env python3
"""
Linear 양자화에서 상대 오차가 근거리에서 높아지는 이유 설명
"""

import numpy as np
import matplotlib.pyplot as plt

def explain_relative_error():
    """왜 Linear에서 relative error가 근거리에서 높은가?"""
    
    print("="*80)
    print("Why Does Linear Quantization Have Higher Relative Error at Near-field?")
    print("="*80)
    print()
    
    # Linear quantization parameters
    min_depth = 0.5
    max_depth = 15.0
    int8_levels = 256
    
    depth_range = max_depth - min_depth
    step_size = depth_range / (int8_levels - 1)
    
    print("📊 Linear Quantization:")
    print(f"   Range: [{min_depth}, {max_depth}]m")
    print(f"   Step size: {step_size:.6f}m = {step_size*1000:.3f}mm")
    print(f"   Max absolute error: ±{step_size/2:.6f}m = ±{step_size/2*1000:.3f}mm")
    print()
    
    # =========================================================================
    # Mathematical Explanation
    # =========================================================================
    print("="*80)
    print("📐 MATHEMATICAL EXPLANATION")
    print("="*80)
    print()
    
    print("Linear quantization gives CONSTANT absolute error:")
    print(f"   Δdepth_abs = ±{step_size/2*1000:.3f}mm (constant for all depths)")
    print()
    
    print("But RELATIVE error depends on the depth value:")
    print()
    print("   Relative Error (%) = (Absolute Error / Depth) × 100")
    print("                      = (±28.431mm / Depth) × 100")
    print()
    
    # Example calculations
    test_depths = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    abs_error = step_size / 2
    
    print("Examples:")
    print(f"{'Depth':<8} {'Absolute Error':<18} {'Calculation':<30} {'Relative Error':<15}")
    print("-"*80)
    
    for d in test_depths:
        rel_error = (abs_error / d) * 100
        calc_str = f"{abs_error*1000:.3f}mm / {d}m"
        print(f"{d:5.1f}m   ±{abs_error*1000:6.3f}mm          "
              f"{calc_str:<30} {rel_error:5.2f}%")
    
    print()
    print("🔍 Key Insight:")
    print("   Same absolute error (28.431mm) divided by SMALLER depth")
    print("   → LARGER relative error percentage")
    print()
    print("   28.431mm / 0.5m  = 5.69%  ← HIGH")
    print("   28.431mm / 15.0m = 0.19%  ← LOW")
    print()
    
    # =========================================================================
    # Analogy
    # =========================================================================
    print("="*80)
    print("🎯 ANALOGY: Money Example")
    print("="*80)
    print()
    
    print("Imagine you make a $10 calculation error:")
    print()
    print("   When calculating $100:   $10 error = 10% error  ← HIGH")
    print("   When calculating $1000:  $10 error = 1% error   ← MEDIUM")
    print("   When calculating $10000: $10 error = 0.1% error ← LOW")
    print()
    print("Same absolute error ($10), but relative impact differs!")
    print()
    
    # =========================================================================
    # Visual Comparison
    # =========================================================================
    print("="*80)
    print("📊 VISUAL COMPARISON")
    print("="*80)
    print()
    
    depths = np.linspace(min_depth, max_depth, 100)
    abs_errors = np.full_like(depths, abs_error * 1000)  # mm
    rel_errors = (abs_error / depths) * 100  # %
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Absolute Error (constant)
    ax1 = axes[0]
    ax1.plot(depths, abs_errors, 'b-', linewidth=3)
    ax1.axhline(y=abs_error*1000, color='r', linestyle='--', alpha=0.7, 
                label=f'Constant: {abs_error*1000:.3f}mm')
    ax1.fill_between(depths, abs_errors*0.95, abs_errors*1.05, alpha=0.2)
    ax1.set_xlabel('Depth (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Absolute Error (mm)', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Quantization: CONSTANT Absolute Error', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min_depth, max_depth])
    ax1.set_ylim([0, 40])
    
    # Annotate key points
    for d in [0.5, 5.0, 15.0]:
        ax1.annotate(f'{abs_error*1000:.1f}mm', 
                    xy=(d, abs_error*1000), 
                    xytext=(d, abs_error*1000 + 5),
                    ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Relative Error (hyperbolic)
    ax2 = axes[1]
    ax2.plot(depths, rel_errors, 'r-', linewidth=3)
    ax2.fill_between(depths, 0, rel_errors, alpha=0.3, color='red')
    ax2.set_xlabel('Depth (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Linear Quantization: VARYING Relative Error (Hyperbolic 1/x curve)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min_depth, max_depth])
    ax2.set_ylim([0, 6])
    
    # Annotate key points
    for d in test_depths:
        rel_err = (abs_error / d) * 100
        ax2.annotate(f'{rel_err:.2f}%', 
                    xy=(d, rel_err), 
                    xytext=(d, rel_err + 0.3),
                    ha='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='lightcoral' if d < 2 else 'lightgreen', 
                             alpha=0.8))
    
    # Plot 3: Formula visualization
    ax3 = axes[2]
    
    # Create bars showing the division
    bar_data = []
    for d in test_depths:
        rel_err = (abs_error / d) * 100
        bar_data.append(rel_err)
    
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
    bars = ax3.bar(test_depths, bar_data, width=0.8, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (d, bar) in enumerate(zip(test_depths, bars)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{bar_data[i]:.2f}%\n({abs_error*1000:.1f}mm ÷ {d}m)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Depth (m)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Relative Error = 28.431mm ÷ Depth\n(Smaller denominator → Larger percentage)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xticks(test_depths)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 7])
    
    # Add formula annotation
    ax3.text(0.5, 0.95, 
            'Formula: Relative Error (%) = (28.431mm / Depth) × 100',
            transform=ax3.transAxes,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
            ha='center', va='top')
    
    plt.tight_layout()
    output_path = 'outputs/linear_relative_error_explanation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    plt.close()
    
    # =========================================================================
    # Mathematical Formula
    # =========================================================================
    print()
    print("="*80)
    print("📐 MATHEMATICAL FORMULA")
    print("="*80)
    print()
    
    print("For LINEAR quantization:")
    print()
    print("   Absolute Error = CONSTANT = step_size / 2")
    print()
    print("   Relative Error = (Absolute Error / Depth) × 100")
    print("                  = (C / Depth) × 100    (C = constant)")
    print("                  = C × (1 / Depth) × 100")
    print()
    print("This is a HYPERBOLIC function: y = C / x")
    print()
    print("Properties of y = 1/x:")
    print("   - As x → 0 (near-field), y → ∞ (error explodes)")
    print("   - As x → ∞ (far-field), y → 0 (error vanishes)")
    print("   - Asymptotic: never zero, never infinite in practical range")
    print()
    
    # =========================================================================
    # Practical Implications
    # =========================================================================
    print("="*80)
    print("💡 PRACTICAL IMPLICATIONS")
    print("="*80)
    print()
    
    print("Why this matters:")
    print()
    print("1. Near-field (0.5m):")
    print("   - Absolute error: 28mm (good!)")
    print("   - Relative error: 5.69% (concerning for precision tasks)")
    print("   - Example: 0.5m object detected as 0.472~0.528m")
    print()
    print("2. Far-field (15m):")
    print("   - Absolute error: 28mm (same)")
    print("   - Relative error: 0.19% (excellent!)")
    print("   - Example: 15m object detected as 14.972~15.028m")
    print()
    print("Trade-off:")
    print("   ✅ Far-field gets VERY accurate (0.19% error)")
    print("   ⚠️  Near-field gets LESS accurate (5.69% error)")
    print()
    
    # =========================================================================
    # Comparison with Other Methods
    # =========================================================================
    print("="*80)
    print("🔄 COMPARISON WITH OTHER METHODS")
    print("="*80)
    print()
    
    print("Why Bounded Inverse is different:")
    print()
    print("   Bounded Inverse:")
    print("      - Absolute error INCREASES with depth (exponentially)")
    print("      - At 15m: 853mm absolute error!")
    print("      - Relative error stays ~5.69% (coincidentally same)")
    print()
    print("   Linear:")
    print("      - Absolute error CONSTANT (28mm)")
    print("      - Relative error DECREASES with depth (hyperbolically)")
    print()
    print("   Log-space:")
    print("      - Absolute error INCREASES linearly with depth")
    print("      - Relative error CONSTANT (0.67%)")
    print()
    
    # =========================================================================
    # Conclusion
    # =========================================================================
    print()
    print("="*80)
    print("🎯 CONCLUSION")
    print("="*80)
    print()
    
    print("Linear quantization's relative error is HIGH at near-field because:")
    print()
    print("   1. Absolute error is CONSTANT (28.431mm)")
    print("   2. Relative error = Constant / Depth")
    print("   3. Smaller depth (denominator) → Larger percentage")
    print()
    print("This is NOT a flaw, but a mathematical property of:")
    print("   Relative Error (%) = Absolute Error / Reference Value")
    print()
    print("Analogy: $10 error on $100 purchase (10%) vs $10,000 purchase (0.1%)")
    print()


if __name__ == '__main__':
    explain_relative_error()
