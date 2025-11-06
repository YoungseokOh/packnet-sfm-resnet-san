"""
GT Depth Distribution Analysis for Optimal Quantization Range

이 스크립트는 실제 GT depth 분포를 분석하여 최적의 depth range를 결정합니다.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def load_gt_depth(filename, test_json_path):
    """Load GT depth from test set"""
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    # Find matching entry
    filename_key = filename.split('.')[0]  # e.g., '0000000567'
    for entry in test_data:
        if entry['new_filename'] == filename_key:
            # Construct depth path (using newest_depth_maps)
            dataset_root = Path(entry['dataset_root'])
            depth_path = dataset_root / 'newest_depth_maps' / f"{filename_key}.png"
            
            if not depth_path.exists():
                return None
            
            # Load depth (PNG format, divide by 256 for meters)
            depth = np.array(Image.open(depth_path), dtype=np.float32) / 256.0
            return depth
    
    return None

def analyze_depth_distribution(depth_values, percentiles=[50, 75, 90, 95, 99, 99.5, 99.9]):
    """Analyze depth distribution statistics"""
    valid_depths = depth_values[depth_values > 0]
    
    stats = {
        'count': len(valid_depths),
        'min': np.min(valid_depths),
        'max': np.max(valid_depths),
        'mean': np.mean(valid_depths),
        'median': np.median(valid_depths),
        'std': np.std(valid_depths),
        'percentiles': {}
    }
    
    for p in percentiles:
        stats['percentiles'][f'p{p}'] = np.percentile(valid_depths, p)
    
    return stats, valid_depths

def plot_depth_distribution(valid_depths, stats, output_path):
    """Plot depth distribution with range options"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histogram (linear scale)
    ax1 = axes[0, 0]
    bins = np.linspace(0, 20, 100)
    ax1.hist(valid_depths, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.2f}m")
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.2f}m")
    ax1.axvline(stats['percentiles']['p95'], color='orange', linestyle='--', linewidth=2, label=f"95%: {stats['percentiles']['p95']:.2f}m")
    ax1.axvline(stats['percentiles']['p99'], color='purple', linestyle='--', linewidth=2, label=f"99%: {stats['percentiles']['p99']:.2f}m")
    ax1.set_xlabel('Depth (m)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('GT Depth Distribution (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. CDF (Cumulative Distribution Function)
    ax2 = axes[0, 1]
    sorted_depths = np.sort(valid_depths)
    cdf = np.arange(1, len(sorted_depths) + 1) / len(sorted_depths) * 100
    ax2.plot(sorted_depths, cdf, linewidth=2, color='blue')
    
    # Mark key percentiles
    for p_name, p_value in [('50%', 50), ('75%', 75), ('90%', 90), ('95%', 95), ('99%', 99)]:
        depth_val = stats['percentiles'][f'p{p_value}']
        ax2.plot(depth_val, p_value, 'ro', markersize=8)
        ax2.text(depth_val + 0.3, p_value, f'{p_name}\n{depth_val:.2f}m', fontsize=9, va='center')
    
    # Range options
    ax2.axvline(7.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Option: 7.5m')
    ax2.axvline(10.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Option: 10.0m')
    ax2.axvline(15.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current: 15.0m')
    
    ax2.set_xlabel('Depth (m)', fontsize=12)
    ax2.set_ylabel('Cumulative %', fontsize=12)
    ax2.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 20)
    
    # 3. Histogram (log scale Y)
    ax3 = axes[1, 0]
    ax3.hist(valid_depths, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(stats['percentiles']['p95'], color='orange', linestyle='--', linewidth=2, label=f"95%: {stats['percentiles']['p95']:.2f}m")
    ax3.axvline(stats['percentiles']['p99'], color='purple', linestyle='--', linewidth=2, label=f"99%: {stats['percentiles']['p99']:.2f}m")
    ax3.set_xlabel('Depth (m)', fontsize=12)
    ax3.set_ylabel('Frequency (log scale)', fontsize=12)
    ax3.set_title('GT Depth Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Range coverage analysis
    ax4 = axes[1, 1]
    range_options = [
        (7.5, '7.5m', 'green'),
        (10.0, '10.0m', 'orange'),
        (12.5, '12.5m', 'blue'),
        (15.0, '15.0m (current)', 'red')
    ]
    
    coverages = []
    quant_errors = []
    labels = []
    
    for max_depth, label, color in range_options:
        coverage = np.sum(valid_depths <= max_depth) / len(valid_depths) * 100
        quant_step = (max_depth - 0.5) / 255
        quant_error = quant_step / 2 * 1000  # mm
        
        coverages.append(coverage)
        quant_errors.append(quant_error)
        labels.append(label)
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(x - width/2, coverages, width, label='Coverage %', color='skyblue', edgecolor='black')
    bars2 = ax4_twin.bar(x + width/2, quant_errors, width, label='Quant Error (mm)', color='salmon', edgecolor='black')
    
    ax4.set_xlabel('Range Option', fontsize=12)
    ax4.set_ylabel('Coverage (%)', fontsize=12, color='blue')
    ax4_twin.set_ylabel('Quantization Error (mm)', fontsize=12, color='red')
    ax4.set_title('Range Options: Coverage vs Quantization Error', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha='right')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax4.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{height1:.1f}%', ha='center', va='bottom', fontsize=9)
        ax4_twin.text(bar2.get_x() + bar2.get_width()/2., height2,
                     f'±{height2:.1f}mm', ha='center', va='bottom', fontsize=9)
    
    ax4.legend(loc='upper left', fontsize=10)
    ax4_twin.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Distribution plot saved to: {output_path}")
    plt.close()

def main():
    print("=" * 80)
    print("🔍 GT Depth Distribution Analysis for Optimal Quantization Range")
    print("=" * 80)
    
    # Paths
    npu_output_dir = Path('outputs/resnetsan_direct_depth_05_15_640x384')
    test_json_path = Path('/workspace/data/ncdb-cls-640x384/splits/combined_test.json')
    output_dir = Path('outputs/gt_depth_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all GT depths
    print("\n📂 Loading GT depths from 91 test images...")
    all_depths = []
    
    npu_files = sorted(npu_output_dir.glob('*.npy'))
    
    for i, npu_file in enumerate(npu_files, 1):
        # Extract filename
        filename = npu_file.stem + '.png'
        
        # Load GT
        gt_depth = load_gt_depth(filename, test_json_path)
        if gt_depth is not None:
            all_depths.append(gt_depth)
            if i % 20 == 0:
                print(f"   Loaded {i}/{len(npu_files)} images...")
    
    print(f"✅ Loaded {len(all_depths)} GT depth maps")
    
    # Combine all depths
    all_depths_array = np.concatenate([d.flatten() for d in all_depths])
    
    # Analyze distribution
    print("\n📊 Analyzing depth distribution...")
    stats, valid_depths = analyze_depth_distribution(all_depths_array)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("📈 GT Depth Statistics")
    print("=" * 80)
    print(f"Total pixels:        {stats['count']:,}")
    print(f"Min depth:           {stats['min']:.3f}m")
    print(f"Max depth:           {stats['max']:.3f}m")
    print(f"Mean depth:          {stats['mean']:.3f}m")
    print(f"Median depth:        {stats['median']:.3f}m")
    print(f"Std deviation:       {stats['std']:.3f}m")
    
    print("\n📊 Percentiles:")
    for p_name, p_value in stats['percentiles'].items():
        percentile_num = float(p_name[1:])
        print(f"  {p_name:>8} : {p_value:6.3f}m  ({100-percentile_num:5.1f}% pixels beyond this)")
    
    # Analyze range options
    print("\n" + "=" * 80)
    print("🎯 Range Options Analysis")
    print("=" * 80)
    
    range_options = [
        (7.5, "Aggressive"),
        (10.0, "Balanced"),
        (12.5, "Conservative"),
        (15.0, "Current")
    ]
    
    print(f"\n{'Range':<15} {'Coverage':<12} {'Lost':<12} {'Quant Step':<15} {'Quant Error':<15} {'RMSE Estimate'}")
    print("-" * 90)
    
    for max_depth, name in range_options:
        coverage = np.sum(valid_depths <= max_depth) / len(valid_depths) * 100
        lost = 100 - coverage
        quant_step = (max_depth - 0.5) / 255
        quant_error = quant_step / 2 * 1000  # mm
        
        # Estimate RMSE (simplified model)
        # Assume quantization error adds in quadrature to FP32 error
        fp32_rmse = 0.390  # from epoch 29
        quant_rmse = quant_error / 1000 / np.sqrt(3)  # Uniform distribution std
        estimated_rmse = np.sqrt(fp32_rmse**2 + quant_rmse**2 + 0.1**2)  # +0.1 for non-linear effects
        
        print(f"[0.5, {max_depth:4.1f}]m {name:<6} {coverage:5.1f}%      {lost:5.1f}%      "
              f"{quant_step*1000:5.1f}mm        ±{quant_error:5.1f}mm        ~{estimated_rmse:.3f}m")
    
    # Save statistics
    stats_output = output_dir / 'gt_depth_statistics.json'
    with open(stats_output, 'w') as f:
        # Convert numpy types to Python types for JSON
        stats_json = {
            'total_pixels': int(stats['count']),
            'min_depth': float(stats['min']),
            'max_depth': float(stats['max']),
            'mean_depth': float(stats['mean']),
            'median_depth': float(stats['median']),
            'std_depth': float(stats['std']),
            'percentiles': {k: float(v) for k, v in stats['percentiles'].items()}
        }
        json.dump(stats_json, f, indent=2)
    print(f"\n✅ Statistics saved to: {stats_output}")
    
    # Plot distribution
    plot_path = output_dir / 'gt_depth_distribution.png'
    plot_depth_distribution(valid_depths, stats, plot_path)
    
    # Recommendation
    print("\n" + "=" * 80)
    print("💡 Recommendation")
    print("=" * 80)
    
    p95 = stats['percentiles']['p95']
    p99 = stats['percentiles']['p99']
    
    if p95 < 7.5:
        recommended = 7.5
        reason = f"95% of pixels < 7.5m (p95={p95:.2f}m)"
    elif p95 < 10.0:
        recommended = 10.0
        reason = f"95% of pixels < 10.0m (p95={p95:.2f}m)"
    elif p99 < 12.5:
        recommended = 12.5
        reason = f"99% of pixels < 12.5m (p99={p99:.2f}m)"
    else:
        recommended = 15.0
        reason = f"Significant pixels > 12.5m (p99={p99:.2f}m)"
    
    print(f"\n✅ Recommended range: [0.5, {recommended}]m")
    print(f"   Reason: {reason}")
    
    quant_step = (recommended - 0.5) / 255
    quant_error = quant_step / 2 * 1000
    print(f"\n   Quantization step: {quant_step*1000:.1f}mm")
    print(f"   Quantization error: ±{quant_error:.1f}mm")
    
    if recommended < 15.0:
        improvement = (28.4 / quant_error - 1) * 100
        print(f"   Improvement vs current: {improvement:.1f}% better quantization!")
    
    coverage = np.sum(valid_depths <= recommended) / len(valid_depths) * 100
    print(f"   Coverage: {coverage:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
    
    return stats

if __name__ == '__main__':
    main()
