#!/usr/bin/env python3
"""
Visualize FP32 vs NPU depth predictions
Creates 6-panel visualization:
  Row 1: RGB, GT (scatter), GT-Inverse (scatter)
  Row 2: FP32, NPU, Diff (FP32-NPU, red colormap)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import cv2
from pathlib import Path
import argparse


def load_gt_depth(gt_path):
    """Load GT depth from 16-bit PNG (value/256 = meters)"""
    gt_img = Image.open(gt_path)
    gt_depth = np.array(gt_img, dtype=np.float32) / 256.0
    return gt_depth


def load_rgb(rgb_path):
    """Load RGB image"""
    rgb = Image.open(rgb_path)
    return np.array(rgb)


def load_fp32_depth(fp32_path):
    """Load FP32 depth prediction"""
    return np.load(fp32_path)


def load_npu_depth(int_path, frac_path, max_depth=15.0):
    """Load and compose NPU dual-head depth"""
    integer_sigmoid = np.load(int_path).squeeze()
    fractional_sigmoid = np.load(frac_path).squeeze()
    depth = integer_sigmoid * max_depth + fractional_sigmoid
    return depth


def visualize_comparison(rgb, gt_depth, fp32_depth, npu_depth, 
                        min_depth=0.5, max_depth=15.0):
    """
    Create 6-panel visualization
    """
    # Create custom colormap: 0.5~0.75m=Pure Red, then 0.1m gradation steps
    depth_range = max_depth - min_depth  # 14.5m
    
    colors_positions = [
        # 0.5m ~ 0.75m: Pure Red (위험 구역)
        ((0.5 - 0.5) / depth_range, (1.0, 0.0, 0.0)),   # 0.5m: Pure Red
        ((0.75 - 0.5) / depth_range, (1.0, 0.0, 0.0)),  # 0.75m: Pure Red
        
        # 0.75m ~ 3.0m: 0.1m steps (Red → Orange → Yellow)
        ((0.85 - 0.5) / depth_range, (1.0, 0.1, 0.0)),  # 0.85m
        ((0.95 - 0.5) / depth_range, (1.0, 0.2, 0.0)),  # 0.95m
        ((1.05 - 0.5) / depth_range, (1.0, 0.3, 0.0)),  # 1.05m
        ((1.15 - 0.5) / depth_range, (1.0, 0.4, 0.0)),  # 1.15m
        ((1.25 - 0.5) / depth_range, (1.0, 0.5, 0.0)),  # 1.25m: Orange
        ((1.35 - 0.5) / depth_range, (1.0, 0.55, 0.0)), # 1.35m
        ((1.45 - 0.5) / depth_range, (1.0, 0.6, 0.0)),  # 1.45m
        ((1.55 - 0.5) / depth_range, (1.0, 0.65, 0.0)), # 1.55m
        ((1.65 - 0.5) / depth_range, (1.0, 0.7, 0.0)),  # 1.65m
        ((1.75 - 0.5) / depth_range, (1.0, 0.75, 0.0)), # 1.75m
        ((1.85 - 0.5) / depth_range, (1.0, 0.8, 0.0)),  # 1.85m
        ((1.95 - 0.5) / depth_range, (1.0, 0.85, 0.0)), # 1.95m
        ((2.05 - 0.5) / depth_range, (1.0, 0.88, 0.0)), # 2.05m
        ((2.15 - 0.5) / depth_range, (1.0, 0.91, 0.0)), # 2.15m
        ((2.25 - 0.5) / depth_range, (1.0, 0.94, 0.0)), # 2.25m
        ((2.35 - 0.5) / depth_range, (1.0, 0.97, 0.0)), # 2.35m
        ((2.45 - 0.5) / depth_range, (1.0, 1.0, 0.0)),  # 2.45m: Yellow
        ((2.55 - 0.5) / depth_range, (0.97, 1.0, 0.03)), # 2.55m
        ((2.65 - 0.5) / depth_range, (0.94, 1.0, 0.06)), # 2.65m
        ((2.75 - 0.5) / depth_range, (0.91, 1.0, 0.09)), # 2.75m
        ((2.85 - 0.5) / depth_range, (0.88, 1.0, 0.12)), # 2.85m
        ((2.95 - 0.5) / depth_range, (0.85, 1.0, 0.15)), # 2.95m
        
        # 3.0m ~ 6.0m: 0.2m steps (Yellow → Green)
        ((3.2 - 0.5) / depth_range, (0.8, 1.0, 0.2)),   # 3.2m
        ((3.4 - 0.5) / depth_range, (0.7, 1.0, 0.3)),   # 3.4m
        ((3.6 - 0.5) / depth_range, (0.6, 1.0, 0.4)),   # 3.6m
        ((3.8 - 0.5) / depth_range, (0.5, 1.0, 0.5)),   # 3.8m
        ((4.0 - 0.5) / depth_range, (0.4, 1.0, 0.6)),   # 4.0m: Green
        ((4.2 - 0.5) / depth_range, (0.35, 1.0, 0.65)), # 4.2m
        ((4.4 - 0.5) / depth_range, (0.3, 1.0, 0.7)),   # 4.4m
        ((4.6 - 0.5) / depth_range, (0.25, 1.0, 0.75)), # 4.6m
        ((4.8 - 0.5) / depth_range, (0.2, 1.0, 0.8)),   # 4.8m
        ((5.0 - 0.5) / depth_range, (0.15, 1.0, 0.85)), # 5.0m
        ((5.2 - 0.5) / depth_range, (0.1, 1.0, 0.9)),   # 5.2m
        ((5.4 - 0.5) / depth_range, (0.05, 1.0, 0.95)), # 5.4m
        ((5.6 - 0.5) / depth_range, (0.0, 1.0, 1.0)),   # 5.6m: Cyan
        
        # 6.0m ~ 10.0m: 0.5m steps (Cyan → Blue)
        ((6.0 - 0.5) / depth_range, (0.0, 0.95, 1.0)),  # 6.0m
        ((6.5 - 0.5) / depth_range, (0.0, 0.9, 1.0)),   # 6.5m
        ((7.0 - 0.5) / depth_range, (0.0, 0.85, 1.0)),  # 7.0m
        ((7.5 - 0.5) / depth_range, (0.0, 0.8, 1.0)),   # 7.5m
        ((8.0 - 0.5) / depth_range, (0.0, 0.7, 1.0)),   # 8.0m
        ((8.5 - 0.5) / depth_range, (0.0, 0.6, 1.0)),   # 8.5m
        ((9.0 - 0.5) / depth_range, (0.0, 0.5, 1.0)),   # 9.0m
        ((9.5 - 0.5) / depth_range, (0.0, 0.4, 1.0)),   # 9.5m
        ((10.0 - 0.5) / depth_range, (0.0, 0.3, 1.0)),  # 10.0m: Blue
        
        # 10.0m ~ 15.0m: 1m steps (Blue)
        ((11.0 - 0.5) / depth_range, (0.0, 0.2, 1.0)),  # 11.0m
        ((12.0 - 0.5) / depth_range, (0.0, 0.15, 1.0)), # 12.0m
        ((13.0 - 0.5) / depth_range, (0.0, 0.1, 1.0)),  # 13.0m
        ((14.0 - 0.5) / depth_range, (0.0, 0.05, 1.0)), # 14.0m
        ((15.0 - 0.5) / depth_range, (0.0, 0.0, 1.0)),  # 15.0m: Pure Blue
    ]
    
    # Normalize positions to [0, 1]
    positions = [p[0] for p in colors_positions]
    colors = [p[1] for p in colors_positions]
    
    # Create colormap
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list('depth_custom', 
                                                     list(zip(positions, colors)), 
                                                     N=512)  # Higher resolution
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Mask valid depth
    valid_mask = (gt_depth > min_depth) & (gt_depth < max_depth)
    
    # Row 1: RGB, GT, GT-Inverse
    # Panel 1: RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Panel 2: GT (scatter, larger points)
    axes[0, 1].imshow(np.ones_like(rgb) * 255)  # White background
    y_coords, x_coords = np.where(valid_mask)
    gt_values = gt_depth[valid_mask]
    scatter1 = axes[0, 1].scatter(x_coords, y_coords, c=gt_values, 
                                   cmap=custom_cmap, s=25, marker='s', alpha=0.8,
                                   vmin=min_depth, vmax=max_depth)
    axes[0, 1].set_xlim(0, rgb.shape[1])
    axes[0, 1].set_ylim(rgb.shape[0], 0)
    axes[0, 1].set_title('GT Depth', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label('Depth (m)', fontsize=10)
    
    # Panel 3: GT-Inverse (scatter, larger points)
    axes[0, 2].imshow(np.ones_like(rgb) * 255)  # White background
    gt_inverse = 1.0 / gt_values
    scatter2 = axes[0, 2].scatter(x_coords, y_coords, c=gt_inverse, 
                                   cmap='plasma', s=25, marker='s', alpha=0.8,
                                   vmin=1.0/max_depth, vmax=1.0/min_depth)
    axes[0, 2].set_xlim(0, rgb.shape[1])
    axes[0, 2].set_ylim(rgb.shape[0], 0)
    axes[0, 2].set_title('GT Inverse Depth', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar2.set_label('Inverse Depth (1/m)', fontsize=10)
    
    # Row 2: FP32, NPU, Diff
    # Panel 4: FP32 Depth (scatter, valid GT pixels only)
    axes[1, 0].imshow(np.ones_like(rgb) * 255)  # White background
    fp32_values_valid = fp32_depth[valid_mask]
    scatter3 = axes[1, 0].scatter(x_coords, y_coords, c=fp32_values_valid, 
                                   cmap=custom_cmap, s=25, marker='s', alpha=0.8,
                                   vmin=min_depth, vmax=max_depth)
    axes[1, 0].set_xlim(0, rgb.shape[1])
    axes[1, 0].set_ylim(rgb.shape[0], 0)
    axes[1, 0].set_title('FP32 Prediction', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    cbar3 = plt.colorbar(scatter3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar3.set_label('Depth (m)', fontsize=10)
    
    # Panel 5: NPU Depth (scatter, valid GT pixels only)
    axes[1, 1].imshow(np.ones_like(rgb) * 255)  # White background
    npu_values_valid = npu_depth[valid_mask]
    scatter4 = axes[1, 1].scatter(x_coords, y_coords, c=npu_values_valid, 
                                   cmap=custom_cmap, s=25, marker='s', alpha=0.8,
                                   vmin=min_depth, vmax=max_depth)
    axes[1, 1].set_xlim(0, rgb.shape[1])
    axes[1, 1].set_ylim(rgb.shape[0], 0)
    axes[1, 1].set_title('NPU Prediction', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    cbar4 = plt.colorbar(scatter4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar4.set_label('Depth (m)', fontsize=10)
    
        # Panel 6: Relative Error Rate (scatter, valid GT pixels only, red colormap)
    # Show relative error rate (%) = |GT - NPU| / GT * 100
    diff_abs = np.abs(gt_depth - npu_depth)
    # Avoid division by zero
    error_rate = np.zeros_like(gt_depth)
    valid_nonzero = (valid_mask) & (gt_depth != 0)
    error_rate[valid_nonzero] = (diff_abs[valid_nonzero] / gt_depth[valid_nonzero]) * 100
    
    error_rate_valid = error_rate[valid_mask]
    
    # Compute error rate statistics
    error_mean = np.mean(error_rate_valid)
    error_max = np.max(error_rate_valid)
    
    # Red colormap for error rate
    axes[1, 2].imshow(np.ones_like(rgb) * 255)  # White background
    # Use 99th percentile to show most errors clearly while clipping extreme outliers
    error_percentile_99 = np.percentile(error_rate_valid, 99)
    error_lim = max(error_percentile_99, 1.0)  # At least 1% range
    
    scatter5 = axes[1, 2].scatter(x_coords, y_coords, c=error_rate_valid, 
                                   cmap='Reds', s=25, marker='s', alpha=0.8,
                                   vmin=0, vmax=error_lim)
    axes[1, 2].set_xlim(0, rgb.shape[1])
    axes[1, 2].set_ylim(rgb.shape[0], 0)
    axes[1, 2].set_title(f'Relative Error Rate |GT - NPU| / GT (%)', 
                         fontsize=13, fontweight='bold')
    axes[1, 2].axis('off')
    cbar5 = plt.colorbar(scatter5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar5.set_label('Error Rate (%) *99th percentile', fontsize=9)
    
    plt.tight_layout()
    return fig, {
        'error_mean': error_mean,
        'error_std': np.std(error_rate_valid),
        'error_max': error_max,
        'valid_pixels': np.sum(valid_mask)
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize FP32 vs NPU depth predictions')
    parser.add_argument('--rgb_dir', type=str, default='Fin_Test_Set_ncdb/RGB',
                       help='Path to RGB images directory')
    parser.add_argument('--gt_dir', type=str, default='Fin_Test_Set_ncdb/GT',
                       help='Path to GT depth directory')
    parser.add_argument('--fp32_dir', type=str, default='Fin_Test_Set_ncdb/fp32',
                       help='Path to FP32 predictions directory')
    parser.add_argument('--npu_dir', type=str, 
                       default='Fin_Test_Set_ncdb/npu/resnetsan_dual_head_seperate_static',
                       help='Path to NPU predictions directory')
    parser.add_argument('--output_dir', type=str, 
                       default='outputs/fp32_vs_npu_comparison/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--min_depth', type=float, default=0.5,
                       help='Minimum depth for evaluation')
    parser.add_argument('--max_depth', type=float, default=15.0,
                       help='Maximum depth for evaluation')
    parser.add_argument('--dual_head_max_depth', type=float, default=15.0,
                       help='Max depth for NPU dual-head composition')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of files
    rgb_files = sorted([f for f in os.listdir(args.rgb_dir) if f.endswith('.png')])
    
    print(f"Found {len(rgb_files)} samples to visualize")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Statistics
    all_stats = []
    
    for i, rgb_file in enumerate(rgb_files):
        base_name = os.path.splitext(rgb_file)[0]
        
        # Construct paths
        rgb_path = os.path.join(args.rgb_dir, rgb_file)
        gt_path = os.path.join(args.gt_dir, f"{base_name}.png")
        fp32_path = os.path.join(args.fp32_dir, f"{base_name}.npy")
        npu_int_path = os.path.join(args.npu_dir, 'integer_sigmoid', f"{base_name}.npy")
        npu_frac_path = os.path.join(args.npu_dir, 'fractional_sigmoid', f"{base_name}.npy")
        
        # Check if all files exist
        if not all([os.path.exists(p) for p in [rgb_path, gt_path, fp32_path, 
                                                  npu_int_path, npu_frac_path]]):
            print(f"⚠️  Skipping {base_name}: missing files")
            continue
        
        # Load data
        rgb = load_rgb(rgb_path)
        gt_depth = load_gt_depth(gt_path)
        fp32_depth = load_fp32_depth(fp32_path)
        npu_depth = load_npu_depth(npu_int_path, npu_frac_path, args.dual_head_max_depth)
        
        # Create visualization
        fig, stats = visualize_comparison(rgb, gt_depth, fp32_depth, npu_depth,
                                         args.min_depth, args.max_depth)
        
        # Save
        output_path = os.path.join(args.output_dir, f"{base_name}.png")
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Store stats (convert numpy types to Python types)
        all_stats.append({
            'filename': base_name,
            'error_mean': float(stats['error_mean']),
            'error_std': float(stats['error_std']),
            'error_max': float(stats['error_max']),
            'valid_pixels': int(stats['valid_pixels'])
        })
        
        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(rgb_files):
            print(f"Processed {i+1}/{len(rgb_files)} samples...")
    
    # Save statistics
    import json
    stats_path = os.path.join(args.output_dir, 'comparison_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'per_sample': all_stats,
            'summary': {
                'total_samples': len(all_stats),
                'mean_error_rate': np.mean([s['error_mean'] for s in all_stats]),
                'mean_error_max': np.mean([s['error_max'] for s in all_stats]),
                'mean_valid_pixels': np.mean([s['valid_pixels'] for s in all_stats])
            }
        }, f, indent=2)
    
    print()
    print("✅ Visualization complete!")
    print(f"   Output: {args.output_dir}")
    print(f"   Statistics: {stats_path}")
    print()
    print("📊 Summary:")
    print(f"   Samples processed: {len(all_stats)}")
    print(f"   Mean Error Rate: {np.mean([s['error_mean'] for s in all_stats]):.2f}%")
    print(f"   Mean max error: {np.mean([s['error_max'] for s in all_stats]):.2f}%")


if __name__ == '__main__':
    main()
