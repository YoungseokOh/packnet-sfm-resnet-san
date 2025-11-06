#!/usr/bin/env python3
"""
PyTorch vs NPU Sigmoid Output 비교

목적:
1. Sigmoid 값 직접 비교 (pixel-wise difference)
2. Linear transformation 범위 확인 (0.5~15m)
3. 통계 및 시각화
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def compare_sigmoid_outputs():
    """Compare PyTorch and NPU sigmoid outputs"""
    
    pytorch_dir = 'outputs/raw_sigmoid_outputs'
    npu_dir = 'outputs/raw_sigmoid_npu_outputs'
    
    image_ids = [
        '0000000038',
        '0000000056',
        '0000000067',
        '0000000077',
        '0000000080'
    ]
    
    print(f"\n{'='*70}")
    print(f"PyTorch vs NPU Sigmoid Comparison")
    print(f"{'='*70}\n")
    
    all_diffs = []
    
    for idx, image_id in enumerate(image_ids, 1):
        print(f"\n{'='*70}")
        print(f"Image {idx}/5: {image_id}")
        print(f"{'='*70}")
        
        # Load PyTorch sigmoid
        pytorch_path = os.path.join(pytorch_dir, f'{image_id}_sigmoid.npy')
        if not os.path.exists(pytorch_path):
            print(f"  ❌ PyTorch file not found: {pytorch_path}")
            continue
        
        pytorch_sig = np.load(pytorch_path)
        print(f"  PyTorch sigmoid:")
        print(f"    Path: {pytorch_path}")
        print(f"    Shape: {pytorch_sig.shape}")
        print(f"    Range: [{pytorch_sig.min():.6f}, {pytorch_sig.max():.6f}]")
        print(f"    Mean: {pytorch_sig.mean():.6f}, Std: {pytorch_sig.std():.6f}")
        
        # Load NPU sigmoid
        npu_path = os.path.join(npu_dir, f'{image_id}.npy')
        if not os.path.exists(npu_path):
            print(f"  ❌ NPU file not found: {npu_path}")
            continue
        
        npu_sig = np.load(npu_path)
        
        # Handle shape: squeeze to (H, W)
        while npu_sig.ndim > 2:
            npu_sig = npu_sig.squeeze(0)
        
        print(f"\n  NPU sigmoid:")
        print(f"    Path: {npu_path}")
        print(f"    Shape: {npu_sig.shape}")
        print(f"    Range: [{npu_sig.min():.6f}, {npu_sig.max():.6f}]")
        print(f"    Mean: {npu_sig.mean():.6f}, Std: {npu_sig.std():.6f}")
        
        # Check shape match
        if pytorch_sig.shape != npu_sig.shape:
            print(f"  ⚠️ Shape mismatch! PyTorch: {pytorch_sig.shape}, NPU: {npu_sig.shape}")
            continue
        
        # Compute pixel-wise difference
        diff = np.abs(pytorch_sig - npu_sig)
        all_diffs.append(diff)
        
        print(f"\n  📊 Pixel-wise Difference (|PyTorch - NPU|):")
        print(f"    Mean diff: {diff.mean():.6f}")
        print(f"    Max diff: {diff.max():.6f}")
        print(f"    Std diff: {diff.std():.6f}")
        print(f"    Median diff: {np.median(diff):.6f}")
        
        # Difference distribution
        thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
        print(f"\n  Pixels with difference > threshold:")
        for thresh in thresholds:
            count = (diff > thresh).sum()
            pct = 100 * count / diff.size
            print(f"    > {thresh:.3f}: {count:6d} pixels ({pct:5.2f}%)")
        
        # Linear transformation check (0.5m ~ 15.0m)
        min_depth = 0.5
        max_depth = 15.0
        
        # PyTorch depth
        min_inv = 1.0 / max_depth
        max_inv = 1.0 / min_depth
        pytorch_inv = min_inv + (max_inv - min_inv) * pytorch_sig
        pytorch_depth = 1.0 / pytorch_inv
        
        # NPU depth
        npu_inv = min_inv + (max_inv - min_inv) * npu_sig
        npu_depth = 1.0 / npu_inv
        
        print(f"\n  📏 Linear Transformation (0.5m ~ 15.0m):")
        print(f"    PyTorch depth range: [{pytorch_depth.min():.2f}, {pytorch_depth.max():.2f}]m")
        print(f"    NPU depth range: [{npu_depth.min():.2f}, {npu_depth.max():.2f}]m")
        
        depth_diff = np.abs(pytorch_depth - npu_depth)
        print(f"\n    Depth difference:")
        print(f"      Mean: {depth_diff.mean():.4f}m")
        print(f"      Max: {depth_diff.max():.4f}m")
        print(f"      Median: {np.median(depth_diff):.4f}m")
    
    # Overall summary
    if len(all_diffs) > 0:
        print(f"\n{'='*70}")
        print(f"Overall Summary ({len(all_diffs)} images)")
        print(f"{'='*70}")
        
        all_diffs_concat = np.concatenate([d.flatten() for d in all_diffs])
        
        print(f"\n  All pixels combined:")
        print(f"    Mean sigmoid diff: {all_diffs_concat.mean():.6f}")
        print(f"    Max sigmoid diff: {all_diffs_concat.max():.6f}")
        print(f"    Std sigmoid diff: {all_diffs_concat.std():.6f}")
        print(f"    Median sigmoid diff: {np.median(all_diffs_concat):.6f}")
        
        print(f"\n  Difference distribution (all pixels):")
        for thresh in [0.001, 0.005, 0.01, 0.05, 0.1]:
            count = (all_diffs_concat > thresh).sum()
            pct = 100 * count / all_diffs_concat.size
            print(f"    > {thresh:.3f}: {count:8d} pixels ({pct:5.2f}%)")
        
        # Save histogram
        output_dir = 'outputs/sigmoid_comparison'
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Histogram
        plt.subplot(1, 2, 1)
        plt.hist(all_diffs_concat, bins=100, edgecolor='black', alpha=0.7)
        plt.xlabel('|PyTorch - NPU| Sigmoid Difference')
        plt.ylabel('Pixel Count')
        plt.title('Sigmoid Difference Distribution')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: CDF
        plt.subplot(1, 2, 2)
        sorted_diffs = np.sort(all_diffs_concat)
        cdf = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
        plt.plot(sorted_diffs, cdf, linewidth=2)
        plt.xlabel('|PyTorch - NPU| Sigmoid Difference')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 0.1])
        
        plt.tight_layout()
        hist_path = os.path.join(output_dir, 'sigmoid_difference_distribution.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✅ Histogram saved: {hist_path}")
        plt.close()
        
        # Save summary to text file
        summary_path = os.path.join(output_dir, 'comparison_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("PyTorch vs NPU Sigmoid Comparison Summary\n")
            f.write("="*70 + "\n\n")
            f.write(f"Images compared: {len(all_diffs)}\n")
            f.write(f"Total pixels: {all_diffs_concat.size:,}\n\n")
            f.write(f"Sigmoid Difference Statistics:\n")
            f.write(f"  Mean:   {all_diffs_concat.mean():.6f}\n")
            f.write(f"  Max:    {all_diffs_concat.max():.6f}\n")
            f.write(f"  Std:    {all_diffs_concat.std():.6f}\n")
            f.write(f"  Median: {np.median(all_diffs_concat):.6f}\n\n")
            f.write(f"Pixels with difference > threshold:\n")
            for thresh in [0.001, 0.005, 0.01, 0.05, 0.1]:
                count = (all_diffs_concat > thresh).sum()
                pct = 100 * count / all_diffs_concat.size
                f.write(f"  > {thresh:.3f}: {count:8d} pixels ({pct:5.2f}%)\n")
        
        print(f"  ✅ Summary saved: {summary_path}")
    
    print(f"\n{'='*70}")
    print("Comparison complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    compare_sigmoid_outputs()
