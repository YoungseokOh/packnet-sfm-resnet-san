#!/usr/bin/env python3
"""
NPU Direct Depth 결과 시각화 (Best 5 & Worst 5)

abs_rel 기준으로 best/worst 선택 후 시각화
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os


def load_gt_depth(new_filename, test_json_path, rgb_dir='assets/test_set_rgb'):
    """GT depth 로드 (combined_test.json 기반)"""
    with open(test_json_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        if entry['new_filename'] == new_filename:
            dataset_root = entry['dataset_root']
            depth_path = Path(dataset_root) / 'newest_depth_maps' / f'{new_filename}.png'
            
            if not depth_path.exists():
                raise FileNotFoundError(f"GT depth not found: {depth_path}")
            
            # PNG 16-bit 로드 → meters
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img, dtype=np.float32) / 256.0
            
            # RGB 이미지 로드 (assets/test_set_rgb에서)
            rgb_path = Path(rgb_dir) / f'{new_filename}.png'
            if rgb_path.exists():
                rgb = np.array(Image.open(rgb_path))
            else:
                # Fallback: dataset_root에서도 찾아보기
                rgb_path_alt = Path(dataset_root) / 'images' / f'{new_filename}.png'
                if rgb_path_alt.exists():
                    rgb = np.array(Image.open(rgb_path_alt))
                else:
                    rgb = None
            
            return depth, rgb
    
    raise ValueError(f"new_filename {new_filename} not found in {test_json_path}")


def compute_depth_metrics(gt, pred, min_depth=0.5, max_depth=15.0):
    """Compute depth metrics"""
    valid_mask = (gt > min_depth) & (gt < max_depth)
    
    if valid_mask.sum() == 0:
        return None
    
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).mean()
    
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'a1': a1,
    }


def visualize_depth_comparison(rgb, gt_depth, pred_depth, metrics, filename, output_path, 
                                min_depth=0.5, max_depth=15.0):
    """Visualize RGB, GT, Pred, and Error map"""
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    
    fig = plt.figure(figsize=(20, 5))
    
    # Valid mask: GT > 0 AND in valid range
    valid_mask = (gt_depth > 0) & (gt_depth >= min_depth) & (gt_depth <= max_depth)
    
    print(f"  Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({100*valid_mask.sum()/valid_mask.size:.1f}%)")
    
    # 1. RGB Image
    ax1 = plt.subplot(1, 5, 1)
    if rgb is not None:
        ax1.imshow(rgb)
        ax1.set_title('RGB Input', fontsize=14, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'RGB Not Available', ha='center', va='center')
        ax1.set_title('RGB Input', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Ground Truth Depth
    ax2 = plt.subplot(1, 5, 2)
    # Create display array: black for invalid, colormap for valid
    gt_display = np.zeros_like(gt_depth)
    gt_display[~valid_mask] = min_depth  # Will be mapped to blue (far)
    gt_display[valid_mask] = gt_depth[valid_mask]
    
    # Reversed colormap: 가까울수록 빨강, 멀수록 파랑
    im2 = ax2.imshow(gt_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    
    # Black overlay for invalid pixels
    invalid_overlay = np.zeros((*gt_depth.shape, 4))
    invalid_overlay[~valid_mask] = [0, 0, 0, 1]  # Black with full opacity
    ax2.imshow(invalid_overlay)
    
    ax2.set_title('Ground Truth Depth\n(Valid pixels only)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # 3. Predicted Depth (Direct Depth) - GT 기준 유효 픽셀만
    ax3 = plt.subplot(1, 5, 3)
    # Resize prediction to GT size if needed
    if pred_depth.shape != gt_depth.shape:
        from scipy.ndimage import zoom
        scale_h = gt_depth.shape[0] / pred_depth.shape[0]
        scale_w = gt_depth.shape[1] / pred_depth.shape[1]
        pred_depth_resized = zoom(pred_depth, (scale_h, scale_w), order=1)
    else:
        pred_depth_resized = pred_depth
    
    pred_display = np.zeros_like(pred_depth_resized)
    pred_display[~valid_mask] = min_depth
    pred_display[valid_mask] = pred_depth_resized[valid_mask]
    
    # Reversed colormap
    im3 = ax3.imshow(pred_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    
    # Black overlay for invalid pixels
    ax3.imshow(invalid_overlay)
    
    ax3.set_title('NPU Direct Depth\n(GT valid pixels only)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # 4. Absolute Error Map
    ax4 = plt.subplot(1, 5, 4)
    error = np.abs(gt_depth - pred_depth_resized)
    error_display = np.zeros_like(error)
    error_display[~valid_mask] = 0
    error_display[valid_mask] = error[valid_mask]
    
    im4 = ax4.imshow(error_display, cmap='hot', vmin=0, vmax=1.0)
    
    # Black overlay for invalid pixels
    ax4.imshow(invalid_overlay)
    
    ax4.set_title('Absolute Error (m)\n(GT valid pixels only)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Error (m)')
    
    # 5. Metrics Text
    ax5 = plt.subplot(1, 5, 5)
    ax5.axis('off')
    
    # Extract valid pixels for statistics
    gt_valid = gt_depth[valid_mask]
    pred_valid = pred_depth_resized[valid_mask]
    error_valid = error[valid_mask]
    
    metrics_text = f"""
    📊 Metrics Summary
    {'='*30}
    
    File: {filename}
    
    Accuracy Metrics:
    • abs_rel:  {metrics['abs_rel']:.4f}
    • RMSE:     {metrics['rmse']:.3f}m
    • δ < 1.25: {metrics['a1']:.4f}
    
    Valid Pixels: {valid_mask.sum():,}
    ({100*valid_mask.sum()/valid_mask.size:.1f}%)
    
    Depth Range (Valid):
    • GT:   [{gt_valid.min():.2f}, {gt_valid.max():.2f}]m
    • Pred: [{pred_valid.min():.2f}, {pred_valid.max():.2f}]m
    
    Error Statistics:
    • Mean:   {error_valid.mean():.3f}m
    • Max:    {error_valid.max():.3f}m
    • Median: {np.median(error_valid):.3f}m
    """
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'NPU Direct Depth Analysis: {filename}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def main():
    # Configuration
    npu_output_dir = Path('outputs/resnetsan_direct_depth_05_15_640x384')
    test_json = '/workspace/data/ncdb-cls-640x384/splits/combined_test.json'
    output_dir = Path('outputs/direct_depth_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_depth = 0.5
    max_depth = 15.0
    
    # Get all NPU files
    npu_files = sorted(npu_output_dir.glob('*.npy'))
    npu_files = [f for f in npu_files if f.stem != 'evaluation_results']
    
    print("="*80)
    print("🎨 NPU Direct Depth Visualization (Best 5 & Worst 5)")
    print("="*80)
    print(f"Total files: {len(npu_files)}")
    print()
    
    # Collect all metrics
    all_results = []
    
    print("📊 Computing metrics for all images...")
    for npu_file in npu_files:
        new_filename = npu_file.stem
        
        # Load NPU prediction
        npu_depth = np.load(npu_file)
        while npu_depth.ndim > 2:
            npu_depth = npu_depth.squeeze(0)
        
        # Load GT
        try:
            gt_depth, rgb = load_gt_depth(new_filename, test_json)
        except Exception as e:
            continue
        
        # Compute metrics
        metrics = compute_depth_metrics(gt_depth, npu_depth, min_depth, max_depth)
        if metrics is None:
            continue
        
        all_results.append({
            'filename': new_filename,
            'metrics': metrics,
            'gt_depth': gt_depth,
            'pred_depth': npu_depth,
            'rgb': rgb
        })
    
    print(f"✅ Processed {len(all_results)} images\n")
    
    # Sort by abs_rel
    all_results.sort(key=lambda x: x['metrics']['abs_rel'])
    
    # Get best 5 and worst 5
    best_5 = all_results[:5]
    worst_5 = all_results[-5:]
    
    # Visualize BEST 5
    print("="*80)
    print("🏆 BEST 5 (Lowest abs_rel)")
    print("="*80)
    
    for i, result in enumerate(best_5, 1):
        filename = result['filename']
        metrics = result['metrics']
        
        print(f"\n{i}. {filename}")
        print(f"   abs_rel: {metrics['abs_rel']:.4f}")
        print(f"   rmse:    {metrics['rmse']:.3f}m")
        print(f"   δ<1.25:  {metrics['a1']:.4f}")
        
        output_path = output_dir / f'best_{i:02d}_{filename}.png'
        visualize_depth_comparison(
            result['rgb'], result['gt_depth'], result['pred_depth'],
            metrics, filename, output_path, min_depth, max_depth
        )
    
    # Visualize WORST 5
    print("\n" + "="*80)
    print("⚠️  WORST 5 (Highest abs_rel)")
    print("="*80)
    
    for i, result in enumerate(worst_5, 1):
        filename = result['filename']
        metrics = result['metrics']
        
        print(f"\n{i}. {filename}")
        print(f"   abs_rel: {metrics['abs_rel']:.4f}")
        print(f"   rmse:    {metrics['rmse']:.3f}m")
        print(f"   δ<1.25:  {metrics['a1']:.4f}")
        
        output_path = output_dir / f'worst_{i:02d}_{filename}.png'
        visualize_depth_comparison(
            result['rgb'], result['gt_depth'], result['pred_depth'],
            metrics, filename, output_path, min_depth, max_depth
        )
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    best_avg_abs_rel = np.mean([r['metrics']['abs_rel'] for r in best_5])
    worst_avg_abs_rel = np.mean([r['metrics']['abs_rel'] for r in worst_5])
    overall_avg_abs_rel = np.mean([r['metrics']['abs_rel'] for r in all_results])
    
    print(f"\nBest 5 average abs_rel:    {best_avg_abs_rel:.4f}")
    print(f"Worst 5 average abs_rel:   {worst_avg_abs_rel:.4f}")
    print(f"Overall average abs_rel:   {overall_avg_abs_rel:.4f}")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
