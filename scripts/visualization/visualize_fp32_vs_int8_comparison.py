#!/usr/bin/env python3
"""
PyTorch FP32 vs NPU INT8 비교 시각화

Layout:
Row 1: [RGB, GT, Metric Summary]
Row 2: [FP32 Pred, INT8 Pred, Absolute Error (FP32 vs INT8)]

Best 5 & Worst 5 샘플 생성 (abs_rel 기준)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_gt_depth_and_rgb(new_filename, test_json_path, rgb_dir='assets/test_set_rgb'):
    """GT depth와 RGB 이미지 로드"""
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
            
            # RGB 이미지 로드 - GT depth와 같은 세션에서 로드해야 함!
            # 우선순위:
            # 1. dataset_root/image_a6 (GT depth와 같은 세션)
            # 2. dataset_root/images (fallback)
            # 3. image_path (다른 세션일 수 있음 - 사용하지 않음)
            rgb = None
            
            # 1. Try dataset_root/image_a6 (same session as GT depth)
            rgb_path = Path(dataset_root) / 'image_a6' / f'{new_filename}.png'
            if rgb_path.exists():
                rgb = np.array(Image.open(rgb_path))
            
            # 2. Try dataset_root/images (fallback, same session)
            if rgb is None:
                rgb_path_alt = Path(dataset_root) / 'images' / f'{new_filename}.png'
                if rgb_path_alt.exists():
                    rgb = np.array(Image.open(rgb_path_alt))
            
            # 3. Last resort: assets directory (may be pre-copied, verify session)
            if rgb is None:
                rgb_path_assets = Path(rgb_dir) / f'{new_filename}.png'
                if rgb_path_assets.exists():
                    rgb = np.array(Image.open(rgb_path_assets))
                else:
                    rgb = None
            
            return depth, rgb
    
    raise ValueError(f"new_filename {new_filename} not found in {test_json_path}")


def compute_depth_metrics(gt, pred, min_depth=0.5, max_depth=15.0):
    """Depth metrics 계산"""
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
        'valid_mask': valid_mask,
    }


def visualize_comparison(rgb, gt_depth, fp32_pred, int8_pred, 
                        fp32_metrics, int8_metrics, filename, output_path,
                        min_depth=0.5, max_depth=15.0):
    """
    2x3 레이아웃으로 시각화
    Row 1: [RGB, GT, Metrics Summary]
    Row 2: [FP32 Pred, INT8 Pred, Absolute Error]
    """
    
    fig = plt.figure(figsize=(20, 10))
    
    valid_mask = fp32_metrics['valid_mask']
    
    # ========== Row 1: RGB, GT, Metrics ==========
    
    # 1.1 RGB Image
    ax1 = plt.subplot(2, 3, 1)
    if rgb is not None:
        ax1.imshow(rgb)
        ax1.set_title('RGB Input', fontsize=14, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'RGB Not Available', ha='center', va='center')
        ax1.set_title('RGB Input', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 1.2 Ground Truth Depth
    ax2 = plt.subplot(2, 3, 2)
    gt_display = np.zeros_like(gt_depth)
    gt_display[~valid_mask] = min_depth
    gt_display[valid_mask] = gt_depth[valid_mask]
    
    im2 = ax2.imshow(gt_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    
    # Black overlay for invalid pixels
    invalid_overlay = np.zeros((*gt_depth.shape, 4))
    invalid_overlay[~valid_mask] = [0, 0, 0, 1]
    ax2.imshow(invalid_overlay)
    
    ax2.set_title('Ground Truth Depth', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # 1.3 Metrics Summary
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    gt_valid = gt_depth[valid_mask]
    fp32_valid = fp32_pred[valid_mask]
    int8_valid = int8_pred[valid_mask]
    
    metrics_text = f"""
📊 Metrics Comparison
{'='*40}

File: {filename}

🎯 PyTorch FP32:
  • abs_rel:  {fp32_metrics['abs_rel']:.6f}
  • RMSE:     {fp32_metrics['rmse']:.4f}m
  • δ < 1.25: {fp32_metrics['a1']:.6f}

🔧 NPU INT8:
  • abs_rel:  {int8_metrics['abs_rel']:.6f}
  • RMSE:     {int8_metrics['rmse']:.4f}m
  • δ < 1.25: {int8_metrics['a1']:.6f}

📈 Degradation (INT8 vs FP32):
  • abs_rel:  {int8_metrics['abs_rel']/fp32_metrics['abs_rel']:.2f}x
  • RMSE:     {int8_metrics['rmse']/fp32_metrics['rmse']:.2f}x
  • δ < 1.25: {fp32_metrics['a1']/int8_metrics['a1']:.3f}x

📏 Valid Pixels: {valid_mask.sum():,}
   ({100*valid_mask.sum()/valid_mask.size:.1f}%)

🌡️ Depth Range (Valid):
  • GT:   [{gt_valid.min():.2f}, {gt_valid.max():.2f}]m
  • FP32: [{fp32_valid.min():.2f}, {fp32_valid.max():.2f}]m
  • INT8: [{int8_valid.min():.2f}, {int8_valid.max():.2f}]m
    """
    
    ax3.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # ========== Row 2: FP32, INT8, Error ==========
    
    # 2.1 PyTorch FP32 Prediction
    ax4 = plt.subplot(2, 3, 4)
    fp32_display = np.zeros_like(fp32_pred)
    fp32_display[~valid_mask] = min_depth
    fp32_display[valid_mask] = fp32_pred[valid_mask]
    
    im4 = ax4.imshow(fp32_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    ax4.imshow(invalid_overlay)
    
    ax4.set_title('PyTorch FP32 Prediction', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # 2.2 NPU INT8 Prediction
    ax5 = plt.subplot(2, 3, 5)
    int8_display = np.zeros_like(int8_pred)
    int8_display[~valid_mask] = min_depth
    int8_display[valid_mask] = int8_pred[valid_mask]
    
    im5 = ax5.imshow(int8_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    ax5.imshow(invalid_overlay)
    
    ax5.set_title('NPU INT8 Prediction', fontsize=14, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # 2.3 Absolute Error (FP32 vs INT8)
    ax6 = plt.subplot(2, 3, 6)
    error = np.abs(fp32_pred - int8_pred)
    error_display = np.zeros_like(error)
    error_display[~valid_mask] = 0
    error_display[valid_mask] = error[valid_mask]
    
    # Calculate error statistics on valid pixels
    error_valid = error[valid_mask]
    
    im6 = ax6.imshow(error_display, cmap='hot', vmin=0, vmax=1.5)
    ax6.imshow(invalid_overlay)
    
    ax6.set_title(f'Absolute Error (FP32 vs INT8)\nMean: {error_valid.mean():.3f}m, Max: {error_valid.max():.3f}m', 
                  fontsize=14, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='Error (m)')
    
    plt.suptitle(f'PyTorch FP32 vs NPU INT8 Comparison: {filename}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def main():
    # Configuration
    fp32_dir = Path('outputs/pytorch_fp32_official_pipeline')
    int8_dir = Path('outputs/resnetsan_direct_depth_05_15_640x384')
    test_json = '/workspace/data/ncdb-cls-640x384/splits/combined_test.json'
    output_dir = Path('outputs/fp32_vs_int8_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_depth = 0.5
    max_depth = 15.0
    
    # Get all FP32 files
    fp32_files = sorted(fp32_dir.glob('*.npy'))
    
    print("="*80)
    print("🎨 PyTorch FP32 vs NPU INT8 Visualization (Best 5 & Worst 5)")
    print("="*80)
    print(f"Total files: {len(fp32_files)}")
    print()
    
    # Collect all results
    all_results = []
    
    print("📊 Computing metrics for all images...")
    for fp32_file in fp32_files:
        new_filename = fp32_file.stem
        
        # Load predictions
        fp32_pred = np.load(fp32_file)
        int8_file = int8_dir / f"{new_filename}.npy"
        
        if not int8_file.exists():
            continue
        
        int8_pred = np.load(int8_file)
        
        # Handle shape differences
        while fp32_pred.ndim > 2:
            fp32_pred = fp32_pred.squeeze(0)
        while int8_pred.ndim > 2:
            int8_pred = int8_pred.squeeze(0)
        
        # Load GT and RGB
        try:
            gt_depth, rgb = load_gt_depth_and_rgb(new_filename, test_json)
        except Exception as e:
            print(f"  ⚠️ Skipping {new_filename}: {e}")
            continue
        
        # Compute metrics
        fp32_metrics = compute_depth_metrics(gt_depth, fp32_pred, min_depth, max_depth)
        int8_metrics = compute_depth_metrics(gt_depth, int8_pred, min_depth, max_depth)
        
        if fp32_metrics is None or int8_metrics is None:
            continue
        
        all_results.append({
            'filename': new_filename,
            'fp32_metrics': fp32_metrics,
            'int8_metrics': int8_metrics,
            'gt_depth': gt_depth,
            'fp32_pred': fp32_pred,
            'int8_pred': int8_pred,
            'rgb': rgb
        })
    
    print(f"✅ Processed {len(all_results)} images\n")
    
    # Sort by INT8 abs_rel (to see which samples are hardest for INT8)
    all_results.sort(key=lambda x: x['int8_metrics']['abs_rel'])
    
    # Get best 5 and worst 5
    best_5 = all_results[:5]
    worst_5 = all_results[-5:]
    
    # Visualize BEST 5
    print("="*80)
    print("🏆 BEST 5 (Lowest INT8 abs_rel)")
    print("="*80)
    
    for i, result in enumerate(best_5, 1):
        filename = result['filename']
        fp32_m = result['fp32_metrics']
        int8_m = result['int8_metrics']
        
        print(f"\n{i}. {filename}")
        print(f"   FP32 abs_rel: {fp32_m['abs_rel']:.6f}")
        print(f"   INT8 abs_rel: {int8_m['abs_rel']:.6f}")
        print(f"   Degradation:  {int8_m['abs_rel']/fp32_m['abs_rel']:.2f}x")
        
        output_path = output_dir / f'best_{i:02d}_{filename}.png'
        visualize_comparison(
            result['rgb'], result['gt_depth'], 
            result['fp32_pred'], result['int8_pred'],
            fp32_m, int8_m, filename, output_path, min_depth, max_depth
        )
    
    # Visualize WORST 5
    print("\n" + "="*80)
    print("⚠️  WORST 5 (Highest INT8 abs_rel)")
    print("="*80)
    
    for i, result in enumerate(worst_5, 1):
        filename = result['filename']
        fp32_m = result['fp32_metrics']
        int8_m = result['int8_metrics']
        
        print(f"\n{i}. {filename}")
        print(f"   FP32 abs_rel: {fp32_m['abs_rel']:.6f}")
        print(f"   INT8 abs_rel: {int8_m['abs_rel']:.6f}")
        print(f"   Degradation:  {int8_m['abs_rel']/fp32_m['abs_rel']:.2f}x")
        
        output_path = output_dir / f'worst_{i:02d}_{filename}.png'
        visualize_comparison(
            result['rgb'], result['gt_depth'], 
            result['fp32_pred'], result['int8_pred'],
            fp32_m, int8_m, filename, output_path, min_depth, max_depth
        )
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    best_avg_int8 = np.mean([r['int8_metrics']['abs_rel'] for r in best_5])
    worst_avg_int8 = np.mean([r['int8_metrics']['abs_rel'] for r in worst_5])
    overall_avg_int8 = np.mean([r['int8_metrics']['abs_rel'] for r in all_results])
    
    best_avg_fp32 = np.mean([r['fp32_metrics']['abs_rel'] for r in best_5])
    worst_avg_fp32 = np.mean([r['fp32_metrics']['abs_rel'] for r in worst_5])
    overall_avg_fp32 = np.mean([r['fp32_metrics']['abs_rel'] for r in all_results])
    
    print(f"\nBest 5 average:")
    print(f"  FP32:  {best_avg_fp32:.6f}")
    print(f"  INT8:  {best_avg_int8:.6f}")
    print(f"  Ratio: {best_avg_int8/best_avg_fp32:.2f}x")
    
    print(f"\nWorst 5 average:")
    print(f"  FP32:  {worst_avg_fp32:.6f}")
    print(f"  INT8:  {worst_avg_int8:.6f}")
    print(f"  Ratio: {worst_avg_int8/worst_avg_fp32:.2f}x")
    
    print(f"\nOverall average:")
    print(f"  FP32:  {overall_avg_fp32:.6f}")
    print(f"  INT8:  {overall_avg_int8:.6f}")
    print(f"  Ratio: {overall_avg_int8/overall_avg_fp32:.2f}x")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
