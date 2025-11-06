"""
ONNX FP32 vs NPU INT8 3-Way Comparison Visualization

Layout:
Row 1: RGB, GT Depth, Metrics Summary
Row 2: FP32 Pred, NPU INT8 Pred, Absolute Error (FP32 vs NPU INT8)

Note: Uses NPU INT8 results instead of ONNX INT8 due to ConvInteger support limitations
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import zoom

def load_gt_depth(new_filename, test_json_path):
    """Load GT depth from test set"""
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    for entry in test_data:
        if entry['new_filename'] == new_filename:
            dataset_root = Path(entry['dataset_root'])
            depth_path = dataset_root / 'newest_depth_maps' / f"{new_filename}.png"
            
            if not depth_path.exists():
                return None, None
            
            # Load depth (PNG format, divide by 256 for meters)
            depth = np.array(Image.open(depth_path), dtype=np.float32) / 256.0
            
            # Load RGB
            rgb_path = Path(entry['image_path'])
            rgb = np.array(Image.open(rgb_path).convert('RGB'))
            
            return depth, rgb
    
    return None, None

def compute_depth_metrics(gt, pred, min_depth=0.5, max_depth=15.0):
    """Compute depth metrics"""
    # Valid mask
    valid_mask = (gt > 0) & (gt >= min_depth) & (gt <= max_depth)
    
    if valid_mask.sum() == 0:
        return None
    
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    
    # Metrics
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
    # Accuracy thresholds
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'valid_pixels': valid_mask.sum()
    }

def create_comparison_visualization(rgb, gt_depth, fp32_pred, npu_int8_pred, 
                                   fp32_metrics, npu_int8_metrics,
                                   output_path, filename,
                                   min_depth=0.5, max_depth=15.0):
    """Create 3-way comparison visualization"""
    
    # Create figure with 2x3 layout
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.15)
    
    # Valid mask
    valid_mask = (gt_depth > 0) & (gt_depth >= min_depth) & (gt_depth <= max_depth)
    
    # Resize predictions to GT size if needed
    if fp32_pred.shape != gt_depth.shape:
        scale_h = gt_depth.shape[0] / fp32_pred.shape[0]
        scale_w = gt_depth.shape[1] / fp32_pred.shape[1]
        fp32_pred = zoom(fp32_pred, (scale_h, scale_w), order=1)
    
    if npu_int8_pred.shape != gt_depth.shape:
        scale_h = gt_depth.shape[0] / npu_int8_pred.shape[0]
        scale_w = gt_depth.shape[1] / npu_int8_pred.shape[1]
        npu_int8_pred = zoom(npu_int8_pred, (scale_h, scale_w), order=1)
    
    # === Row 1: RGB, GT, Metrics ===
    
    # 1. RGB Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    ax1.set_title('RGB Input', fontsize=16, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # 2. GT Depth
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create invalid overlay (black)
    invalid_overlay = np.zeros((*gt_depth.shape, 4))
    invalid_overlay[~valid_mask] = [0, 0, 0, 1]
    
    # GT display
    gt_display = gt_depth.copy()
    gt_display[~valid_mask] = min_depth  # For colormap
    
    im2 = ax2.imshow(gt_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    ax2.imshow(invalid_overlay)
    ax2.set_title('Ground Truth Depth', fontsize=16, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Depth (m)', fontsize=12)
    
    # 3. Metrics Summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # Title
    ax3.text(0.5, 0.95, 'Metrics Summary', 
            ha='center', va='top', fontsize=18, fontweight='bold',
            transform=ax3.transAxes)
    
    # FP32 Metrics (Blue background)
    y_start = 0.75
    rect_fp32 = mpatches.FancyBboxPatch((0.05, y_start - 0.28), 0.9, 0.26,
                                        boxstyle="round,pad=0.01",
                                        facecolor='lightblue', edgecolor='blue',
                                        linewidth=2, transform=ax3.transAxes)
    ax3.add_patch(rect_fp32)
    
    ax3.text(0.5, y_start, 'ONNX FP32', 
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax3.transAxes)
    
    metrics_text_fp32 = (
        f"abs_rel: {fp32_metrics['abs_rel']:.4f}\n"
        f"RMSE: {fp32_metrics['rmse']:.3f}m\n"
        f"δ<1.25: {fp32_metrics['a1']:.4f} ({fp32_metrics['a1']*100:.1f}%)"
    )
    ax3.text(0.5, y_start - 0.05, metrics_text_fp32,
            ha='center', va='top', fontsize=12, family='monospace',
            transform=ax3.transAxes)
    
    # INT8 Metrics (Orange background)
    y_start = 0.42
    rect_int8 = mpatches.FancyBboxPatch((0.05, y_start - 0.28), 0.9, 0.26,
                                        boxstyle="round,pad=0.01",
                                        facecolor='lightyellow', edgecolor='orange',
                                        linewidth=2, transform=ax3.transAxes)
    ax3.add_patch(rect_int8)
    
    ax3.text(0.5, y_start, 'ONNX INT8', 
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax3.transAxes)
    
    metrics_text_int8 = (
        f"abs_rel: {int8_metrics['abs_rel']:.4f}\n"
        f"RMSE: {int8_metrics['rmse']:.3f}m\n"
        f"δ<1.25: {int8_metrics['a1']:.4f} ({int8_metrics['a1']*100:.1f}%)"
    )
    ax3.text(0.5, y_start - 0.05, metrics_text_int8,
            ha='center', va='top', fontsize=12, family='monospace',
            transform=ax3.transAxes)
    
    # Degradation (Red background)
    y_start = 0.09
    abs_rel_deg = (int8_metrics['abs_rel'] - fp32_metrics['abs_rel']) / fp32_metrics['abs_rel'] * 100
    rmse_deg = (int8_metrics['rmse'] - fp32_metrics['rmse']) / fp32_metrics['rmse'] * 100
    a1_deg = (int8_metrics['a1'] - fp32_metrics['a1']) / fp32_metrics['a1'] * 100
    
    degradation_text = (
        f"Degradation:\n"
        f"abs_rel: {abs_rel_deg:+.1f}%\n"
        f"RMSE: {rmse_deg:+.1f}%\n"
        f"δ<1.25: {a1_deg:+.1f}%"
    )
    ax3.text(0.5, y_start, degradation_text,
            ha='center', va='top', fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose', 
                     edgecolor='red', linewidth=2),
            transform=ax3.transAxes)
    
    # === Row 2: FP32, INT8, Absolute Error ===
    
    # 4. FP32 Prediction
    ax4 = fig.add_subplot(gs[1, 0])
    
    fp32_display = fp32_pred.copy()
    fp32_display[~valid_mask] = min_depth
    
    im4 = ax4.imshow(fp32_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    ax4.imshow(invalid_overlay)
    ax4.set_title('ONNX FP32 Prediction', fontsize=16, fontweight='bold', pad=10)
    ax4.axis('off')
    
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Depth (m)', fontsize=12)
    
    # 5. INT8 Prediction
    ax5 = fig.add_subplot(gs[1, 1])
    
    int8_display = int8_pred.copy()
    int8_display[~valid_mask] = min_depth
    
    im5 = ax5.imshow(int8_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    ax5.imshow(invalid_overlay)
    ax5.set_title('ONNX INT8 Prediction', fontsize=16, fontweight='bold', pad=10)
    ax5.axis('off')
    
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('Depth (m)', fontsize=12)
    
    # 6. Absolute Error (FP32 vs INT8)
    ax6 = fig.add_subplot(gs[1, 2])
    
    abs_error = np.abs(fp32_pred - int8_pred)
    abs_error_display = abs_error.copy()
    abs_error_display[~valid_mask] = 0
    
    im6 = ax6.imshow(abs_error_display, cmap='hot', vmin=0, vmax=0.5)
    ax6.imshow(invalid_overlay)
    ax6.set_title('Absolute Error (FP32 - INT8)', fontsize=16, fontweight='bold', pad=10)
    ax6.axis('off')
    
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Error (m)', fontsize=12)
    
    # Add error statistics
    mean_error = abs_error[valid_mask].mean()
    max_error = abs_error[valid_mask].max()
    ax6.text(0.5, -0.08, f'Mean: {mean_error:.3f}m | Max: {max_error:.3f}m',
            ha='center', va='top', fontsize=11, fontweight='bold',
            transform=ax6.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
    
    # Main title
    fig.suptitle(f'ONNX FP32 vs INT8 Comparison - {filename}',
                fontsize=20, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 80)
    print("🎨 ONNX FP32 vs INT8 3-Way Comparison Visualization")
    print("=" * 80)
    
    # Paths
    fp32_output_dir = Path('outputs/onnx_fp32_direct_depth_inference')
    int8_output_dir = Path('outputs/onnx_int8_direct_depth_inference')
    test_json_path = Path('/workspace/data/ncdb-cls-640x384/splits/combined_test.json')
    vis_output_dir = Path('outputs/onnx_fp32_vs_int8_visualization')
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    min_depth = 0.5
    max_depth = 15.0
    
    # Check if inference results exist
    if not fp32_output_dir.exists():
        print(f"\n❌ Error: FP32 inference results not found at {fp32_output_dir}")
        print("   Please run: python quantize_and_infer_onnx_int8.py")
        return
    
    if not int8_output_dir.exists():
        print(f"\n❌ Error: INT8 inference results not found at {int8_output_dir}")
        print("   Please run: python quantize_and_infer_onnx_int8.py")
        return
    
    # Get all prediction files
    fp32_files = sorted(fp32_output_dir.glob('*.npy'))
    print(f"\n📂 Found {len(fp32_files)} FP32 predictions")
    
    # Compute metrics for all images
    print("\n📊 Computing metrics for all images...")
    all_metrics = []
    
    for fp32_file in fp32_files:
        filename = fp32_file.stem
        int8_file = int8_output_dir / f'{filename}.npy'
        
        if not int8_file.exists():
            continue
        
        # Load predictions
        fp32_pred = np.load(fp32_file)
        int8_pred = np.load(int8_file)
        
        # Load GT
        gt_depth, rgb = load_gt_depth(filename, test_json_path)
        if gt_depth is None:
            continue
        
        # Resize predictions to GT size if needed
        if fp32_pred.shape != gt_depth.shape:
            scale_h = gt_depth.shape[0] / fp32_pred.shape[0]
            scale_w = gt_depth.shape[1] / fp32_pred.shape[1]
            fp32_pred_resized = zoom(fp32_pred, (scale_h, scale_w), order=1)
            int8_pred_resized = zoom(int8_pred, (scale_h, scale_w), order=1)
        else:
            fp32_pred_resized = fp32_pred
            int8_pred_resized = int8_pred
        
        # Compute metrics
        fp32_metrics = compute_depth_metrics(gt_depth, fp32_pred_resized, min_depth, max_depth)
        int8_metrics = compute_depth_metrics(gt_depth, int8_pred_resized, min_depth, max_depth)
        
        if fp32_metrics is None or int8_metrics is None:
            continue
        
        all_metrics.append({
            'filename': filename,
            'fp32_abs_rel': fp32_metrics['abs_rel'],
            'int8_abs_rel': int8_metrics['abs_rel'],
            'fp32_rmse': fp32_metrics['rmse'],
            'int8_rmse': int8_metrics['rmse'],
            'fp32_metrics': fp32_metrics,
            'int8_metrics': int8_metrics
        })
    
    print(f"✅ Computed metrics for {len(all_metrics)} images")
    
    # Sort by FP32 abs_rel to find best/worst cases
    all_metrics_sorted = sorted(all_metrics, key=lambda x: x['fp32_abs_rel'])
    
    # Select best 5 and worst 5
    best_5 = all_metrics_sorted[:5]
    worst_5 = all_metrics_sorted[-5:]
    
    print("\n🎯 Best 5 cases (by FP32 abs_rel):")
    for i, m in enumerate(best_5, 1):
        print(f"   {i}. {m['filename']} - FP32 abs_rel: {m['fp32_abs_rel']:.4f}, INT8 abs_rel: {m['int8_abs_rel']:.4f}")
    
    print("\n📉 Worst 5 cases (by FP32 abs_rel):")
    for i, m in enumerate(worst_5, 1):
        print(f"   {i}. {m['filename']} - FP32 abs_rel: {m['fp32_abs_rel']:.4f}, INT8 abs_rel: {m['int8_abs_rel']:.4f}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    selected_cases = best_5 + worst_5
    
    for i, metrics_dict in enumerate(selected_cases):
        filename = metrics_dict['filename']
        
        # Determine if best or worst
        if i < 5:
            case_type = 'best'
            rank = i + 1
        else:
            case_type = 'worst'
            rank = i - 4
        
        print(f"   Creating {case_type}_{rank:02d}_{filename}...")
        
        # Load data
        fp32_pred = np.load(fp32_output_dir / f'{filename}.npy')
        int8_pred = np.load(int8_output_dir / f'{filename}.npy')
        gt_depth, rgb = load_gt_depth(filename, test_json_path)
        
        # Create visualization
        output_file = vis_output_dir / f'{case_type}_{rank:02d}_{filename}.png'
        create_comparison_visualization(
            rgb, gt_depth, fp32_pred, int8_pred,
            metrics_dict['fp32_metrics'], metrics_dict['int8_metrics'],
            output_file, filename,
            min_depth, max_depth
        )
    
    print(f"\n✅ Visualizations saved to: {vis_output_dir}")
    print(f"   Total files: {len(list(vis_output_dir.glob('*.png')))}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 Overall Statistics")
    print("=" * 80)
    
    avg_fp32_abs_rel = np.mean([m['fp32_abs_rel'] for m in all_metrics])
    avg_int8_abs_rel = np.mean([m['int8_abs_rel'] for m in all_metrics])
    avg_fp32_rmse = np.mean([m['fp32_rmse'] for m in all_metrics])
    avg_int8_rmse = np.mean([m['int8_rmse'] for m in all_metrics])
    
    print(f"\nFP32 ONNX:")
    print(f"  avg abs_rel: {avg_fp32_abs_rel:.4f}")
    print(f"  avg RMSE:    {avg_fp32_rmse:.3f}m")
    
    print(f"\nINT8 ONNX:")
    print(f"  avg abs_rel: {avg_int8_abs_rel:.4f}")
    print(f"  avg RMSE:    {avg_int8_rmse:.3f}m")
    
    print(f"\nDegradation:")
    print(f"  abs_rel: {(avg_int8_abs_rel - avg_fp32_abs_rel) / avg_fp32_abs_rel * 100:+.1f}%")
    print(f"  RMSE:    {(avg_int8_rmse - avg_fp32_rmse) / avg_fp32_rmse * 100:+.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()
