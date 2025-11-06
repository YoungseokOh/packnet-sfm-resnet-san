"""
PyTorch FP32 vs ONNX FP32 vs NPU INT8 Comparison Visualization

Layout:
Row 1: RGB, GT Depth, Metrics Summary
Row 2: PyTorch FP32, ONNX FP32, NPU INT8
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

def create_comparison_visualization(rgb, gt_depth, 
                                   pytorch_fp32_pred, onnx_fp32_pred, npu_int8_pred,
                                   pytorch_fp32_metrics, onnx_fp32_metrics, npu_int8_metrics,
                                   output_path, filename,
                                   min_depth=0.5, max_depth=15.0):
    """Create 3-way comparison visualization"""
    
    # Create figure with 2x3 layout
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.15)
    
    # Valid mask
    valid_mask = (gt_depth > 0) & (gt_depth >= min_depth) & (gt_depth <= max_depth)
    
    # Resize predictions to GT size if needed
    predictions = [pytorch_fp32_pred, onnx_fp32_pred, npu_int8_pred]
    resized_preds = []
    
    for pred in predictions:
        if pred.shape != gt_depth.shape:
            scale_h = gt_depth.shape[0] / pred.shape[0]
            scale_w = gt_depth.shape[1] / pred.shape[1]
            pred_resized = zoom(pred, (scale_h, scale_w), order=1)
            resized_preds.append(pred_resized)
        else:
            resized_preds.append(pred)
    
    pytorch_fp32_pred, onnx_fp32_pred, npu_int8_pred = resized_preds
    
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
    gt_display[~valid_mask] = min_depth
    
    im2 = ax2.imshow(gt_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    ax2.imshow(invalid_overlay)
    ax2.set_title('Ground Truth Depth', fontsize=16, fontweight='bold', pad=10)
    ax2.axis('off')
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Depth (m)', fontsize=12)
    
    # 3. Metrics Summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    ax3.text(0.5, 0.95, 'Metrics Summary', 
            ha='center', va='top', fontsize=18, fontweight='bold',
            transform=ax3.transAxes)
    
    # PyTorch FP32 (Green - Best)
    y_start = 0.78
    rect_pytorch = mpatches.FancyBboxPatch((0.05, y_start - 0.22), 0.9, 0.2,
                                           boxstyle="round,pad=0.01",
                                           facecolor='lightgreen', edgecolor='green',
                                           linewidth=2, transform=ax3.transAxes)
    ax3.add_patch(rect_pytorch)
    
    ax3.text(0.5, y_start, 'PyTorch FP32 (Checkpoint)', 
            ha='center', va='top', fontsize=13, fontweight='bold',
            transform=ax3.transAxes)
    
    metrics_text_pytorch = (
        f"abs_rel: {pytorch_fp32_metrics['abs_rel']:.4f}\n"
        f"RMSE: {pytorch_fp32_metrics['rmse']:.3f}m | "
        f"δ<1.25: {pytorch_fp32_metrics['a1']:.4f}"
    )
    ax3.text(0.5, y_start - 0.05, metrics_text_pytorch,
            ha='center', va='top', fontsize=11, family='monospace',
            transform=ax3.transAxes)
    
    # ONNX FP32 (Blue)
    y_start = 0.52
    rect_onnx = mpatches.FancyBboxPatch((0.05, y_start - 0.22), 0.9, 0.2,
                                        boxstyle="round,pad=0.01",
                                        facecolor='lightblue', edgecolor='blue',
                                        linewidth=2, transform=ax3.transAxes)
    ax3.add_patch(rect_onnx)
    
    ax3.text(0.5, y_start, 'ONNX FP32', 
            ha='center', va='top', fontsize=13, fontweight='bold',
            transform=ax3.transAxes)
    
    metrics_text_onnx = (
        f"abs_rel: {onnx_fp32_metrics['abs_rel']:.4f}\n"
        f"RMSE: {onnx_fp32_metrics['rmse']:.3f}m | "
        f"δ<1.25: {onnx_fp32_metrics['a1']:.4f}"
    )
    ax3.text(0.5, y_start - 0.05, metrics_text_onnx,
            ha='center', va='top', fontsize=11, family='monospace',
            transform=ax3.transAxes)
    
    # NPU INT8 (Orange)
    y_start = 0.26
    rect_npu = mpatches.FancyBboxPatch((0.05, y_start - 0.22), 0.9, 0.2,
                                       boxstyle="round,pad=0.01",
                                       facecolor='lightyellow', edgecolor='orange',
                                       linewidth=2, transform=ax3.transAxes)
    ax3.add_patch(rect_npu)
    
    ax3.text(0.5, y_start, 'NPU INT8', 
            ha='center', va='top', fontsize=13, fontweight='bold',
            transform=ax3.transAxes)
    
    metrics_text_npu = (
        f"abs_rel: {npu_int8_metrics['abs_rel']:.4f}\n"
        f"RMSE: {npu_int8_metrics['rmse']:.3f}m | "
        f"δ<1.25: {npu_int8_metrics['a1']:.4f}"
    )
    ax3.text(0.5, y_start - 0.05, metrics_text_npu,
            ha='center', va='top', fontsize=11, family='monospace',
            transform=ax3.transAxes)
    
    # === Row 2: Predictions ===
    
    # 4. PyTorch FP32 Prediction
    ax4 = fig.add_subplot(gs[1, 0])
    
    pytorch_display = pytorch_fp32_pred.copy()
    pytorch_display[~valid_mask] = min_depth
    
    im4 = ax4.imshow(pytorch_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    ax4.imshow(invalid_overlay)
    ax4.set_title('PyTorch FP32 Prediction', fontsize=16, fontweight='bold', pad=10)
    ax4.axis('off')
    
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Depth (m)', fontsize=12)
    
    # 5. ONNX FP32 Prediction
    ax5 = fig.add_subplot(gs[1, 1])
    
    onnx_display = onnx_fp32_pred.copy()
    onnx_display[~valid_mask] = min_depth
    
    im5 = ax5.imshow(onnx_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    ax5.imshow(invalid_overlay)
    ax5.set_title('ONNX FP32 Prediction', fontsize=16, fontweight='bold', pad=10)
    ax5.axis('off')
    
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('Depth (m)', fontsize=12)
    
    # 6. NPU INT8 Prediction
    ax6 = fig.add_subplot(gs[1, 2])
    
    npu_display = npu_int8_pred.copy()
    npu_display[~valid_mask] = min_depth
    
    im6 = ax6.imshow(npu_display, cmap='turbo_r', vmin=min_depth, vmax=max_depth)
    ax6.imshow(invalid_overlay)
    ax6.set_title('NPU INT8 Prediction', fontsize=16, fontweight='bold', pad=10)
    ax6.axis('off')
    
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Depth (m)', fontsize=12)
    
    # Main title
    fig.suptitle(f'PyTorch FP32 vs ONNX FP32 vs NPU INT8 - {filename}',
                fontsize=20, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 80)
    print("🎨 PyTorch FP32 vs ONNX FP32 vs NPU INT8 Comparison")
    print("=" * 80)
    
    # Paths
    pytorch_fp32_dir = Path('outputs/pytorch_fp32_direct_depth_inference')
    onnx_fp32_dir = Path('outputs/onnx_fp32_direct_depth_inference')
    npu_int8_dir = Path('outputs/resnetsan_direct_depth_05_15_640x384')
    test_json_path = Path('/workspace/data/ncdb-cls-640x384/splits/combined_test.json')
    vis_output_dir = Path('outputs/pytorch_vs_onnx_vs_npu_visualization')
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    min_depth = 0.5
    max_depth = 15.0
    
    # Check directories
    for name, path in [('PyTorch FP32', pytorch_fp32_dir), 
                       ('ONNX FP32', onnx_fp32_dir), 
                       ('NPU INT8', npu_int8_dir)]:
        if not path.exists():
            print(f"\n❌ Error: {name} results not found at {path}")
            return
    
    # Get all files
    pytorch_files = sorted(pytorch_fp32_dir.glob('*.npy'))
    print(f"\n📂 Found {len(pytorch_files)} PyTorch FP32 predictions")
    
    # Compute metrics for all images
    print("\n📊 Computing metrics...")
    all_metrics = []
    
    for pytorch_file in pytorch_files:
        filename = pytorch_file.stem
        onnx_file = onnx_fp32_dir / f'{filename}.npy'
        npu_file = npu_int8_dir / f'{filename}.npy'
        
        if not onnx_file.exists() or not npu_file.exists():
            continue
        
        # Load predictions
        pytorch_pred = np.load(pytorch_file)
        onnx_pred = np.load(onnx_file)
        npu_pred = np.load(npu_file)
        
        # Squeeze NPU if needed
        if len(npu_pred.shape) == 3:
            npu_pred = npu_pred[0]
        
        # Load GT
        gt_depth, rgb = load_gt_depth(filename, test_json_path)
        if gt_depth is None:
            continue
        
        # Resize if needed
        def resize_if_needed(pred, target_shape):
            if pred.shape != target_shape:
                scale_h = target_shape[0] / pred.shape[0]
                scale_w = target_shape[1] / pred.shape[1]
                return zoom(pred, (scale_h, scale_w), order=1)
            return pred
        
        pytorch_pred = resize_if_needed(pytorch_pred, gt_depth.shape)
        onnx_pred = resize_if_needed(onnx_pred, gt_depth.shape)
        npu_pred = resize_if_needed(npu_pred, gt_depth.shape)
        
        # Compute metrics
        pytorch_metrics = compute_depth_metrics(gt_depth, pytorch_pred, min_depth, max_depth)
        onnx_metrics = compute_depth_metrics(gt_depth, onnx_pred, min_depth, max_depth)
        npu_metrics = compute_depth_metrics(gt_depth, npu_pred, min_depth, max_depth)
        
        if pytorch_metrics is None or onnx_metrics is None or npu_metrics is None:
            continue
        
        all_metrics.append({
            'filename': filename,
            'pytorch_abs_rel': pytorch_metrics['abs_rel'],
            'onnx_abs_rel': onnx_metrics['abs_rel'],
            'npu_abs_rel': npu_metrics['abs_rel'],
            'pytorch_metrics': pytorch_metrics,
            'onnx_metrics': onnx_metrics,
            'npu_metrics': npu_metrics
        })
    
    print(f"✅ Computed metrics for {len(all_metrics)} images")
    
    # Sort by PyTorch abs_rel
    all_metrics_sorted = sorted(all_metrics, key=lambda x: x['pytorch_abs_rel'])
    
    # Select best 5 and worst 5
    best_5 = all_metrics_sorted[:5]
    worst_5 = all_metrics_sorted[-5:]
    
    print("\n🎯 Best 5 cases (by PyTorch FP32 abs_rel):")
    for i, m in enumerate(best_5, 1):
        print(f"   {i}. {m['filename']} - PyTorch: {m['pytorch_abs_rel']:.4f}, ONNX: {m['onnx_abs_rel']:.4f}, NPU: {m['npu_abs_rel']:.4f}")
    
    print("\n📉 Worst 5 cases (by PyTorch FP32 abs_rel):")
    for i, m in enumerate(worst_5, 1):
        print(f"   {i}. {m['filename']} - PyTorch: {m['pytorch_abs_rel']:.4f}, ONNX: {m['onnx_abs_rel']:.4f}, NPU: {m['npu_abs_rel']:.4f}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    selected_cases = best_5 + worst_5
    
    for i, metrics_dict in enumerate(selected_cases):
        filename = metrics_dict['filename']
        
        if i < 5:
            case_type = 'best'
            rank = i + 1
        else:
            case_type = 'worst'
            rank = i - 4
        
        print(f"   Creating {case_type}_{rank:02d}_{filename}...")
        
        # Load data
        pytorch_pred = np.load(pytorch_fp32_dir / f'{filename}.npy')
        onnx_pred = np.load(onnx_fp32_dir / f'{filename}.npy')
        npu_pred = np.load(npu_int8_dir / f'{filename}.npy')
        
        if len(npu_pred.shape) == 3:
            npu_pred = npu_pred[0]
        
        gt_depth, rgb = load_gt_depth(filename, test_json_path)
        
        # Create visualization
        output_file = vis_output_dir / f'{case_type}_{rank:02d}_{filename}.png'
        create_comparison_visualization(
            rgb, gt_depth, pytorch_pred, onnx_pred, npu_pred,
            metrics_dict['pytorch_metrics'], metrics_dict['onnx_metrics'], metrics_dict['npu_metrics'],
            output_file, filename,
            min_depth, max_depth
        )
    
    print(f"\n✅ Visualizations saved to: {vis_output_dir}")
    print(f"   Total files: {len(list(vis_output_dir.glob('*.png')))}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 Overall Statistics")
    print("=" * 80)
    
    avg_pytorch = np.mean([m['pytorch_abs_rel'] for m in all_metrics])
    avg_onnx = np.mean([m['onnx_abs_rel'] for m in all_metrics])
    avg_npu = np.mean([m['npu_abs_rel'] for m in all_metrics])
    
    avg_pytorch_rmse = np.mean([m['pytorch_metrics']['rmse'] for m in all_metrics])
    avg_onnx_rmse = np.mean([m['onnx_metrics']['rmse'] for m in all_metrics])
    avg_npu_rmse = np.mean([m['npu_metrics']['rmse'] for m in all_metrics])
    
    print(f"\nPyTorch FP32:")
    print(f"  avg abs_rel: {avg_pytorch:.4f}")
    print(f"  avg RMSE:    {avg_pytorch_rmse:.3f}m")
    
    print(f"\nONNX FP32:")
    print(f"  avg abs_rel: {avg_onnx:.4f}")
    print(f"  avg RMSE:    {avg_onnx_rmse:.3f}m")
    print(f"  vs PyTorch:  abs_rel {(avg_onnx - avg_pytorch) / avg_pytorch * 100:+.1f}%, RMSE {(avg_onnx_rmse - avg_pytorch_rmse) / avg_pytorch_rmse * 100:+.1f}%")
    
    print(f"\nNPU INT8:")
    print(f"  avg abs_rel: {avg_npu:.4f}")
    print(f"  avg RMSE:    {avg_npu_rmse:.3f}m")
    print(f"  vs PyTorch:  abs_rel {(avg_npu - avg_pytorch) / avg_pytorch * 100:+.1f}%, RMSE {(avg_npu_rmse - avg_pytorch_rmse) / avg_pytorch_rmse * 100:+.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()
