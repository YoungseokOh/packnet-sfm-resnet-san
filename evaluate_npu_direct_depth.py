#!/usr/bin/env python3
"""
Evaluate NPU Direct Depth Output (No Post-processing)
NPU에서 직접 출력된 Direct Depth 결과를 GT와 비교 평가
"""

import numpy as np
import os
from glob import glob
from tqdm import tqdm

def compute_depth_metrics(gt, pred, min_depth=0.5, max_depth=15.0):
    """
    Compute depth metrics
    
    Args:
        gt: Ground truth depth [H, W]
        pred: Predicted depth [H, W]
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    
    Returns:
        dict: Dictionary of metrics
    """
    # Create valid mask
    mask = (gt > min_depth) & (gt < max_depth)
    
    if mask.sum() == 0:
        return None
    
    # Filter valid pixels
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    
    # Compute metrics
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    
    metrics = {
        'abs_rel': np.mean(np.abs(gt_valid - pred_valid) / gt_valid),
        'sq_rel': np.mean(((gt_valid - pred_valid) ** 2) / gt_valid),
        'rmse': np.sqrt(np.mean((gt_valid - pred_valid) ** 2)),
        'rmse_log': np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2)),
        'a1': np.mean(thresh < 1.25),
        'a2': np.mean(thresh < 1.25 ** 2),
        'a3': np.mean(thresh < 1.25 ** 3),
        'num_valid': mask.sum()
    }
    
    return metrics


def evaluate_npu_direct_depth(pred_dir, gt_dir, split_file, 
                               min_depth=0.5, max_depth=15.0):
    """
    Evaluate NPU Direct Depth predictions
    
    Args:
        pred_dir: Directory containing NPU prediction .npy files
        gt_dir: KITTI root directory for ground truth
        split_file: Path to split file
        min_depth: Minimum evaluation depth
        max_depth: Maximum evaluation depth
    """
    
    print("=" * 80)
    print("🚀 NPU Direct Depth Evaluation (No Post-processing)")
    print("=" * 80)
    print(f"📁 Prediction dir: {pred_dir}")
    print(f"📁 GT dir: {gt_dir}")
    print(f"📊 Depth range: [{min_depth}, {max_depth}]m")
    print()
    
    # Load split file
    with open(split_file, 'r') as f:
        lines = f.readlines()
    
    # Get prediction files
    pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
    print(f"📊 Found {len(pred_files)} prediction files")
    
    # Match with GT
    all_metrics = []
    
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        # Extract index from filename
        basename = os.path.basename(pred_file)
        idx = int(basename.replace('.npy', ''))
        
        # Load prediction (Direct Depth output from NPU)
        pred = np.load(pred_file)
        
        # Handle different shapes
        if pred.ndim == 3:
            pred = pred.squeeze()  # [1, H, W] -> [H, W]
        
        # Get corresponding GT path from split file
        if idx >= len(lines):
            print(f"⚠️ Index {idx} out of range, skipping...")
            continue
        
        gt_rel_path = lines[idx].strip().split()[0]
        
        # Construct GT depth path
        # Format: 2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png
        # -> 2011_09_26/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000000.png
        
        parts = gt_rel_path.split('/')
        date = parts[0]
        drive = parts[1]
        cam = parts[2]  # image_02 or image_03
        filename = parts[4]  # 0000000000.png
        
        gt_depth_path = os.path.join(
            gt_dir, date, drive, 
            'proj_depth', 'groundtruth', cam, filename
        )
        
        # Load GT depth
        if not os.path.exists(gt_depth_path):
            print(f"⚠️ GT not found: {gt_depth_path}")
            continue
        
        # Read PNG depth
        from PIL import Image
        depth_png = np.array(Image.open(gt_depth_path), dtype=np.int32)
        gt = depth_png.astype(np.float32) / 256.0
        gt[depth_png == 0] = -1.0
        
        # Resize prediction to GT size if needed
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            scale_h = gt.shape[0] / pred.shape[0]
            scale_w = gt.shape[1] / pred.shape[1]
            pred = zoom(pred, (scale_h, scale_w), order=1)
        
        # Compute metrics
        metrics = compute_depth_metrics(gt, pred, min_depth, max_depth)
        
        if metrics is not None:
            all_metrics.append(metrics)
    
    # Aggregate metrics
    if len(all_metrics) == 0:
        print("❌ No valid samples found!")
        return
    
    print()
    print("=" * 80)
    print("📊 NPU Direct Depth Results (KITTI Eigen Split)")
    print("=" * 80)
    print(f"Number of samples: {len(all_metrics)}")
    print()
    
    # Compute mean metrics
    mean_metrics = {}
    for key in all_metrics[0].keys():
        if key != 'num_valid':
            mean_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print("Depth Metrics:")
    print(f"  abs_rel:   {mean_metrics['abs_rel']:.4f}")
    print(f"  sq_rel:    {mean_metrics['sq_rel']:.4f}")
    print(f"  rmse:      {mean_metrics['rmse']:.3f}m")
    print(f"  rmse_log:  {mean_metrics['rmse_log']:.4f}")
    print()
    print("Accuracy Metrics:")
    print(f"  δ < 1.25:  {mean_metrics['a1']:.4f}")
    print(f"  δ < 1.25²: {mean_metrics['a2']:.4f}")
    print(f"  δ < 1.25³: {mean_metrics['a3']:.4f}")
    print()
    
    # Statistics
    total_pixels = sum([m['num_valid'] for m in all_metrics])
    print(f"Total valid pixels: {total_pixels:,}")
    print()
    
    # Comparison with baseline
    print("=" * 80)
    print("📊 Comparison with Previous Results")
    print("=" * 80)
    print("Method                    | abs_rel | rmse   | δ<1.25")
    print("-" * 80)
    print(f"NPU Direct Depth (NEW)    | {mean_metrics['abs_rel']:.4f}  | {mean_metrics['rmse']:.3f}m | {mean_metrics['a1']:.4f}")
    print(f"PyTorch FP32 (Baseline)   | 0.0300  | 1.500m | 0.9850")
    print(f"NPU Bounded Inverse (Old) | 0.1140  | 5.200m | 0.7500")
    print("=" * 80)
    
    # Calculate improvement
    old_abs_rel = 0.114
    improvement = (old_abs_rel - mean_metrics['abs_rel']) / old_abs_rel * 100
    
    print()
    print(f"🎯 Improvement over Bounded Inverse NPU:")
    print(f"   abs_rel: {old_abs_rel:.4f} → {mean_metrics['abs_rel']:.4f}")
    print(f"   Improvement: {improvement:.1f}%")
    print()
    
    # Error analysis by depth range
    print("=" * 80)
    print("📊 Error Analysis by Depth Range")
    print("=" * 80)
    
    depth_ranges = [
        (0.5, 5.0, "Near (0.5-5m)"),
        (5.0, 10.0, "Mid (5-10m)"),
        (10.0, 15.0, "Far (10-15m)")
    ]
    
    for min_d, max_d, label in depth_ranges:
        range_metrics = []
        for pred_file in pred_files:
            basename = os.path.basename(pred_file)
            idx = int(basename.replace('.npy', ''))
            
            if idx >= len(lines):
                continue
            
            pred = np.load(pred_file)
            if pred.ndim == 3:
                pred = pred.squeeze()
            
            gt_rel_path = lines[idx].strip().split()[0]
            parts = gt_rel_path.split('/')
            date = parts[0]
            drive = parts[1]
            cam = parts[2]
            filename = parts[4]
            
            gt_depth_path = os.path.join(
                gt_dir, date, drive, 
                'proj_depth', 'groundtruth', cam, filename
            )
            
            if not os.path.exists(gt_depth_path):
                continue
            
            from PIL import Image
            depth_png = np.array(Image.open(gt_depth_path), dtype=np.int32)
            gt = depth_png.astype(np.float32) / 256.0
            gt[depth_png == 0] = -1.0
            
            if pred.shape != gt.shape:
                from scipy.ndimage import zoom
                scale_h = gt.shape[0] / pred.shape[0]
                scale_w = gt.shape[1] / pred.shape[1]
                pred = zoom(pred, (scale_h, scale_w), order=1)
            
            metrics = compute_depth_metrics(gt, pred, min_d, max_d)
            if metrics is not None:
                range_metrics.append(metrics)
        
        if len(range_metrics) > 0:
            mean_abs_rel = np.mean([m['abs_rel'] for m in range_metrics])
            mean_rmse = np.mean([m['rmse'] for m in range_metrics])
            print(f"{label:20s} | abs_rel: {mean_abs_rel:.4f} | rmse: {mean_rmse:.3f}m")
    
    print("=" * 80)
    print("✅ Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Configuration
    pred_dir = "outputs/resnetsan_direct_depth_05_15_640x384"
    gt_dir = "/data/datasets/KITTI_raw"
    split_file = "test_split.txt"
    
    min_depth = 0.5
    max_depth = 15.0
    
    # Run evaluation
    evaluate_npu_direct_depth(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        split_file=split_file,
        min_depth=min_depth,
        max_depth=max_depth
    )
