#!/usr/bin/env python3
"""
Compute metrics on validation set to compare with epoch 29 training results
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def compute_metrics(pred_depth, gt_depth, depth_png, min_depth=0.5, max_depth=15.0, use_gt_scale=True):
    """Compute depth metrics"""
    # Create valid mask: 
    # 1. PNG value was not 0 (valid LiDAR return)
    # 2. GT depth in valid range [min_depth, max_depth]
    # 3. GT depth is finite
    valid_mask = (depth_png > 0) & (gt_depth > min_depth) & (gt_depth < max_depth) & np.isfinite(gt_depth)
    
    if valid_mask.sum() == 0:
        return None
    
    pred = pred_depth[valid_mask]
    gt = gt_depth[valid_mask]
    
    # Ground-truth median scaling if needed
    if use_gt_scale:
        gt_median = np.median(gt)
        pred_median = np.median(pred)
        scale = gt_median / pred_median
        pred = pred * scale
    
    # Compute metrics
    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean(((pred - gt) ** 2) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))
    
    # Threshold metrics
    thresh = np.maximum((gt / pred), (pred / gt))
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
        'a3': a3
    }

def main():
    # Paths
    pred_dir = Path('outputs/pytorch_fp32_direct_depth_inference_val')
    gt_base = Path('/workspace/data/ncdb-cls-640x384')
    val_json = '/workspace/data/ncdb-cls-640x384/splits/combined_val.json'
    
    # Load validation split
    with open(val_json) as f:
        val_samples = json.load(f)
    
    print(f"📊 Computing metrics on {len(val_samples)} validation samples\n")
    
    all_metrics = []
    
    for sample in tqdm(val_samples, desc="Computing metrics"):
        # Load prediction
        filename = sample['new_filename']
        pred_path = pred_dir / f"{filename}.npy"
        
        if not pred_path.exists():
            print(f"⚠️  Missing prediction: {filename}")
            continue
        
        pred_depth = np.load(pred_path)
        
        # Load GT depth from newest_depth_maps (PNG format)
        dataset_root = sample['dataset_root']
        depth_path = Path(dataset_root) / 'newest_depth_maps' / f"{filename}.png"
        
        if not depth_path.exists():
            print(f"⚠️  Missing GT depth: {filename}")
            continue
        
        from PIL import Image
        depth_png = np.array(Image.open(depth_path), dtype=np.uint16)
        gt_depth = depth_png.astype(np.float32) / 256.0
        
        # Compute metrics (pass depth_png for 0-value masking)
        metrics = compute_metrics(pred_depth, gt_depth, depth_png)
        
        if metrics is not None:
            all_metrics.append(metrics)
    
    print(f"\n✅ Computed metrics for {len(all_metrics)} samples\n")
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("="*80)
    print("📊 PyTorch FP32 Inference (Validation Set)")
    print("="*80)
    print(f"abs_rel:   {avg_metrics['abs_rel']:.6f}")
    print(f"sq_rel:    {avg_metrics['sq_rel']:.6f}")
    print(f"RMSE:      {avg_metrics['rmse']:.6f}m")
    print(f"RMSE_log:  {avg_metrics['rmse_log']:.6f}")
    print(f"δ < 1.25:  {avg_metrics['a1']:.6f}")
    print(f"δ < 1.25²: {avg_metrics['a2']:.6f}")
    print(f"δ < 1.25³: {avg_metrics['a3']:.6f}")
    print()
    
    # Compare with epoch 29
    epoch_29 = {
        'abs_rel': 0.04281532019376755,
        'sq_rel': 0.03516845032572746,
        'rmse': 0.39022570848464966,
        'rmse_log': 0.08337537199258804,
        'a1': 0.9753890037536621,
        'a2': 0.9925681948661804,
        'a3': 0.9971358776092529
    }
    
    print("="*80)
    print("📊 Epoch 29 (Training Validation Results)")
    print("="*80)
    print(f"abs_rel:   {epoch_29['abs_rel']:.6f}")
    print(f"sq_rel:    {epoch_29['sq_rel']:.6f}")
    print(f"RMSE:      {epoch_29['rmse']:.6f}m")
    print(f"RMSE_log:  {epoch_29['rmse_log']:.6f}")
    print(f"δ < 1.25:  {epoch_29['a1']:.6f}")
    print(f"δ < 1.25²: {epoch_29['a2']:.6f}")
    print(f"δ < 1.25³: {epoch_29['a3']:.6f}")
    print()
    
    print("="*80)
    print("📊 Difference (PyTorch Inference - Epoch 29)")
    print("="*80)
    print(f"abs_rel:   {avg_metrics['abs_rel'] - epoch_29['abs_rel']:+.6f} ({(avg_metrics['abs_rel']/epoch_29['abs_rel']-1)*100:+.2f}%)")
    print(f"sq_rel:    {avg_metrics['sq_rel'] - epoch_29['sq_rel']:+.6f} ({(avg_metrics['sq_rel']/epoch_29['sq_rel']-1)*100:+.2f}%)")
    print(f"RMSE:      {avg_metrics['rmse'] - epoch_29['rmse']:+.6f}m ({(avg_metrics['rmse']/epoch_29['rmse']-1)*100:+.2f}%)")
    print(f"RMSE_log:  {avg_metrics['rmse_log'] - epoch_29['rmse_log']:+.6f} ({(avg_metrics['rmse_log']/epoch_29['rmse_log']-1)*100:+.2f}%)")
    print(f"δ < 1.25:  {avg_metrics['a1'] - epoch_29['a1']:+.6f} ({(avg_metrics['a1']/epoch_29['a1']-1)*100:+.2f}%)")
    print(f"δ < 1.25²: {avg_metrics['a2'] - epoch_29['a2']:+.6f} ({(avg_metrics['a2']/epoch_29['a2']-1)*100:+.2f}%)")
    print(f"δ < 1.25³: {avg_metrics['a3'] - epoch_29['a3']:+.6f} ({(avg_metrics['a3']/epoch_29['a3']-1)*100:+.2f}%)")
    print("="*80)

if __name__ == '__main__':
    main()
