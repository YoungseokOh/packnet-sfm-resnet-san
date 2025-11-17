#!/usr/bin/env python3
"""
Evaluate pre-computed depth predictions using the same metrics
as the official training/evaluation pipeline

This script loads pre-computed .npy depth files and GT depth,
then computes metrics using the exact same function as training.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from packnet_sfm.utils.depth import compute_depth_metrics
from packnet_sfm.utils.config import parse_test_file
from yacs.config import CfgNode


def load_gt_depth(sample, base_dir):
    """Load GT depth from newest_depth_maps"""
    dataset_root = sample['dataset_root']
    filename = sample['new_filename']
    depth_path = Path(dataset_root) / 'newest_depth_maps' / f'{filename}.png'
    
    if not depth_path.exists():
        return None
    
    depth_png = np.array(Image.open(depth_path), dtype=np.uint16)
    gt_depth = depth_png.astype(np.float32) / 256.0
    
    return gt_depth


def evaluate_precomputed_depths(depth_dir, checkpoint_path, config_path):
    """
    Evaluate pre-computed depth predictions
    
    Args:
        depth_dir: Directory with .npy depth files
        checkpoint_path: Path to checkpoint (for loading config)
        config_path: Path to config YAML
    """
    print("\n" + "="*80)
    print(f"📊 Evaluating Pre-computed Depths")
    print("="*80)
    print(f"Depth dir:   {depth_dir}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Config:      {config_path}")
    print("="*80 + "\n")
    
    # Load configuration
    config, _ = parse_test_file(checkpoint_path, config_path)
    
    # Get test split path
    test_split = config.datasets.test.split[0]
    
    # Make path absolute if needed
    if not test_split.startswith('/'):
        test_split = Path('/workspace/data/ncdb-cls-640x384') / test_split
    
    print(f"📂 Loading test split: {test_split}\n")
    
    # Load test samples
    with open(test_split) as f:
        test_samples = json.load(f)
    
    print(f"Found {len(test_samples)} test samples\n")
    
    # Prepare for evaluation
    depth_dir = Path(depth_dir)
    
    # Collect all predictions and GT for batch evaluation
    all_preds_no_scale = []
    all_gts_no_scale = []
    all_preds_with_scale = []
    all_gts_with_scale = []
    
    # Process each sample
    print("Loading predictions and GT...")
    for sample in tqdm(test_samples):
        filename = sample['new_filename']
        
        # Load prediction
        pred_path = depth_dir / f"{filename}.npy"
        if not pred_path.exists():
            continue
        
        pred_depth = np.load(pred_path)
        
        # Handle shape (squeeze if needed)
        if pred_depth.ndim == 3:
            pred_depth = pred_depth.squeeze(0)
        
        # Load GT depth
        gt_depth = load_gt_depth(sample, Path('/workspace/data/ncdb-cls-640x384'))
        if gt_depth is None:
            continue
        
        # Convert to tensors
        pred_tensor = torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0)
        
        # Collect for batch evaluation
        all_preds_no_scale.append(pred_tensor)
        all_gts_no_scale.append(gt_tensor)
        all_preds_with_scale.append(pred_tensor)
        all_gts_with_scale.append(gt_tensor)
    
    print(f"\n✅ Loaded {len(all_preds_no_scale)} samples\n")
    
    # Stack into batches
    pred_batch_no_scale = torch.cat(all_preds_no_scale, dim=0)
    gt_batch_no_scale = torch.cat(all_gts_no_scale, dim=0)
    pred_batch_with_scale = torch.cat(all_preds_with_scale, dim=0)
    gt_batch_with_scale = torch.cat(all_gts_with_scale, dim=0)
    
    print(f"Computing metrics on full batch...")
    print(f"  Batch shape: pred={pred_batch_no_scale.shape}, gt={gt_batch_no_scale.shape}\n")
    
    # Compute metrics on full batch (like official eval does)
    metrics_no_scale = compute_depth_metrics(config.model.params, gt_batch_no_scale, pred_batch_no_scale, use_gt_scale=False)
    metrics_with_scale = compute_depth_metrics(config.model.params, gt_batch_with_scale, pred_batch_with_scale, use_gt_scale=True)
    
    # Print results
    metric_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    
    print("="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print()
    print("WITHOUT GT MEDIAN SCALING:")
    print("-" * 80)
    for i, name in enumerate(metric_names):
        print(f"  {name:12s}: {metrics_no_scale[i].item():.6f}")
    
    print()
    print("WITH GT MEDIAN SCALING:")
    print("-" * 80)
    for i, name in enumerate(metric_names):
        print(f"  {name:12s}: {metrics_with_scale[i].item():.6f}")
    
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pre-computed depth predictions'
    )
    parser.add_argument('--depth_dir', type=str, required=True,
                       help='Directory with .npy depth files')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint (.ckpt) for loading config')
    parser.add_argument('--config', type=str, default=None,
                       help='Config (.yaml) file')
    
    args = parser.parse_args()
    
    evaluate_precomputed_depths(
        args.depth_dir,
        args.checkpoint,
        args.config
    )


if __name__ == '__main__':
    main()
