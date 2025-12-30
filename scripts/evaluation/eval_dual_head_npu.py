#!/usr/bin/env python3
"""
Dual-Head NPU Results Evaluation Script

This script evaluates pre-computed NPU inference results (integer_sigmoid.npy, fractional_sigmoid.npy)
using the EXACT SAME metric computation logic as packnet_sfm/utils/depth.py

CRITICAL: Uses the same depth composition formula as the model:
    depth = integer_sigmoid * max_depth + fractional_sigmoid

Unlike evaluate_dual_head.py, this script:
- Uses the EXACT same metric formulas as training validation
- Correctly uses max_depth from the trained model (e.g., 10.0m, not hardcoded 15.0)
- Supports different depth_types for GT loading

Usage:
    python scripts/evaluation/eval_dual_head_npu.py \\
        --npy_dir /path/to/npu_outputs \\
        --dataset_path /workspace/data/ncdb-cls-indoor/test_set \\
        --depth_type depth \\
        --max_depth 10.0 \\
        --min_depth 0.1

Example with default indoor settings:
    python scripts/evaluation/eval_dual_head_npu.py \\
        --npy_dir /workspace/data/ncdb-cls-indoor/test_set/npu_output \\
        --dataset_path /workspace/data/ncdb-cls-indoor/test_set \\
        --depth_type depth_synthetic \\
        --max_depth 10.0
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob


# DEPTH_TYPE_MAPPING - same as ncdb_dataset.py
DEPTH_TYPE_MAPPING = {
    'distance': 'newest_distance_maps',
    'depth': 'newest_original_depth_maps',
    'depth_synthetic': 'newest_depth_maps',
    'distance_original': 'newest_original_distance_maps',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate NPU Dual-Head Results')
    
    # Required
    parser.add_argument('--npy_dir', type=str, required=True,
                        help='Directory containing NPU output .npy files')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset (test_set or validation_set)')
    
    # Depth composition parameters
    parser.add_argument('--max_depth', type=float, default=10.0,
                        help='Max depth for dual-head composition (default: 10.0m for indoor)')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Min depth for valid region filtering (default: 0.1m)')
    
    # GT type
    parser.add_argument('--depth_type', type=str, default='depth',
                        choices=['depth', 'depth_synthetic', 'distance', 'distance_original'],
                        help='Depth type for GT loading')
    
    # GT scaling option
    parser.add_argument('--no_gt_scale', action='store_true',
                        help='Disable GT median scaling (evaluate_dual_head.py does NOT use scaling)')
    
    # Mask (optional)
    parser.add_argument('--mask_file', type=str, default=None,
                        help='Optional mask file to filter evaluation region')
    
    # Output
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save detailed results to JSON file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-sample metrics')
    
    return parser.parse_args()


def load_gt_depth(depth_path):
    """Load ground truth depth map from .npz, .npy, or .png file"""
    if depth_path.endswith('.npz'):
        data = np.load(depth_path)
        return data['depth'].astype(np.float32)
    elif depth_path.endswith('.npy'):
        return np.load(depth_path).astype(np.float32)
    elif depth_path.endswith('.png'):
        from PIL import Image
        depth_png = np.array(Image.open(depth_path), dtype=np.int32)
        # Standard 16-bit depth encoding: depth = pixel_value / 256.0 (in meters)
        return depth_png.astype(np.float32) / 256.0
    else:
        raise ValueError(f"Unsupported depth file format: {depth_path}")


def compute_dual_head_depth(integer_sigmoid, fractional_sigmoid, max_depth):
    """
    Compose depth from dual-head outputs.
    
    This is the EXACT formula used in the model:
        depth = integer_sigmoid * max_depth + fractional_sigmoid
    
    Parameters
    ----------
    integer_sigmoid : np.ndarray
        Integer head output (sigmoid, range [0, 1])
    fractional_sigmoid : np.ndarray
        Fractional head output (range [0, 1])
    max_depth : float
        Maximum depth in meters
    
    Returns
    -------
    depth : np.ndarray
        Composed depth map in meters
    """
    depth = integer_sigmoid * max_depth + fractional_sigmoid
    return depth


def compute_depth_metrics_numpy(gt, pred, min_depth, max_depth, use_gt_scale=True):
    """
    Compute depth metrics - EXACT same logic as packnet_sfm/utils/depth.py
    
    Parameters
    ----------
    gt : np.ndarray [H, W]
        Ground truth depth
    pred : np.ndarray [H, W]
        Predicted depth
    min_depth : float
        Minimum valid depth
    max_depth : float
        Maximum valid depth
    use_gt_scale : bool
        Whether to use GT median scaling
    
    Returns
    -------
    metrics : dict
        Dictionary with all metrics
    """
    # Valid mask - same logic as depth.py
    valid = (gt > min_depth) & (gt < max_depth)
    
    if valid.sum() == 0:
        print("WARNING: No valid pixels!")
        return None
    
    # Get valid values
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    
    # GT median scaling if requested (same as depth.py)
    if use_gt_scale:
        gt_median = np.median(gt_valid)
        pred_median = np.median(pred_valid)
        scale = gt_median / pred_median
        pred_valid = pred_valid * scale
    
    # Compute metrics - EXACT formulas from depth.py
    diff = gt_valid - pred_valid
    
    # Threshold ratio - same formula
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    
    # Accuracy metrics
    a1 = (thresh < 1.25).astype(np.float32).mean()
    a2 = (thresh < 1.25 ** 2).astype(np.float32).mean()
    a3 = (thresh < 1.25 ** 3).astype(np.float32).mean()
    
    # Error metrics
    abs_rel = np.mean(np.abs(diff) / gt_valid)
    sq_rel = np.mean(diff ** 2 / gt_valid)
    rmse = np.sqrt(np.mean(diff ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
    return {
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'a1': float(a1),
        'a2': float(a2),
        'a3': float(a3),
        'valid_pixels': int(valid.sum()),
        'total_pixels': int(valid.size),
    }


def main():
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("🔍 DUAL-HEAD NPU EVALUATION")
    print("=" * 80)
    print(f"NPY Dir: {args.npy_dir}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Depth Type: {args.depth_type}")
    print(f"Depth Range: [{args.min_depth}, {args.max_depth}]m")
    print(f"GT Median Scaling: {'OFF' if args.no_gt_scale else 'ON'}")
    print("=" * 80 + "\n")
    
    # Find NPY files - support multiple directory structures
    integer_files = []
    fractional_files = []
    
    # Structure 1: npy_dir/integer_sigmoid/*.npy and npy_dir/fractional_sigmoid/*.npy
    int_subdir = os.path.join(args.npy_dir, 'integer_sigmoid')
    frac_subdir = os.path.join(args.npy_dir, 'fractional_sigmoid')
    
    if os.path.isdir(int_subdir) and os.path.isdir(frac_subdir):
        integer_files = sorted(glob(os.path.join(int_subdir, '*.npy')))
        fractional_files = sorted(glob(os.path.join(frac_subdir, '*.npy')))
        print(f"📂 Found subdirectory structure: integer_sigmoid/, fractional_sigmoid/")
    
    # Structure 2: npy_dir/*_integer_sigmoid.npy
    if not integer_files:
        integer_files = sorted(glob(os.path.join(args.npy_dir, '*_integer_sigmoid.npy')))
        fractional_files = sorted(glob(os.path.join(args.npy_dir, '*_fractional_sigmoid.npy')))
    
    # Structure 3: npy_dir/integer_sigmoid_*.npy
    if not integer_files:
        integer_files = sorted(glob(os.path.join(args.npy_dir, 'integer_sigmoid_*.npy')))
        fractional_files = sorted(glob(os.path.join(args.npy_dir, 'fractional_sigmoid_*.npy')))
    
    if not integer_files:
        print(f"❌ No NPY files found in {args.npy_dir}")
        print(f"   Supported structures:")
        print(f"   1. {args.npy_dir}/integer_sigmoid/*.npy")
        print(f"   2. {args.npy_dir}/*_integer_sigmoid.npy")
        print(f"   3. {args.npy_dir}/integer_sigmoid_*.npy")
        return
    
    print(f"📂 Found {len(integer_files)} NPY file pairs")
    
    # GT folder
    gt_folder = DEPTH_TYPE_MAPPING[args.depth_type]
    gt_dir = os.path.join(args.dataset_path, gt_folder)
    
    if not os.path.exists(gt_dir):
        print(f"❌ GT folder not found: {gt_dir}")
        return
    
    gt_files = sorted(glob(os.path.join(gt_dir, '*.npz')) + 
                       glob(os.path.join(gt_dir, '*.png')) +
                       glob(os.path.join(gt_dir, '*.npy')))
    print(f"📂 Found {len(gt_files)} GT files in {gt_folder}")
    
    # Load mask if provided
    mask = None
    if args.mask_file:
        from PIL import Image
        mask_img = np.array(Image.open(args.mask_file).convert('L'))
        mask = mask_img > 127
        print(f"🎭 Loaded mask: {args.mask_file}")
        print(f"   Valid region: {mask.sum()} / {mask.size} pixels ({100*mask.mean():.1f}%)")
    
    # Evaluate each sample
    all_metrics = []
    sample_names = []
    
    print(f"\n⏳ Evaluating {min(len(integer_files), len(gt_files))} samples...")
    
    for i, (int_file, frac_file) in enumerate(tqdm(zip(integer_files, fractional_files), 
                                                     total=len(integer_files), 
                                                     disable=args.verbose)):
        # Extract sample name from filename
        basename = os.path.basename(int_file)
        # Handle different naming patterns
        sample_name = basename.replace('_integer_sigmoid.npy', '').replace('integer_sigmoid_', '').replace('.npy', '')
        
        # Find matching GT
        gt_file = None
        for gf in gt_files:
            gt_basename = os.path.basename(gf).replace('.npz', '')
            if sample_name == gt_basename or sample_name in gf:
                gt_file = gf
                break
        
        if gt_file is None:
            if args.verbose:
                print(f"⚠️  No GT found for: {sample_name}")
            continue
        
        # Load data
        integer_sigmoid = np.load(int_file).squeeze()
        fractional_sigmoid = np.load(frac_file).squeeze()
        gt = load_gt_depth(gt_file)
        
        # Handle size mismatch
        if integer_sigmoid.shape != gt.shape:
            # Resize prediction to GT size
            from PIL import Image
            pred_h, pred_w = integer_sigmoid.shape
            gt_h, gt_w = gt.shape
            
            integer_sigmoid = np.array(Image.fromarray(integer_sigmoid).resize((gt_w, gt_h), Image.BILINEAR))
            fractional_sigmoid = np.array(Image.fromarray(fractional_sigmoid).resize((gt_w, gt_h), Image.BILINEAR))
        
        # Compose depth
        pred_depth = compute_dual_head_depth(integer_sigmoid, fractional_sigmoid, args.max_depth)
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape != gt.shape:
                from PIL import Image
                mask_resized = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
            else:
                mask_resized = mask
            
            # Set invalid regions in GT to 0 (will be filtered by min_depth check)
            gt = gt * mask_resized
        
        # Compute metrics
        metrics = compute_depth_metrics_numpy(
            gt, pred_depth, 
            args.min_depth, args.max_depth,
            use_gt_scale=not args.no_gt_scale
        )
        
        if metrics is None:
            continue
        
        all_metrics.append(metrics)
        sample_names.append(sample_name)
        
        if args.verbose:
            print(f"\n[{i+1}] {sample_name}")
            print(f"    abs_rel: {metrics['abs_rel']:.4f} | a1: {metrics['a1']*100:.2f}%")
            print(f"    Pred range: [{pred_depth.min():.2f}, {pred_depth.max():.2f}]m")
            print(f"    GT range: [{gt[gt > 0].min():.2f}, {gt[gt > 0].max():.2f}]m")
    
    # Aggregate metrics
    if not all_metrics:
        print("❌ No valid samples evaluated!")
        return
    
    print(f"\n✅ Successfully evaluated {len(all_metrics)} samples")
    
    # Compute averages
    avg_metrics = {
        'abs_rel': np.mean([m['abs_rel'] for m in all_metrics]),
        'sq_rel': np.mean([m['sq_rel'] for m in all_metrics]),
        'rmse': np.mean([m['rmse'] for m in all_metrics]),
        'rmse_log': np.mean([m['rmse_log'] for m in all_metrics]),
        'a1': np.mean([m['a1'] for m in all_metrics]),
        'a2': np.mean([m['a2'] for m in all_metrics]),
        'a3': np.mean([m['a3'] for m in all_metrics]),
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Metric':<12} {'Value':<12}")
    print("-" * 24)
    print(f"{'abs_rel':<12} {avg_metrics['abs_rel']:.4f}")
    print(f"{'sq_rel':<12} {avg_metrics['sq_rel']:.4f}")
    print(f"{'rmse':<12} {avg_metrics['rmse']:.4f}")
    print(f"{'rmse_log':<12} {avg_metrics['rmse_log']:.4f}")
    print(f"{'a1':<12} {avg_metrics['a1']*100:.2f}%")
    print(f"{'a2':<12} {avg_metrics['a2']*100:.2f}%")
    print(f"{'a3':<12} {avg_metrics['a3']*100:.2f}%")
    print("=" * 80)
    
    # Standard deviation
    print("\n📈 Standard Deviations:")
    for key in ['abs_rel', 'a1']:
        values = [m[key] for m in all_metrics]
        std = np.std(values)
        print(f"   {key}: {std:.4f}")
    
    # Save to JSON if requested
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'config': {
                'npy_dir': args.npy_dir,
                'dataset_path': args.dataset_path,
                'depth_type': args.depth_type,
                'max_depth': args.max_depth,
                'min_depth': args.min_depth,
                'mask_file': args.mask_file,
            },
            'summary': avg_metrics,
            'num_samples': len(all_metrics),
            'per_sample': {name: m for name, m in zip(sample_names, all_metrics)}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
