#!/usr/bin/env python3
"""
Simple evaluation script for Dual-Head depth predictions
Supports both PyTorch checkpoint and pre-computed depth files (FP32/INT8 NPU outputs)

Usage:
    # Evaluate PyTorch checkpoint
    python scripts/evaluate_dual_head_simple.py \
        --checkpoint checkpoints/.../epoch=28.ckpt \
        --dataset_root /path/to/kitti \
        --test_file data_splits/kitti_eigen_test.txt
    
    # Evaluate pre-computed depths (FP32 ONNX or INT8 NPU)
    python scripts/evaluate_dual_head_simple.py \
        --precomputed_dir outputs/fp32_inference \
        --dataset_root /path/to/kitti \
        --test_file data_splits/kitti_eigen_test.txt
    
    # Compare FP32 vs INT8
    python scripts/evaluate_dual_head_simple.py \
        --precomputed_dir outputs/fp32_inference \
        --compare_dir outputs/int8_inference \
        --dataset_root /path/to/kitti \
        --test_file data_splits/kitti_eigen_test.txt
    # If Integer/Fractional outputs are stored in separate folders (per-model):
    python scripts/evaluate_dual_head_simple.py \
        --precomputed_dir outputs/dual_head_test_model \
        --precomputed_separate_dirs --precomputed_model_name resnetsan01_fp32 --precomputed_precision fp32 \
        --dataset_root /path/to/ncdb --test_file /path/to/splits/combined_test.json
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json


def compute_depth_metrics(gt, pred, min_depth=1e-3, max_depth=80.0):
    """
    Compute depth metrics (same as training)
    
    Args:
        gt: Ground truth depth [H, W]
        pred: Predicted depth [H, W]
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    
    Returns:
        dict: Metrics dictionary
    """
    # Mask out invalid depths (inclusive bounds to match official eval)
    mask = (gt >= min_depth) & (gt <= max_depth)
    
    if mask.sum() == 0:
        return None
    
    gt = gt[mask]
    pred = pred[mask]
    
    # Threshold
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    # Error metrics
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }


def load_gt_depth(dataset_root, filename, dataset_root_override=None):
    """
    Load ground truth depth from KITTI or NCDB dataset
    
    Args:
        dataset_root: Path to dataset root
        filename: Filename without extension (e.g., '0000000147')
        dataset_root_override: Override dataset_root for NCDB multi-sequence structure
    
    Returns:
        np.ndarray: Ground truth depth [H, W] or None if not found
    """
    # Use override if provided (for NCDB multi-sequence datasets)
    base_path = Path(dataset_root_override) if dataset_root_override else Path(dataset_root)
    
    # Try newest_depth_maps first (NCDB and KITTI)
    depth_path = base_path / 'newest_depth_maps' / f'{filename}.png'
    
    if not depth_path.exists():
        # Try KITTI depth_selection structure
        depth_path = Path(dataset_root) / 'depth_selection' / 'val_selection_cropped' / 'groundtruth_depth' / f'{filename}.png'
    
    if not depth_path.exists():
        return None
    
    # Load depth from PNG (stored as uint16, scale by 256)
    depth_png = np.array(Image.open(depth_path), dtype=np.uint16)
    gt_depth = depth_png.astype(np.float32) / 256.0
    
    return gt_depth


def load_precomputed_depth(depth_dir, filename):
    """
    Load pre-computed depth from .npy file
    
    Args:
        depth_dir: Directory containing .npy depth files
        filename: Filename without extension
    
    Returns:
        np.ndarray: Depth [H, W] or None if not found
    """
    npy_path = Path(depth_dir) / f'{filename}.npy'
    
    if not npy_path.exists():
        return None
    
    # Load depth from .npy file
    depth = np.load(npy_path)
    
    # Handle different shapes
    if depth.ndim == 4:  # [1, 1, H, W]
        depth = depth[0, 0]
    elif depth.ndim == 3:  # [1, H, W]
        depth = depth[0]
    
    return depth


def load_precomputed_dual_head(depth_dir, filename, max_depth=15.0, separate_dirs=False, model_name=None, precision='fp32'):
    """
    Load pre-computed dual-head outputs and compose depth
    
    Args:
        depth_dir: Directory containing .npz files with integer/fractional
        filename: Filename without extension
        max_depth: Maximum depth for composition
    
    Returns:
        tuple: (depth, integer_sigmoid, fractional_sigmoid) or (None, None, None)
    """
    # If separate directory layout is requested, check those paths first
    base_dir = Path(depth_dir)
    if separate_dirs and model_name is not None:
        integer_path = base_dir / model_name / f'integer_{precision}' / f'{filename}.npy'
        fractional_path = base_dir / model_name / f'fractional_{precision}' / f'{filename}.npy'
        depth_path_dir = base_dir / model_name / f'depth_{precision}' / f'{filename}.npy'
        if integer_path.exists() and fractional_path.exists():
            integer_sig = np.load(integer_path)
            fractional_sig = np.load(fractional_path)
            # Auto-detect scaling (if arrays are in meters, divide by max_depth)
            try:
                if np.nanmax(integer_sig) > 1.1:
                    integer_sig = integer_sig / max_depth
                    print(f"⚠️  Auto-correction: integer_sig loaded from separate dirs appears to be in meters; divided by {max_depth} to normalize to [0,1].")
                if np.nanmax(fractional_sig) > 1.1:
                    fractional_sig = fractional_sig / max_depth
                    print(f"⚠️  Auto-correction: fractional_sig loaded from separate dirs appears to be scaled by max_depth; divided by {max_depth} to convert to meters [0,1].")
            except Exception:
                pass
            # Compose depth
            depth = integer_sig * max_depth + fractional_sig
            # If depth also exists separately, load it as well
            if depth_path_dir.exists():
                depth_file = np.load(depth_path_dir)
                depth = depth_file
            return depth, integer_sig, fractional_sig

    npz_path = base_dir / f'{filename}.npz'

    if not npz_path.exists():
        # Try loading composed depth directly
        depth = load_precomputed_depth(depth_dir, filename)
        if depth is not None:
            return depth, None, None
        return None, None, None
    
    # Load dual-head outputs
    data = np.load(npz_path)
    
    # Check what's available
    if 'depth' in data or 'depth_composed' in data:
        # Already composed depth
        depth = data.get('depth', data.get('depth_composed'))
        if depth.ndim == 4:
            depth = depth[0, 0]
        elif depth.ndim == 3:
            depth = depth[0]
        integer_sig = data.get('integer_sigmoid', data.get('integer'))
        fractional_sig = data.get('fractional_sigmoid', data.get('fractional'))
        return depth, integer_sig, fractional_sig
    
    elif 'integer_sigmoid' in data and 'fractional_sigmoid' in data:
        # Separate outputs - compose depth
        integer_sig = data['integer_sigmoid']
        fractional_sig = data['fractional_sigmoid']
        
        if integer_sig.ndim == 4:
            integer_sig = integer_sig[0, 0]
            fractional_sig = fractional_sig[0, 0]
        elif integer_sig.ndim == 3:
            integer_sig = integer_sig[0]
            fractional_sig = fractional_sig[0]
        # Auto-detect if integer/fractional were stored in meters (scaled by max_depth)
        try:
            if np.nanmax(integer_sig) > 1.1:
                # integer appears to be in meters -> normalize
                integer_sig = integer_sig / max_depth
                print(f"⚠️  Auto-correction: integer_sig appears to be in meters; divided by {max_depth} to normalize to [0,1].")
            if np.nanmax(fractional_sig) > 1.1:
                # fractional was stored incorrectly scaled by max_depth -> normalize
                fractional_sig = fractional_sig / max_depth
                print(f"⚠️  Auto-correction: fractional_sig appears to be scaled by max_depth; divided by {max_depth} to convert to meters [0,1].")
        except Exception:
            # Skip auto correction if shape/dtype issues
            pass
        
    # Compose depth
    depth = integer_sig * max_depth + fractional_sig
    return depth, integer_sig, fractional_sig
    
    return None, None, None


def read_test_file(test_file_path):
    """
    Read test file (.txt or .json format)
    
    Args:
        test_file_path: Path to test file
    
    Returns:
        list: List of dictionaries with 'filename' and optional 'dataset_root_override'
    """
    test_file_path = Path(test_file_path)
    
    # Handle JSON format (NCDB style)
    if test_file_path.suffix == '.json':
        import json
        with open(test_file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON test file must be a list")
        
        results = []
        for entry in data:
            # NCDB format: has 'new_filename' and 'dataset_root'
            if 'new_filename' in entry:
                results.append({
                    'filename': entry['new_filename'],
                    'dataset_root_override': entry.get('dataset_root')
                })
            # Fallback: try image_path
            elif 'image_path' in entry:
                img_path = Path(entry['image_path'])
                filename = img_path.stem
                # Extract dataset_root from image_path
                # Example: /workspace/data/ncdb/.../synced_data/image_a6/0000000567.png
                if 'synced_data' in img_path.parts:
                    idx = img_path.parts.index('synced_data')
                    dataset_root_override = str(Path(*img_path.parts[:idx+1]))
                else:
                    dataset_root_override = None
                results.append({
                    'filename': filename,
                    'dataset_root_override': dataset_root_override
                })
        
        return results
    
    # Handle TXT format (KITTI style)
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
    
    results = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Format 1: "0000000147.png"
        # Format 2: "2011_09_26/2011_09_26_drive_0001_sync 0000000147"
        if '.png' in line:
            filename = line.split('.png')[0].split('/')[-1]
        else:
            parts = line.split()
            if len(parts) >= 2:
                filename = parts[-1]
            else:
                filename = line
        
        results.append({
            'filename': filename,
            'dataset_root_override': None
        })
    
    return results


def evaluate(args):
    """Main evaluation function"""
    
    print("\n" + "="*80)
    print("📊 Dual-Head Depth Evaluation")
    print("="*80)
    
    # Load test files
    print(f"\n📂 Loading test file: {args.test_file}")
    test_entries = read_test_file(args.test_file)
    print(f"   Found {len(test_entries)} test samples")
    
    # Evaluation setup
    print(f"\n⚙️  Evaluation setup:")
    print(f"   Dataset root:    {args.dataset_root}")
    print(f"   Min depth:       {args.min_depth}m")
    print(f"   Max depth:       {args.max_depth}m")
    print(f"   Eval max depth:  {args.eval_max_depth}m")
    print(f"   Dual-head max:   {args.dual_head_max_depth}m")
    
    if args.precomputed_dir:
        print(f"   Precomputed dir: {args.precomputed_dir}")
        # Auto-detect model name if separate_dirs is used and model_name is not given
        if getattr(args, 'precomputed_separate_dirs', False) and not getattr(args, 'precomputed_model_name', None):
            base_dir = Path(args.precomputed_dir)
            detected_model_name = None
            for child in base_dir.iterdir():
                if not child.is_dir():
                    continue
                if (child / f'integer_{getattr(args, "precomputed_precision", "fp32")}').exists() and (child / f'fractional_{getattr(args, "precomputed_precision", "fp32")}').exists():
                    detected_model_name = child.name
                    break
            if detected_model_name:
                print(f"   Detected precomputed model name: {detected_model_name}")
                args.precomputed_model_name = detected_model_name
    if args.compare_dir:
        print(f"   Compare dir:     {args.compare_dir}")
        if getattr(args, 'compare_separate_dirs', False) and not getattr(args, 'compare_model_name', None):
            base_dir = Path(args.compare_dir)
            detected_model_name = None
            for child in base_dir.iterdir():
                if not child.is_dir():
                    continue
                if (child / f'integer_{getattr(args, "compare_precision", "fp32")}').exists() and (child / f'fractional_{getattr(args, "compare_precision", "fp32")}').exists():
                    detected_model_name = child.name
                    break
            if detected_model_name:
                print(f"   Detected compare model name: {detected_model_name}")
                args.compare_model_name = detected_model_name
    
    # Collect metrics
    all_metrics = []
    compare_metrics = []
    
    print(f"\n🔍 Evaluating {len(test_entries)} samples...")
    
    for entry in tqdm(test_entries):
        filename = entry['filename']
        dataset_root_override = entry['dataset_root_override']
        
        # Load GT depth (with dataset_root_override for NCDB)
        gt_depth = load_gt_depth(args.dataset_root, filename, dataset_root_override)
        if gt_depth is None:
            #print(f"⚠️  GT depth not found: {filename}")
            continue
        
        # Load predicted depth
        if args.precomputed_dir:
            pred_depth, _, _ = load_precomputed_dual_head(
                args.precomputed_dir,
                filename,
                args.dual_head_max_depth,
                args.precomputed_separate_dirs,
                args.precomputed_model_name,
                args.precomputed_precision,
            )
            if pred_depth is None:
                #print(f"⚠️  Predicted depth not found: {filename}")
                continue
        else:
            raise NotImplementedError("PyTorch checkpoint evaluation not yet implemented")
        
        # Compute metrics
        metrics = compute_depth_metrics(
            gt_depth, pred_depth, 
            args.min_depth, args.eval_max_depth
        )
        
        if metrics is not None:
            all_metrics.append(metrics)
        
        # Compare with second model if provided
        if args.compare_dir:
            compare_depth, _, _ = load_precomputed_dual_head(
                args.compare_dir,
                filename,
                args.dual_head_max_depth,
                args.compare_separate_dirs,
                args.compare_model_name,
                args.compare_precision,
            )
            if compare_depth is not None:
                compare_met = compute_depth_metrics(
                    gt_depth, compare_depth,
                    args.min_depth, args.eval_max_depth
                )
                if compare_met is not None:
                    compare_metrics.append(compare_met)
    
    # Aggregate metrics
    if len(all_metrics) == 0:
        print("\n❌ No valid samples found!")
        return
    
    print(f"\n✅ Evaluated {len(all_metrics)} valid samples")
    
    # Compute mean metrics
    mean_metrics = {}
    for key in all_metrics[0].keys():
        mean_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Print results
    print("\n" + "="*80)
    print("📊 Evaluation Results")
    print("="*80)
    print(f"\nModel: {args.precomputed_dir if args.precomputed_dir else 'PyTorch'}")
    print(f"Samples: {len(all_metrics)}")
    print(f"\nMetrics:")
    print(f"  abs_rel:  {mean_metrics['abs_rel']:.4f}")
    print(f"  sq_rel:   {mean_metrics['sq_rel']:.3f}")
    print(f"  rmse:     {mean_metrics['rmse']:.3f}")
    print(f"  rmse_log: {mean_metrics['rmse_log']:.4f}")
    print(f"  a1:       {mean_metrics['a1']*100:.2f}%")
    print(f"  a2:       {mean_metrics['a2']*100:.2f}%")
    print(f"  a3:       {mean_metrics['a3']*100:.2f}%")
    
    # Compare results if available
    if len(compare_metrics) > 0:
        compare_mean = {}
        for key in compare_metrics[0].keys():
            compare_mean[key] = np.mean([m[key] for m in compare_metrics])
        
        print("\n" + "-"*80)
        print(f"\nComparison Model: {args.compare_dir}")
        print(f"Samples: {len(compare_metrics)}")
        print(f"\nMetrics:")
        print(f"  abs_rel:  {compare_mean['abs_rel']:.4f} (Δ {compare_mean['abs_rel']-mean_metrics['abs_rel']:+.4f})")
        print(f"  sq_rel:   {compare_mean['sq_rel']:.3f} (Δ {compare_mean['sq_rel']-mean_metrics['sq_rel']:+.3f})")
        print(f"  rmse:     {compare_mean['rmse']:.3f} (Δ {compare_mean['rmse']-mean_metrics['rmse']:+.3f})")
        print(f"  rmse_log: {compare_mean['rmse_log']:.4f} (Δ {compare_mean['rmse_log']-mean_metrics['rmse_log']:+.4f})")
        print(f"  a1:       {compare_mean['a1']*100:.2f}% (Δ {(compare_mean['a1']-mean_metrics['a1'])*100:+.2f}%)")
        print(f"  a2:       {compare_mean['a2']*100:.2f}% (Δ {(compare_mean['a2']-mean_metrics['a2'])*100:+.2f}%)")
        print(f"  a3:       {compare_mean['a3']*100:.2f}% (Δ {(compare_mean['a3']-mean_metrics['a3'])*100:+.2f}%)")
    
    # Save results if requested
    if args.output_json:
        results = {
            'model': str(args.precomputed_dir if args.precomputed_dir else 'PyTorch'),
            'num_samples': len(all_metrics),
            'metrics': {k: float(v) for k, v in mean_metrics.items()}
        }
        
        if len(compare_metrics) > 0:
            results['comparison'] = {
                'model': str(args.compare_dir),
                'num_samples': len(compare_metrics),
                'metrics': {k: float(v) for k, v in compare_mean.items()}
            }
        
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {args.output_json}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Dual-Head depth predictions')
    
    # Input options
    parser.add_argument('--checkpoint', type=str, help='Path to PyTorch checkpoint')
    parser.add_argument('--precomputed_dir', type=str, 
                       help='Directory with pre-computed depth files (.npy or .npz)')
    parser.add_argument('--compare_dir', type=str,
                       help='Directory with comparison depth files (e.g., INT8 vs FP32)')
    # Support for separate integer/fractional directories per model
    parser.add_argument('--precomputed_separate_dirs', action='store_true', help='Precomputed dir uses model_name/integer_precision, fractional_precision subdirs')
    parser.add_argument('--precomputed_model_name', type=str, default=None, help='Model name used in precomputed separate dirs')
    parser.add_argument('--precomputed_precision', type=str, default='fp32', choices=['fp32','int8'], help='Precision for precomputed separate dirs')
    parser.add_argument('--compare_separate_dirs', action='store_true', help='Compare dir uses model_name/integer_precision, fractional_precision subdirs')
    parser.add_argument('--compare_model_name', type=str, default=None, help='Model name used in compare separate dirs')
    parser.add_argument('--compare_precision', type=str, default='fp32', choices=['fp32','int8'], help='Precision for compare separate dirs')
    
    # Dataset options
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Path to KITTI dataset root')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test file list')
    
    # Depth range options
    parser.add_argument('--min_depth', type=float, default=1e-3,
                       help='Minimum valid depth for evaluation')
    parser.add_argument('--max_depth', type=float, default=80.0,
                       help='Maximum valid depth in dataset')
    parser.add_argument('--eval_max_depth', type=float, default=80.0,
                       help='Maximum depth for evaluation metrics')
    parser.add_argument('--dual_head_max_depth', type=float, default=15.0,
                       help='Max depth for dual-head composition')
    
    # Output options
    parser.add_argument('--output_json', type=str,
                       help='Path to save results as JSON')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.checkpoint and not args.precomputed_dir:
        parser.error("Either --checkpoint or --precomputed_dir must be provided")
    
    evaluate(args)


if __name__ == '__main__':
    main()
