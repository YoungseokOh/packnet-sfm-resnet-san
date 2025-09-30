#!/usr/bin/env python3
"""
Compare evaluation results between:
1. infer_ncdb.py --eval (on-the-fly evaluation)
2. evaluate_ncdb_depth_maps.py (saved depth evaluation)

This script helps identify discrepancies in evaluation metrics.
"""

import argparse
import json
import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.depth import (
    write_depth, inv2depth, load_depth, compute_depth_metrics, 
    post_process_inv_depth
)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare PackNet-SfM evaluation methods')
    
    # Model and config
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint (.ckpt)')
    parser.add_argument('--config', type=str, required=True, help='YAML config')
    
    # Dataset
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Dataset root for NCDB')
    parser.add_argument('--split_json', type=str, required=True,
                        help='Split JSON file')
    parser.add_argument('--depth_variants', type=str,
                        default='newest_depth_maps,newest_synthetic_depth_maps,new_depth_maps,depth_maps',
                        help='Comma-separated priority of depth variants')
    
    # Evaluation settings
    parser.add_argument('--resize_w', type=int, default=640, help='Resize width')
    parser.add_argument('--resize_h', type=int, default=384, help='Resize height')
    parser.add_argument('--interp', type=str, default='lanczos',
                        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'])
    parser.add_argument('--mask_file', type=str, default=None,
                        help='Binary ROI mask image (png)')
    parser.add_argument('--min_depth', type=float, default=0.3)
    parser.add_argument('--max_depth', type=float, default=100.0)
    parser.add_argument('--crop', type=str, default='', choices=['', 'garg'])
    parser.add_argument('--scale_output', type=str, default='top-center')
    parser.add_argument('--use_gt_scale', action='store_true')
    parser.add_argument('--flip_tta', action='store_true')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save comparison results')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples for detailed comparison (0 for all)')
    parser.add_argument('--eval_full', action='store_true',
                        help='Evaluate on full dataset in addition to sample comparison')
    
    return parser.parse_args()


def _pil_interp(name: str):
    """Get PIL interpolation method"""
    name = (name or '').lower()
    if name in ('nearest', 'nn'):
        return Image.NEAREST
    if name in ('bilinear', 'linear'):
        return Image.BILINEAR
    if name in ('bicubic', 'cubic'):
        return Image.BICUBIC
    return Image.LANCZOS


def resize_to(image: Image.Image, height: int, width: int, interp: str = 'lanczos'):
    """Resize PIL image"""
    resample = _pil_interp(interp)
    return image.resize((int(width), int(height)), resample=resample)


def load_split_entries(split_path: Path, dataset_root: Path) -> List[Dict]:
    """Load and normalize split entries"""
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    
    with open(split_path, 'r') as f:
        data = json.load(f)
    
    entries = []
    for item in data:
        if 'image_path' in item:
            # Convert image_path to standard format
            p = Path(item['image_path'])
            stem = p.stem
            base_dir = p.parent
            if base_dir.name == 'image_a6':
                base_dir = base_dir.parent
            
            # Try to make relative to dataset_root
            try:
                rel_base = str(base_dir.relative_to(dataset_root))
            except:
                rel_base = str(base_dir)
            
            entries.append({
                'dataset_root': rel_base,
                'new_filename': stem,
                'image_path': item['image_path']
            })
        elif 'dataset_root' in item and 'new_filename' in item:
            # Standard format
            base = Path(item['dataset_root'])
            if not base.is_absolute():
                base = dataset_root / base
            stem = item['new_filename']
            img_path = str(base / 'image_a6' / f'{stem}.png')
            
            entries.append({
                'dataset_root': item['dataset_root'],
                'new_filename': item['new_filename'],
                'image_path': img_path
            })
    
    return entries


def find_gt_path(dataset_root: Path, entry: Dict, variants: List[str]) -> Optional[Path]:
    """Find GT depth path for an entry"""
    base = Path(entry['dataset_root'])
    if not base.is_absolute():
        base = dataset_root / base
    stem = entry['new_filename']
    
    for v in variants:
        p = base / v / f'{stem}.png'
        if p.exists():
            return p
    return None


def load_mask(mask_file: str, h: int, w: int) -> np.ndarray:
    """Load and resize binary mask"""
    m = (np.array(Image.open(mask_file).convert('L')) > 0).astype(np.uint8)
    if m.shape[0] != h or m.shape[1] != w:
        m_img = Image.fromarray((m * 255).astype(np.uint8), mode='L')
        m_img = m_img.resize((w, h), Image.NEAREST)
        m = (np.array(m_img) > 0).astype(np.uint8)
    return m


@torch.no_grad()
def method1_infer_eval(model_wrapper, image_path: str, gt_path: str, 
                       args, device, dtype) -> Dict:
    """
    Method 1: infer_ncdb.py style - evaluate on-the-fly
    Returns metrics and intermediate values for debugging
    """
    image_shape = (args.resize_h, args.resize_w)
    
    # Load and preprocess image
    image = load_image(image_path)
    image = resize_to(image, height=image_shape[0], width=image_shape[1], 
                     interp=args.interp)
    image_tensor = to_tensor(image).unsqueeze(0).to(device=device, dtype=dtype)
    
    # Run inference
    pred_inv_depth = model_wrapper.depth(image_tensor)['inv_depths'][0]
    
    # Apply TTA if requested
    if args.flip_tta:
        flipped = torch.flip(image_tensor, dims=[3])
        pred_inv_depth_flipped = model_wrapper.depth(flipped)['inv_depths'][0]
        pred_inv_depth = post_process_inv_depth(pred_inv_depth, pred_inv_depth_flipped, 
                                               method='mean')
    
    # Convert to depth
    pred_depth = inv2depth(pred_inv_depth)
    
    # Load GT
    gt_np = load_depth(str(gt_path))
    
    # Apply mask to GT if provided
    if args.mask_file and os.path.exists(args.mask_file):
        mh, mw = gt_np.shape[:2]
        m = load_mask(args.mask_file, mh, mw).astype(np.float32)
        gt_np = gt_np * m
    
    # Prepare tensors for metrics
    gt_t = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    
    # Create eval args
    eval_args = Namespace(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        crop=args.crop,
        scale_output=args.scale_output
    )
    
    # Compute metrics
    metrics = compute_depth_metrics(eval_args, gt_t, pred_depth, 
                                   use_gt_scale=args.use_gt_scale)
    
    return {
        'metrics': metrics,
        'pred_depth': pred_depth.cpu(),
        'gt_depth': gt_t.cpu(),
        'image_shape': image_shape,
        'pred_shape': pred_depth.shape,
        'gt_shape': gt_t.shape
    }


@torch.no_grad()
def method2_save_load_eval(model_wrapper, image_path: str, gt_path: str,
                           args, device, dtype, temp_dir: str) -> Dict:
    """
    Method 2: Save depth then load and evaluate (evaluate_ncdb_depth_maps.py style)
    Returns metrics and intermediate values for debugging
    """
    image_shape = (args.resize_h, args.resize_w)
    
    # Load and preprocess image
    image = load_image(image_path)
    image = resize_to(image, height=image_shape[0], width=image_shape[1], 
                     interp=args.interp)
    image_tensor = to_tensor(image).unsqueeze(0).to(device=device, dtype=dtype)
    
    # Run inference
    pred_inv_depth = model_wrapper.depth(image_tensor)['inv_depths'][0]
    
    # Apply TTA if requested
    if args.flip_tta:
        flipped = torch.flip(image_tensor, dims=[3])
        pred_inv_depth_flipped = model_wrapper.depth(flipped)['inv_depths'][0]
        pred_inv_depth = post_process_inv_depth(pred_inv_depth, pred_inv_depth_flipped, 
                                               method='mean')
    
    # Convert to depth
    pred_depth = inv2depth(pred_inv_depth)
    
    # Save depth to NPZ
    stem = Path(image_path).stem
    temp_npz = os.path.join(temp_dir, f'{stem}.npz')
    write_depth(temp_npz, depth=pred_depth)
    
    # Load back the saved depth
    pred_np_loaded = load_depth(temp_npz)
    
    # Load GT
    gt_np = load_depth(str(gt_path))
    
    # Apply mask to GT if provided
    if args.mask_file and os.path.exists(args.mask_file):
        mh, mw = gt_np.shape[:2]
        m = load_mask(args.mask_file, mh, mw).astype(np.float32)
        gt_np = gt_np * m
    
    # Prepare tensors for metrics
    gt_t = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0)
    pred_t_loaded = torch.tensor(pred_np_loaded).unsqueeze(0).unsqueeze(0)
    
    # Create eval args
    eval_args = Namespace(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        crop=args.crop,
        scale_output=args.scale_output
    )
    
    # Compute metrics
    metrics = compute_depth_metrics(eval_args, gt_t, pred_t_loaded, 
                                   use_gt_scale=args.use_gt_scale)
    
    # Also check save/load consistency
    pred_np_direct = pred_depth.cpu().numpy().squeeze()
    save_load_diff = np.abs(pred_np_loaded - pred_np_direct)
    
    return {
        'metrics': metrics,
        'pred_depth_direct': pred_depth.cpu(),
        'pred_depth_loaded': pred_t_loaded,
        'gt_depth': gt_t,
        'save_load_max_diff': save_load_diff.max(),
        'save_load_mean_diff': save_load_diff.mean(),
        'pred_shape_direct': pred_depth.shape,
        'pred_shape_loaded': pred_t_loaded.shape,
        'gt_shape': gt_t.shape
    }


def evaluate_full_dataset(model_wrapper, valid_entries, args, device, dtype, temp_dir):
    """Evaluate both methods on the full dataset"""
    metric_names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    
    all_metrics1 = []
    all_metrics2 = []
    
    print(f"\nEvaluating full dataset ({len(valid_entries)} samples)...")
    print("This may take a while...")
    
    for entry, gt_path in tqdm(valid_entries, desc="Full evaluation"):
        image_path = entry['image_path']
        
        try:
            # Method 1: Direct evaluation
            result1 = method1_infer_eval(model_wrapper, image_path, gt_path, 
                                         args, device, dtype)
            
            # Method 2: Save-load evaluation
            result2 = method2_save_load_eval(model_wrapper, image_path, gt_path,
                                             args, device, dtype, temp_dir)
            
            # Extract metrics
            metrics1 = result1['metrics'].cpu().numpy()
            metrics2 = result2['metrics'].cpu().numpy()
            
            all_metrics1.append(metrics1)
            all_metrics2.append(metrics2)
            
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            continue
    
    # Convert to numpy arrays
    all_metrics1 = np.array(all_metrics1)
    all_metrics2 = np.array(all_metrics2)
    
    # Compute averages
    avg_metrics1 = all_metrics1.mean(axis=0)
    avg_metrics2 = all_metrics2.mean(axis=0)
    
    # Print results in a nice table
    print(f"\n{'='*80}")
    print(f"FULL DATASET EVALUATION RESULTS ({len(all_metrics1)} samples)")
    print(f"{'='*80}")
    print(f"\n{'Metric':<10} {'Method 1 (Direct)':<20} {'Method 2 (Save/Load)':<20} {'Difference':<15}")
    print(f"{'-'*65}")
    
    for i, name in enumerate(metric_names):
        diff = avg_metrics1[i] - avg_metrics2[i]
        diff_pct = (diff / avg_metrics1[i] * 100) if avg_metrics1[i] != 0 else 0
        print(f"{name:<10} {avg_metrics1[i]:<20.6f} {avg_metrics2[i]:<20.6f} {diff:>7.6f} ({diff_pct:>+6.2f}%)")
    
    # Additional statistics
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    
    print("\nMethod differences (Method1 - Method2):")
    diffs = all_metrics1 - all_metrics2
    
    print(f"\n{'Metric':<10} {'Mean Diff':<15} {'Std Diff':<15} {'Max |Diff|':<15} {'% Samples |Diff|>0.001':<20}")
    print(f"{'-'*70}")
    
    for i, name in enumerate(metric_names):
        mean_diff = diffs[:, i].mean()
        std_diff = diffs[:, i].std()
        max_abs_diff = np.abs(diffs[:, i]).max()
        pct_sig = (np.abs(diffs[:, i]) > 0.001).mean() * 100
        print(f"{name:<10} {mean_diff:>14.6f} {std_diff:>14.6f} {max_abs_diff:>14.6f} {pct_sig:>19.1f}%")
    
    return {
        'all_metrics1': all_metrics1,
        'all_metrics2': all_metrics2,
        'avg_metrics1': avg_metrics1,
        'avg_metrics2': avg_metrics2,
        'diffs': diffs
    }


def compare_methods(args):
    """Main comparison function"""
    # Setup paths
    dataset_root = Path(args.dataset_root)
    split_path = Path(args.split_json)
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    
    # Load model
    print("Loading model...")
    config, state_dict = parse_test_file(args.checkpoint)
    model_wrapper = ModelWrapper(config, load_datasets=False)
    model_wrapper.load_state_dict(state_dict)
    
    # Setup device and dtype
    dtype = torch.float16 if hasattr(config, 'half') and config.half else torch.float32
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_wrapper = model_wrapper.to(device, dtype=dtype)
    model_wrapper.eval()
    
    # Load split entries
    print("Loading split entries...")
    entries = load_split_entries(split_path, dataset_root)
    variants = [v.strip() for v in args.depth_variants.split(',')]
    
    # Filter entries with valid GT
    valid_entries = []
    for entry in entries:
        gt_path = find_gt_path(dataset_root, entry, variants)
        if gt_path:
            valid_entries.append((entry, gt_path))
    
    print(f"Found {len(valid_entries)} valid entries with GT")
    
    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, 'temp_depths')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize results
    results = {
        'config': vars(args),
        'total_samples': len(valid_entries),
        'metric_names': ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    }
    
    # Full dataset evaluation if requested
    if args.eval_full or args.num_samples == 0:
        full_results = evaluate_full_dataset(model_wrapper, valid_entries, args, 
                                            device, dtype, temp_dir)
        results['full_dataset'] = {
            'num_samples': len(full_results['all_metrics1']),
            'avg_metrics_method1': full_results['avg_metrics1'].tolist(),
            'avg_metrics_method2': full_results['avg_metrics2'].tolist(),
            'avg_diff': (full_results['avg_metrics1'] - full_results['avg_metrics2']).tolist(),
            'diff_statistics': {
                'mean': full_results['diffs'].mean(axis=0).tolist(),
                'std': full_results['diffs'].std(axis=0).tolist(),
                'max_abs': np.abs(full_results['diffs']).max(axis=0).tolist()
            }
        }
    
    # Detailed sample comparison
    if args.num_samples > 0:
        num_samples = min(args.num_samples, len(valid_entries))
        sample_results = []
        
        print(f"\nDetailed comparison for {num_samples} samples...")
        for i, (entry, gt_path) in enumerate(tqdm(valid_entries[:num_samples])):
            image_path = entry['image_path']
            stem = entry['new_filename']
            
            # Method 1: Direct evaluation
            result1 = method1_infer_eval(model_wrapper, image_path, gt_path, 
                                         args, device, dtype)
            
            # Method 2: Save-load evaluation
            result2 = method2_save_load_eval(model_wrapper, image_path, gt_path,
                                             args, device, dtype, temp_dir)
            
            # Extract metrics
            metrics1 = result1['metrics'].cpu().numpy()
            metrics2 = result2['metrics'].cpu().numpy()
            
            # Store results
            sample_result = {
                'stem': stem,
                'image_path': image_path,
                'gt_path': str(gt_path),
                'method1_metrics': metrics1.tolist(),
                'method2_metrics': metrics2.tolist(),
                'metrics_diff': (metrics1 - metrics2).tolist(),
                'save_load_diff': {
                    'max': float(result2['save_load_max_diff']),
                    'mean': float(result2['save_load_mean_diff'])
                },
                'shapes': {
                    'method1_pred': list(result1['pred_shape']),
                    'method1_gt': list(result1['gt_shape']),
                    'method2_pred_direct': list(result2['pred_shape_direct']),
                    'method2_pred_loaded': list(result2['pred_shape_loaded']),
                    'method2_gt': list(result2['gt_shape'])
                }
            }
            sample_results.append(sample_result)
            
            # Print detailed comparison for first few samples
            if i < 3:
                print(f"\n{'='*60}")
                print(f"Sample {i+1}: {stem}")
                print(f"{'='*60}")
                print(f"Image: {image_path}")
                print(f"GT:    {gt_path}")
                print(f"\nShapes:")
                print(f"  Method1 - pred: {result1['pred_shape']}, gt: {result1['gt_shape']}")
                print(f"  Method2 - pred_direct: {result2['pred_shape_direct']}, "
                      f"pred_loaded: {result2['pred_shape_loaded']}, gt: {result2['gt_shape']}")
                print(f"\nSave/Load consistency:")
                print(f"  Max diff:  {result2['save_load_max_diff']:.6f}")
                print(f"  Mean diff: {result2['save_load_mean_diff']:.6f}")
                print(f"\nMetrics comparison:")
                print(f"{'Metric':<10} {'Method1':<12} {'Method2':<12} {'Diff':<12}")
                print(f"{'-'*46}")
                for j, name in enumerate(results['metric_names']):
                    m1 = metrics1[j]
                    m2 = metrics2[j]
                    diff = m1 - m2
                    print(f"{name:<10} {m1:<12.6f} {m2:<12.6f} {diff:<12.6f}")
        
        # Compute sample statistics
        all_metrics1 = np.array([r['method1_metrics'] for r in sample_results])
        all_metrics2 = np.array([r['method2_metrics'] for r in sample_results])
        all_diffs = all_metrics1 - all_metrics2
        
        results['sample_comparison'] = {
            'num_samples': num_samples,
            'samples': sample_results,
            'avg_metrics_method1': all_metrics1.mean(axis=0).tolist(),
            'avg_metrics_method2': all_metrics2.mean(axis=0).tolist(),
            'avg_diff': all_diffs.mean(axis=0).tolist(),
            'std_diff': all_diffs.std(axis=0).tolist()
        }
    
    # Save results to JSON
    output_file = os.path.join(args.output_dir, 'comparison_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Save summary to text file
    summary_file = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Comparison Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Configuration:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTotal samples in dataset: {len(valid_entries)}\n")
        
        if 'full_dataset' in results:
            f.write(f"\n{'='*80}\n")
            f.write(f"FULL DATASET RESULTS ({results['full_dataset']['num_samples']} samples)\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{'Metric':<10} {'Method1':<15} {'Method2':<15} {'Difference':<15}\n")
            f.write(f"{'-'*55}\n")
            for i, name in enumerate(results['metric_names']):
                m1 = results['full_dataset']['avg_metrics_method1'][i]
                m2 = results['full_dataset']['avg_metrics_method2'][i]
                diff = results['full_dataset']['avg_diff'][i]
                f.write(f"{name:<10} {m1:<15.6f} {m2:<15.6f} {diff:<15.6f}\n")
        
        if 'sample_comparison' in results:
            f.write(f"\n{'='*80}\n")
            f.write(f"SAMPLE COMPARISON ({results['sample_comparison']['num_samples']} samples)\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{'Metric':<10} {'Method1':<15} {'Method2':<15} {'Diff Mean':<15} {'Diff Std':<15}\n")
            f.write(f"{'-'*70}\n")
            for i, name in enumerate(results['metric_names']):
                m1 = results['sample_comparison']['avg_metrics_method1'][i]
                m2 = results['sample_comparison']['avg_metrics_method2'][i]
                diff_mean = results['sample_comparison']['avg_diff'][i]
                diff_std = results['sample_comparison']['std_diff'][i]
                f.write(f"{name:<10} {m1:<15.6f} {m2:<15.6f} {diff_mean:<15.6f} {diff_std:<15.6f}\n")
    
    print(f"Summary saved to {summary_file}")
    
    # Clean up temp directory
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print("\nComparison complete!")


def main():
    args = parse_args()
    compare_methods(args)


if __name__ == '__main__':
    main()