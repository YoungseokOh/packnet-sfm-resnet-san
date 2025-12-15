#!/usr/bin/env python3
"""
Unified evaluation script for FP32 and NPU Dual-Head depth estimation.
Compares results on identical samples for fair comparison.

Usage:
    # FP32 only
    python scripts/evaluation/evaluate_unified.py --mode fp32 --checkpoint <path>
    
    # NPU only
    python scripts/evaluation/evaluate_unified.py --mode npu --npu_dir <path>
    
    # Compare both
    python scripts/evaluation/evaluate_unified.py --mode compare \
        --checkpoint <path> --npu_dir <path>
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ================================
# METRICS COMPUTATION
# ================================

def compute_depth_metrics(gt: np.ndarray, pred: np.ndarray, 
                          min_depth: float = 0.5, max_depth: float = 15.0) -> Optional[Dict]:
    """
    Compute depth metrics using exclusive mask (same as official evaluation).
    
    Args:
        gt: Ground truth depth [H, W]
        pred: Predicted depth [H, W]
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    
    Returns:
        dict: Metrics or None if no valid pixels
    """
    # Exclusive mask (same as packnet_sfm/utils/depth.py)
    mask = (gt > min_depth) & (gt < max_depth)
    
    if mask.sum() == 0:
        return None
    
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    
    # Threshold-based accuracy
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    # Error metrics
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
    # Additional stats for debugging
    abs_error = np.abs(gt_valid - pred_valid)
    
    return {
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'a1': float(a1),
        'a2': float(a2),
        'a3': float(a3),
        'valid_pixels': int(mask.sum()),
        'mean_abs_error': float(abs_error.mean()),
        'max_abs_error': float(abs_error.max()),
        'gt_mean': float(gt_valid.mean()),
        'pred_mean': float(pred_valid.mean()),
    }


# ================================
# LOADING FUNCTIONS
# ================================

def load_gt_depth(filepath: Path) -> np.ndarray:
    """Load GT depth from PNG (uint16 / 256)."""
    img = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"Failed to load: {filepath}")
    return img.astype(np.float32) / 256.0


def load_npu_dual_head(npu_dir: Path, sample_id: str, max_depth: float = 15.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load NPU dual-head outputs and compose depth."""
    integer_path = npu_dir / "integer_sigmoid" / f"{sample_id}.npy"
    fractional_path = npu_dir / "fractional_sigmoid" / f"{sample_id}.npy"
    
    if not integer_path.exists() or not fractional_path.exists():
        raise FileNotFoundError(f"NPU outputs not found for {sample_id}")
    
    integer_sig = np.load(integer_path).astype(np.float32)
    fractional_sig = np.load(fractional_path).astype(np.float32)
    
    # Squeeze if needed
    if integer_sig.ndim == 4:
        integer_sig = integer_sig[0, 0]
        fractional_sig = fractional_sig[0, 0]
    elif integer_sig.ndim == 3 and integer_sig.shape[0] == 1:
        integer_sig = integer_sig[0]
        fractional_sig = fractional_sig[0]
    
    # Compose depth: integer * 15 + fractional
    depth = integer_sig * max_depth + fractional_sig
    
    return integer_sig, fractional_sig, depth


def load_fp32_model(checkpoint: str):
    """Load FP32 model from checkpoint."""
    from packnet_sfm.models.model_wrapper import ModelWrapper
    from packnet_sfm.utils.config import parse_test_file
    
    logger.info(f"Loading FP32 model: {checkpoint}")
    config, state_dict = parse_test_file(checkpoint, None)
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_state_dict(state_dict, strict=False)
    model_wrapper = model_wrapper.to('cuda')
    model_wrapper.eval()
    
    max_depth = float(config.model.params.max_depth)
    logger.info(f"Model loaded. max_depth={max_depth}")
    
    return model_wrapper, max_depth


def run_fp32_inference(model_wrapper, image_path: Path, max_depth: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run FP32 inference on single image."""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 384), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()
    
    # Forward pass
    with torch.no_grad():
        output = model_wrapper.model({'rgb': img_tensor})
        integer_sig = output[('integer', 0)].cpu().numpy()[0, 0]
        fractional_sig = output[('fractional', 0)].cpu().numpy()[0, 0]
    
    # Compose depth
    depth = integer_sig * max_depth + fractional_sig
    
    return integer_sig, fractional_sig, depth


# ================================
# MAIN EVALUATION
# ================================

def evaluate_compare(args):
    """Compare FP32 and NPU results on same samples."""
    
    # Load FP32 model
    model_wrapper, max_depth = load_fp32_model(args.checkpoint)
    
    # Setup paths
    gt_dir = Path(args.gt_dir)
    npu_dir = Path(args.npu_dir)
    gt_files = sorted(gt_dir.glob('*.png'))
    
    # Load test entries for image paths
    with open(args.test_file) as f:
        test_entries = json.load(f)
    
    filename_to_root = {}
    for entry in test_entries:
        fname = entry['new_filename']
        if fname not in filename_to_root:
            filename_to_root[fname] = entry['dataset_root']
    
    logger.info("=" * 100)
    logger.info("FP32 vs NPU DUAL-HEAD COMPARISON")
    logger.info("=" * 100)
    logger.info(f"GT samples: {len(gt_files)}")
    logger.info(f"Depth range: [{args.min_depth}, {args.max_depth}] m")
    logger.info(f"Max depth (composition): {max_depth} m")
    logger.info("=" * 100)
    
    # Results storage
    fp32_results = []
    npu_results = []
    comparison_results = []
    
    # Header
    logger.info(f"\n{'Sample':<12} | {'FP32 abs_rel':>12} {'FP32 a1':>10} | {'NPU abs_rel':>12} {'NPU a1':>10} | {'Diff abs_rel':>12}")
    logger.info("-" * 90)
    
    for i, gt_path in enumerate(gt_files):
        filename = gt_path.stem
        
        # Find image path
        dataset_root = filename_to_root.get(filename)
        if not dataset_root:
            continue
        
        image_path = None
        for img_folder in ['image_a6', 'images', 'image']:
            for ext in ['.png', '.jpg']:
                candidate = Path(dataset_root) / img_folder / f'{filename}{ext}'
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path:
                break
        
        if not image_path:
            continue
        
        # Check NPU results exist
        if not (npu_dir / "integer_sigmoid" / f"{filename}.npy").exists():
            continue
        
        try:
            # Load GT
            gt_depth = load_gt_depth(gt_path)
            
            # FP32 inference
            fp32_int, fp32_frac, fp32_depth = run_fp32_inference(model_wrapper, image_path, max_depth)
            
            # NPU load
            npu_int, npu_frac, npu_depth = load_npu_dual_head(npu_dir, filename, max_depth)
            
            # Resize if needed
            if fp32_depth.shape != gt_depth.shape:
                fp32_depth = cv2.resize(fp32_depth, (gt_depth.shape[1], gt_depth.shape[0]))
                fp32_int = cv2.resize(fp32_int, (gt_depth.shape[1], gt_depth.shape[0]))
                fp32_frac = cv2.resize(fp32_frac, (gt_depth.shape[1], gt_depth.shape[0]))
            
            if npu_depth.shape != gt_depth.shape:
                npu_depth = cv2.resize(npu_depth, (gt_depth.shape[1], gt_depth.shape[0]))
                npu_int = cv2.resize(npu_int, (gt_depth.shape[1], gt_depth.shape[0]))
                npu_frac = cv2.resize(npu_frac, (gt_depth.shape[1], gt_depth.shape[0]))
            
            # Compute metrics
            fp32_metrics = compute_depth_metrics(gt_depth, fp32_depth, args.min_depth, args.max_depth)
            npu_metrics = compute_depth_metrics(gt_depth, npu_depth, args.min_depth, args.max_depth)
            
            if fp32_metrics is None or npu_metrics is None:
                continue
            
            fp32_metrics['filename'] = filename
            npu_metrics['filename'] = filename
            fp32_results.append(fp32_metrics)
            npu_results.append(npu_metrics)
            
            # Comparison
            diff_abs_rel = npu_metrics['abs_rel'] - fp32_metrics['abs_rel']
            comparison_results.append({
                'filename': filename,
                'fp32_abs_rel': fp32_metrics['abs_rel'],
                'npu_abs_rel': npu_metrics['abs_rel'],
                'diff_abs_rel': diff_abs_rel,
                'fp32_a1': fp32_metrics['a1'],
                'npu_a1': npu_metrics['a1'],
                'fp32_int_mean': float(fp32_int.mean()),
                'npu_int_mean': float(npu_int.mean()),
                'fp32_frac_mean': float(fp32_frac.mean()),
                'npu_frac_mean': float(npu_frac.mean()),
                'fp32_depth_mean': float(fp32_depth.mean()),
                'npu_depth_mean': float(npu_depth.mean()),
            })
            
            # Log every sample
            marker = "⚠️" if abs(diff_abs_rel) > 0.05 else ""
            logger.info(
                f"{filename:<12} | {fp32_metrics['abs_rel']:>12.4f} {fp32_metrics['a1']*100:>9.1f}% | "
                f"{npu_metrics['abs_rel']:>12.4f} {npu_metrics['a1']*100:>9.1f}% | "
                f"{diff_abs_rel:>+12.4f} {marker}"
            )
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue
    
    # Aggregate results
    logger.info("-" * 90)
    
    fp32_agg = {k: np.mean([r[k] for r in fp32_results]) 
                for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}
    npu_agg = {k: np.mean([r[k] for r in npu_results]) 
               for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}
    
    logger.info(f"\n{'='*100}")
    logger.info(f"AGGREGATE RESULTS ({len(fp32_results)} samples)")
    logger.info(f"{'='*100}")
    logger.info(f"{'Metric':<12} | {'FP32':>12} | {'NPU INT8':>12} | {'Diff':>12} | {'Diff %':>10}")
    logger.info("-" * 70)
    
    for metric in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
        fp32_val = fp32_agg[metric]
        npu_val = npu_agg[metric]
        diff = npu_val - fp32_val
        diff_pct = (diff / fp32_val) * 100 if fp32_val != 0 else 0
        
        if metric in ['a1', 'a2', 'a3']:
            logger.info(f"{metric:<12} | {fp32_val*100:>11.2f}% | {npu_val*100:>11.2f}% | {diff*100:>+11.2f}% | {diff_pct:>+9.2f}%")
        else:
            logger.info(f"{metric:<12} | {fp32_val:>12.4f} | {npu_val:>12.4f} | {diff:>+12.4f} | {diff_pct:>+9.2f}%")
    
    # Find worst cases
    logger.info(f"\n{'='*100}")
    logger.info("TOP 10 WORST DEGRADATION (NPU vs FP32)")
    logger.info(f"{'='*100}")
    
    sorted_comp = sorted(comparison_results, key=lambda x: x['diff_abs_rel'], reverse=True)[:10]
    logger.info(f"{'Sample':<12} | {'FP32 abs_rel':>12} | {'NPU abs_rel':>12} | {'Diff':>12} | {'FP32 depth':>12} | {'NPU depth':>12}")
    logger.info("-" * 90)
    
    for c in sorted_comp:
        logger.info(
            f"{c['filename']:<12} | {c['fp32_abs_rel']:>12.4f} | {c['npu_abs_rel']:>12.4f} | "
            f"{c['diff_abs_rel']:>+12.4f} | {c['fp32_depth_mean']:>12.2f} | {c['npu_depth_mean']:>12.2f}"
        )
    
    # Component analysis
    logger.info(f"\n{'='*100}")
    logger.info("DUAL-HEAD COMPONENT ANALYSIS")
    logger.info(f"{'='*100}")
    
    fp32_int_means = [c['fp32_int_mean'] for c in comparison_results]
    npu_int_means = [c['npu_int_mean'] for c in comparison_results]
    fp32_frac_means = [c['fp32_frac_mean'] for c in comparison_results]
    npu_frac_means = [c['npu_frac_mean'] for c in comparison_results]
    
    logger.info(f"Integer sigmoid (FP32):  mean={np.mean(fp32_int_means):.4f}, std={np.std(fp32_int_means):.4f}")
    logger.info(f"Integer sigmoid (NPU):   mean={np.mean(npu_int_means):.4f}, std={np.std(npu_int_means):.4f}")
    logger.info(f"Integer diff:            mean={np.mean(np.array(npu_int_means) - np.array(fp32_int_means)):.6f}")
    logger.info(f"")
    logger.info(f"Fractional sigmoid (FP32): mean={np.mean(fp32_frac_means):.4f}, std={np.std(fp32_frac_means):.4f}")
    logger.info(f"Fractional sigmoid (NPU):  mean={np.mean(npu_frac_means):.4f}, std={np.std(npu_frac_means):.4f}")
    logger.info(f"Fractional diff:           mean={np.mean(np.array(npu_frac_means) - np.array(fp32_frac_means)):.6f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'num_samples': len(fp32_results),
        'fp32_aggregate': fp32_agg,
        'npu_aggregate': npu_agg,
        'fp32_per_image': fp32_results,
        'npu_per_image': npu_results,
        'comparison': comparison_results,
    }
    
    output_file = output_dir / 'fp32_vs_npu_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    logger.info("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Unified FP32/NPU Dual-Head Evaluation')
    parser.add_argument('--mode', type=str, default='compare', choices=['fp32', 'npu', 'compare'])
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/resnetsan01_dual_head_ncdb_v2_640x384_0.5_to_15m_use_film/default_config-train_resnet_san_ncdb_dual_head_640x384-2025.11.26-06h32m26s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt')
    parser.add_argument('--npu_dir', type=str, 
                       default='test_set_v2/npu/resnetsan_dual_head_e50_ncdb_v2_640x384_05_to_15m_A6_ES')
    parser.add_argument('--gt_dir', type=str, default='test_set_v2/GT')
    parser.add_argument('--test_file', type=str, 
                       default='/workspace/data/ncdb-cls-640x384/splits/combined_test.json')
    parser.add_argument('--min_depth', type=float, default=0.5)
    parser.add_argument('--max_depth', type=float, default=15.0)
    parser.add_argument('--output_dir', type=str, default='outputs/comparison')
    args = parser.parse_args()
    
    if args.mode == 'compare':
        evaluate_compare(args)
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented yet")


if __name__ == '__main__':
    main()
