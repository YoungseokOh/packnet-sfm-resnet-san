#!/usr/bin/env python3
"""
Evaluate Dual-Head depth estimation with proper metrics.
Based on validated FP32 evaluation code (evaluate_dual_head_simple.py).

Supports:
- Dual-head outputs (integer + fractional + composed)
- Standard KITTI metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
- JSON output with detailed results

Usage:
    python evaluate_dual_head.py

Configuration:
    - Edit OUTPUT_TYPE, NPU_DIR, GT_DIR below
    - Ensure dual-head directory structure:
      NPU_DIR/
        ├── integer/
        ├── fractional/
        └── composed/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================
# CONFIGURATION
# ================================

# Depth range for evaluation
MIN_DEPTH = 0.5  # Minimum depth in meters
MAX_DEPTH = 15.0  # Maximum depth in meters

# Max depth used for dual-head composition
# NOTE: Must match the value used during NPU inference!
DUAL_HEAD_MAX_DEPTH = 15.0  # meters

# Paths (CLI-overridable defaults)
RESULTS_DIR = Path("/root/neuralflow/service/compiler/aiware/assets/resnetsan_sub1_a6es_results")
DEFAULT_NPU_DIR = Path('outputs/resnetsan_dual_head_seperate_static')
DEFAULT_GT_DIR = None  # will try to infer using test split if not provided
DEFAULT_OUTPUT_JSON = None

# ================================
# LOAD FUNCTIONS
# ================================

def load_dual_head_result(npu_dir: Path, sample_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dual-head outputs (integer, fractional) from NPU results and compute composed depth.
    
    Args:
        npu_dir: Base NPU output directory (contains integer_sigmoid/, fractional_sigmoid/ subdirs)
        sample_id: Sample identifier (e.g., "0000000038")
    
    Returns:
        tuple: (integer_sigmoid, fractional_sigmoid, depth_composed)
               All arrays are [H, W] shape, float32
    """
    integer_path = npu_dir / "integer_sigmoid" / f"{sample_id}.npy"
    fractional_path = npu_dir / "fractional_sigmoid" / f"{sample_id}.npy"
    
    if not integer_path.exists() or not fractional_path.exists():
        raise FileNotFoundError(
            f"Dual-head outputs not found for {sample_id}. "
            f"Expected paths:\n  {integer_path}\n  {fractional_path}"
        )
    
    # Load integer and fractional outputs
    integer_sigmoid = np.load(integer_path).astype(np.float32)
    fractional_sigmoid = np.load(fractional_path).astype(np.float32)
    
    # Handle shape: squeeze if [1, H, W] or [1, 1, H, W]
    if integer_sigmoid.ndim == 4:  # [1, 1, H, W]
        integer_sigmoid = integer_sigmoid[0, 0]
        fractional_sigmoid = fractional_sigmoid[0, 0]
    elif integer_sigmoid.ndim == 3 and integer_sigmoid.shape[0] == 1:  # [1, H, W]
        integer_sigmoid = integer_sigmoid[0]
        fractional_sigmoid = fractional_sigmoid[0]
    
    # Compute composed depth: depth = integer * 15.0 + fractional
    # Integer sigmoid [0,1] represents integer part [0, 15] meters
    # Fractional sigmoid [0,1] represents fractional part [0, 1) meters
    # Combined range: [0, 16) meters (theoretical max is 15.999...)
    depth_composed = integer_sigmoid * DUAL_HEAD_MAX_DEPTH + fractional_sigmoid
    
    return integer_sigmoid, fractional_sigmoid, depth_composed


def load_gt_depth(filepath: Path) -> np.ndarray:
    """
    Load GT depth from .png file.
    GT depth is stored as uint16, scale by 256 to get metric depth.
    
    Args:
        filepath: Path to GT depth .png file
    
    Returns:
        np.ndarray: GT depth in meters, shape [H, W]
    """
    img = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"Failed to load: {filepath}")
    
    # GT is stored as uint16, divide by 256 to get metric depth
    gt_depth = img.astype(np.float32) / 256.0
    
    return gt_depth


def find_gt_for_image(image_id: str, gt_dir: Optional[Path] = None, split_entries: Optional[list] = None, debug: bool = False) -> Optional[Path]:
    """
    Find the GT .png filepath for a given image id.
    Checks the provided gt_dir first. If not found, optionally uses split_entries (combined_test.json) to
    find dataset_root and infer the newest_depth_maps path.
    If still not found, falls back to searching recursively under the split file's dataset root(s).
    """
    # First: check explicit GT dir
    if gt_dir is not None:
        p = gt_dir / f"{image_id}.png"
        if p.exists():
            if debug:
                logger.debug(f"GT found in provided GT_DIR: {p}")
            return p

    # Second: check split entries
    if split_entries:
        # find the entry corresponding to image id
        for e in split_entries:
            # Some entries use 'new_filename' or 'image_path'
            new_fn = str(e.get('new_filename') or e.get('filename') or e.get('image_id') or '')
            if new_fn == image_id or str(e.get('image_path', '')).endswith(f"/{image_id}.png"):
                ds_root = e.get('dataset_root')
                if ds_root:
                    # Try newest_depth_maps path under dataset_root
                    g = Path(ds_root) / 'newest_depth_maps' / f"{image_id}.png"
                    if g.exists():
                        if debug:
                            logger.debug(f"GT found via split dataset_root: {g}")
                        return g
                    # Try the image_path replaced with newest_depth_maps sibling
                    image_path = e.get('image_path')
                    if image_path:
                        img_p = Path(image_path)
                        alt = img_p.parents[1] / 'newest_depth_maps' / f"{image_id}.png"
                        if alt.exists():
                            if debug:
                                logger.debug(f"GT found via split image path substitution: {alt}")
                            return alt

    # Third: naive recursive search under common dataset root (expensive but robust fallback)
    if split_entries:
        # gather distinct dataset_roots and try searching under each of them
        roots = set(e.get('dataset_root') for e in split_entries if e.get('dataset_root'))
        for r in roots:
            candidate = Path(r)
            # Search a few plausible subfolders for newest_depth_maps
            for sub in (candidate, candidate / 'synced_data', candidate / 'images'):
                p = sub / 'newest_depth_maps' / f"{image_id}.png"
                if p.exists():
                    if debug:
                        logger.debug(f"GT found via recursive search: {p}")
                    return p

    # Last resort: search workspace for any file matching the image id (slow)
    pattern = f"**/{image_id}.png"
    for fn in Path('/workspace').glob(pattern):
        if fn.exists():
            if debug:
                logger.debug(f"GT found via workspace recursive search: {fn}")
            return fn

    if debug:
        logger.warning(f"GT file not found for image_id {image_id}. Tried GT_DIR: {gt_dir}")
    return None

# ================================
# CONVERSION FUNCTIONS
# ================================

def sigmoid_to_depth(sigmoid_val: np.ndarray) -> np.ndarray:
    """
    Convert sigmoid output [0,1] to depth.
    NOTE: This function is NOT used for dual-head evaluation!
    Dual-head uses direct composition: integer * 15 + fractional
    
    For single-head models:
    depth = sigmoid_val * MAX_DEPTH → [0, 15.0] m
    """
    depth = sigmoid_val * MAX_DEPTH
    return depth

# ================================
# METRICS COMPUTATION
# ================================

def compute_depth_metrics(gt, pred, min_depth=1e-3, max_depth=80.0, mask_type: str = 'exclusive'):
    """
    Compute depth metrics (same as FP32 evaluation).
    Based on validated evaluate_dual_head_simple.py code.
    
    Args:
        gt: Ground truth depth [H, W]
        pred: Predicted depth [H, W]
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    
    Returns:
        dict: Metrics dictionary or None if no valid pixels
    """
    # Mask out invalid depths
    if mask_type == 'inclusive':
        mask = (gt >= min_depth) & (gt <= max_depth)
    else:
        mask = (gt > min_depth) & (gt < max_depth)
    
    if mask.sum() == 0:
        return None
    
    gt = gt[mask]
    pred = pred[mask]
    
    # Threshold-based accuracy
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
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'a1': float(a1),
        'a2': float(a2),
        'a3': float(a3),
        'valid_pixels': int(mask.sum()),
        'total_pixels': int(gt.size)
    }

# ================================
# MAIN EVALUATION
# ================================

def main():
    """Main evaluation function."""
    global DUAL_HEAD_MAX_DEPTH
    logger.info("=" * 90)
    parser = argparse.ArgumentParser(description='Evaluate Dual-Head NPU outputs against GT')
    parser.add_argument('--npu_dir', type=str, default=str(DEFAULT_NPU_DIR), help='NPU output directory (contains integer_sigmoid, fractional_sigmoid)')
    parser.add_argument('--gt_dir', type=str, default=None, help='GT directory containing <image_id>.png (optional)')
    parser.add_argument('--test_file', type=str, default='/workspace/data/ncdb-cls-640x384/splits/combined_test.json', help='JSON split file to map image ids to dataset roots')
    parser.add_argument('--min_depth', type=float, default=0.5, help='Min depth to evaluate')
    parser.add_argument('--max_depth', type=float, default=15.0, help='Max depth to evaluate')
    parser.add_argument('--mask', choices=['inclusive', 'exclusive'], default='exclusive', help='Mask behaviour for min/max bounds')
    parser.add_argument('--dual_head_max_depth', type=float, default=15.0, help='Dual-head max depth used for composition')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save evaluation json output')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging (prints GT paths etc)')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Read evaluation parameters as locals (avoid shadowing module globals)
    min_depth = float(args.min_depth)
    max_depth = float(args.max_depth)
    dual_head_max_depth = float(args.dual_head_max_depth)
    # Update global so load_dual_head_result picks the correct composition parameter
    DUAL_HEAD_MAX_DEPTH = dual_head_max_depth

    logger.info("DUAL-HEAD DEPTH ESTIMATION EVALUATION")
    logger.info("=" * 90)
    logger.info(f"Model:          Dual-Head (Integer + Fractional)")
    logger.info(f"Depth range:    [{min_depth}, {max_depth}] m")
    logger.info(f"Composition:    depth = integer * {dual_head_max_depth} + fractional")
    NPU_DIR = Path(args.npu_dir)
    GT_DIR = Path(args.gt_dir) if args.gt_dir else None
    OUTPUT_JSON = Path(args.output_json) if args.output_json else (NPU_DIR / "evaluation_results.json")
    mask_type = args.mask
    test_file = Path(args.test_file) if args.test_file else None
    logger.info(f"NPU results:    {NPU_DIR}")
    logger.info(f"GT directory:   {GT_DIR}")
    logger.info(f"Test file:      {test_file}")
    logger.info(f"Mask type:      {mask_type}")
    logger.info("=" * 90)
    
    # Check directories
    if not NPU_DIR.exists():
        logger.error(f"NPU directory not found: {NPU_DIR}")
        return
    
    if GT_DIR is not None and not GT_DIR.exists():
        logger.error(f"GT directory not found: {GT_DIR}")
        return
    
    # Check dual-head subdirectories
    integer_dir = NPU_DIR / "integer_sigmoid"
    fractional_dir = NPU_DIR / "fractional_sigmoid"
    
    if not integer_dir.exists() or not fractional_dir.exists():
        logger.error("Dual-head directory structure not found!")
        logger.error(f"Expected: {NPU_DIR}/integer_sigmoid/, {NPU_DIR}/fractional_sigmoid/")
        return
    
    # Load test split entries
    split_entries = None
    if test_file and test_file.exists():
        try:
            split_entries = json.load(open(test_file))
        except Exception:
            logger.warning(f"Failed to load test file: {test_file}")

    # Get sample IDs from integer_sigmoid directory
    integer_files = sorted(list(integer_dir.glob("*.npy")))
    logger.info(f"\nTotal samples: {len(integer_files)}\n")
    
    if len(integer_files) == 0:
        logger.error("No .npy files found in integer_sigmoid directory!")
        return
    
    # Header
    logger.info(
        f"{'Image ID':<15} {'abs_rel':>12} {'sq_rel':>12} {'rmse':>12} {'rmse_log':>12} "
        f"{'a1':>10} {'a2':>10} {'a3':>10} {'valid%':>8}"
    )
    logger.info("-" * 110)
    
    results_list = []
    debug_printed = False  # Flag to print debug info only once
    
    for idx, integer_file in enumerate(integer_files, 1):
        image_id = integer_file.stem
        # Try to locate the GT file path
        gt_file = None
        # Primary: explicit GT dir if provided
        if GT_DIR is not None:
            gt_file = GT_DIR / f"{image_id}.png"
        # Secondary: use split entries to map dataset_root per sample
        if (gt_file is None or not gt_file.exists()) and split_entries is not None:
            found = find_gt_for_image(image_id, gt_dir=GT_DIR, split_entries=split_entries, debug=args.debug)
            if found:
                gt_file = found
        
        if gt_file is None or not Path(gt_file).exists():
            logger.warning(f"GT not found: {image_id}; tried: {GT_DIR} and split_file paths")
            continue
        
        try:
            # Load dual-head outputs
            integer_sig, fractional_sig, depth_composed = load_dual_head_result(NPU_DIR, image_id)
            
            # Load GT depth (already in meters after /256)
            gt_depth = load_gt_depth(gt_file)
            
            # Debug output for first image
            if not debug_printed:
                logger.info("\n" + "=" * 90)
                logger.info("DEBUG: Dual-Head Output Verification (First Image)")
                logger.info("=" * 90)
                logger.info(f"Integer sigmoid:")
                logger.info(f"  Range: [{integer_sig.min():.6f}, {integer_sig.max():.6f}]")
                logger.info(f"  Mean:  {integer_sig.mean():.6f}")
                logger.info(f"  Contribution to depth: [{integer_sig.min()*DUAL_HEAD_MAX_DEPTH:.2f}, {integer_sig.max()*DUAL_HEAD_MAX_DEPTH:.2f}] m")
                logger.info(f"Fractional sigmoid:")
                logger.info(f"  Range: [{fractional_sig.min():.6f}, {fractional_sig.max():.6f}]")
                logger.info(f"  Mean:  {fractional_sig.mean():.6f}")
                logger.info(f"  Contribution to depth: [{fractional_sig.min()*DUAL_HEAD_MAX_DEPTH:.2f}, {fractional_sig.max()*DUAL_HEAD_MAX_DEPTH:.2f}] m")
                logger.info(f"Composed depth:")
                logger.info(f"  Range: [{depth_composed.min():.2f}, {depth_composed.max():.2f}] m")
                logger.info(f"  Mean:  {depth_composed.mean():.2f} m")
                logger.info(f"GT depth (valid pixels only):")
                logger.info(f"  Range: [{gt_depth[gt_depth > 0].min():.2f}, {gt_depth[gt_depth > 0].max():.2f}] m")
                logger.info(f"  Mean:  {gt_depth[gt_depth > 0].mean():.2f} m")
                logger.info(f"Composition formula: depth = integer * {DUAL_HEAD_MAX_DEPTH} + fractional")
                logger.info("=" * 90 + "\n")
                debug_printed = True
            
            # Compute metrics using composed depth
            metrics = compute_depth_metrics(
                gt_depth,
                depth_composed,
                min_depth,
                max_depth,
                mask_type
            )
            
            if metrics is None:
                logger.warning(f"No valid pixels: {image_id}")
                continue
            
            # Calculate valid pixel percentage
            valid_pct = (metrics['valid_pixels'] / metrics['total_pixels']) * 100
            
            # Store results
            result = {
                "image_id": image_id,
                "gt_path": str(gt_file),
                **metrics
            }
            results_list.append(result)
            
            # Print row
            logger.info(
                f"{image_id:<15} "
                f"{metrics['abs_rel']:>12.4f} "
                f"{metrics['sq_rel']:>12.2f} "
                f"{metrics['rmse']:>12.4f} "
                f"{metrics['rmse_log']:>12.4f} "
                f"{metrics['a1']:>9.1%} "
                f"{metrics['a2']:>9.1%} "
                f"{metrics['a3']:>9.1%} "
                f"{valid_pct:>7.1f}%"
            )
            
        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
            continue
    
    # Compute aggregate metrics
    if not results_list:
        logger.error("\n❌ No valid results!")
        return
    
    logger.info("-" * 110)
    
    agg_metrics = {}
    for key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
        agg_metrics[key] = np.mean([r[key] for r in results_list])
    
    total_valid = sum(r['valid_pixels'] for r in results_list)
    total_pixels = sum(r['total_pixels'] for r in results_list)
    overall_valid_pct = (total_valid / total_pixels) * 100 if total_pixels > 0 else 0
    
    # Print summary
    logger.info(f"\n{'='*90}")
    logger.info(f"{'AGGREGATE METRICS':<15} "
                f"{agg_metrics['abs_rel']:>12.4f} "
                f"{agg_metrics['sq_rel']:>12.2f} "
                f"{agg_metrics['rmse']:>12.4f} "
                f"{agg_metrics['rmse_log']:>12.4f} "
                f"{agg_metrics['a1']:>9.1%} "
                f"{agg_metrics['a2']:>9.1%} "
                f"{agg_metrics['a3']:>9.1%} "
                f"{overall_valid_pct:>7.1f}%")
    logger.info(f"{'='*90}")
    
    logger.info(f"\n✅ Evaluated {len(results_list)} samples successfully")
    logger.info(f"📊 Valid pixels: {total_valid:,} / {total_pixels:,} ({overall_valid_pct:.1f}%)")
    
    # Save JSON results
    output_data = {
        "model": "Dual-Head ResNetSAN",
        "model_file": str(NPU_DIR.name),
        "num_samples": len(results_list),
        "config": {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "dual_head_max_depth": DUAL_HEAD_MAX_DEPTH,
            "composition_formula": f"depth = integer_sigmoid * {DUAL_HEAD_MAX_DEPTH} + fractional_sigmoid",
            "output_type": "DUAL_HEAD"
        },
        "aggregate_metrics": agg_metrics,
        "overall_valid_pixels": int(total_valid),
        "overall_total_pixels": int(total_pixels),
        "overall_valid_percentage": float(overall_valid_pct),
        "per_image_results": results_list
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {OUTPUT_JSON}")
    logger.info("=" * 90 + "\n")


if __name__ == "__main__":
    main()
