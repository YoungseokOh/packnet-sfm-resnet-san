#!/usr/bin/env python3
"""Visualize NPU dual-head predictions vs GT depth (LiDAR).

Creates side-by-side comparison:
  - Left: GT depth (valid pixels from LiDAR)
  - Right: NPU prediction (same valid pixel locations)

Usage:
    # Single image test
    python visualize_npu_vs_gt.py \
        --npu_dir /path/to/npu_results \
        --gt_dir /path/to/newest_original_depth_maps \
        --rgb_dir /path/to/image_a6 \
        --output_dir /path/to/video_results \
        --num_samples 1

    # All images
    python visualize_npu_vs_gt.py \
        --npu_dir /path/to/npu_results \
        --gt_dir /path/to/newest_original_depth_maps \
        --rgb_dir /path/to/image_a6 \
        --output_dir /path/to/video_results
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packnet_sfm.visualization.colormaps import create_custom_depth_colormap


def load_npu_dual_head(int_path: Path, frac_path: Path, max_depth: float = 10.0) -> np.ndarray:
    """Load NPU dual-head outputs and compose depth.
    
    Args:
        int_path: Path to integer_sigmoid .npy file
        frac_path: Path to fractional_sigmoid .npy file
        max_depth: Maximum depth for composition
    
    Returns:
        depth: [H, W] depth in meters
    """
    integer_sigmoid = np.load(int_path)
    fractional_sigmoid = np.load(frac_path)
    
    # Handle various shapes: [1, 1, H, W] or [1, H, W] or [H, W]
    if integer_sigmoid.ndim == 4:
        integer_sigmoid = integer_sigmoid[0, 0]
    elif integer_sigmoid.ndim == 3:
        integer_sigmoid = integer_sigmoid[0]
    
    if fractional_sigmoid.ndim == 4:
        fractional_sigmoid = fractional_sigmoid[0, 0]
    elif fractional_sigmoid.ndim == 3:
        fractional_sigmoid = fractional_sigmoid[0]
    
    # Dual-head composition: depth = integer * max_depth + fractional
    depth = integer_sigmoid * max_depth + fractional_sigmoid
    
    return depth


def load_gt_depth_png(path: Path) -> np.ndarray:
    """Load GT depth PNG (uint16 with value = meters * 256)."""
    arr = np.asarray(Image.open(path))
    arr_f = arr.astype(np.float32)
    if arr_f.max() > 255:
        arr_f /= 256.0
    return arr_f


def get_cmap(vmin: float, vmax: float):
    """Get custom depth colormap."""
    return create_custom_depth_colormap(vmin, vmax)


def colorize_points(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Colorize depth values using custom colormap."""
    norm = np.clip((values - vmin) / max(1e-6, (vmax - vmin)), 0, 1)
    cm = get_cmap(vmin, vmax)
    return (cm(norm)[:, :3] * 255).astype(np.uint8)


def draw_points(rgb: np.ndarray, uv: np.ndarray, colors: np.ndarray, radius: int = 1) -> np.ndarray:
    """Draw colored points on RGB image."""
    out = rgb.copy()
    for (u, v), c in zip(uv, colors):
        cv2.circle(out, (int(u), int(v)), radius, (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
    return out


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    """Put label text on image."""
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description='Visualize NPU vs GT depth')
    parser.add_argument('--npu_dir', type=str, required=True,
                        help='NPU results directory (contains integer_sigmoid/, fractional_sigmoid/)')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='GT depth maps directory (newest_original_depth_maps)')
    parser.add_argument('--rgb_dir', type=str, required=True,
                        help='RGB images directory (image_a6)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for visualization')
    parser.add_argument('--max_depth', type=float, default=10.0,
                        help='Max depth for dual-head composition')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Min depth for visualization')
    parser.add_argument('--vis_max_depth', type=float, default=10.0,
                        help='Max depth for visualization colormap')
    parser.add_argument('--point_radius', type=int, default=1,
                        help='Point radius for overlay')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of samples to process (0=all)')
    parser.add_argument('--mask_path', type=str, default=None,
                        help='Optional binary mask (white=valid)')
    args = parser.parse_args()

    npu_dir = Path(args.npu_dir)
    gt_dir = Path(args.gt_dir)
    rgb_dir = Path(args.rgb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    int_dir = npu_dir / 'integer_sigmoid'
    frac_dir = npu_dir / 'fractional_sigmoid'

    # Load optional mask
    mask = None
    if args.mask_path:
        mask_path = Path(args.mask_path)
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            print(f'[INFO] Loaded mask: {mask_path}')

    # Get all npy files
    int_files = sorted(int_dir.glob('*.npy'))
    print(f'[INFO] Found {len(int_files)} NPU result files')

    if args.num_samples > 0:
        int_files = int_files[:args.num_samples]

    processed = 0
    for int_path in int_files:
        stem = int_path.stem
        frac_path = frac_dir / f'{stem}.npy'
        gt_path = gt_dir / f'{stem}.png'
        rgb_path = rgb_dir / f'{stem}.jpg'
        
        if not rgb_path.exists():
            rgb_path = rgb_dir / f'{stem}.png'
        
        if not all([frac_path.exists(), gt_path.exists(), rgb_path.exists()]):
            print(f'[SKIP] Missing files for {stem}')
            continue

        # Load data
        pred_depth = load_npu_dual_head(int_path, frac_path, args.max_depth)
        gt_depth = load_gt_depth_png(gt_path)
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'))

        H, W = gt_depth.shape
        
        # Resize if needed
        if pred_depth.shape != (H, W):
            pred_depth = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_LINEAR)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        # Resize mask if needed
        if mask is not None and mask.shape != (H, W):
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        # Get valid pixels from GT depth
        valid = gt_depth > 0
        if mask_resized is not None:
            valid = valid & (mask_resized > 0)

        # Get coordinates
        v_coords, u_coords = np.where(valid)
        if len(v_coords) == 0:
            print(f'[SKIP] No valid pixels for {stem}')
            continue

        gt_vals = gt_depth[v_coords, u_coords]
        pred_vals = pred_depth[v_coords, u_coords]

        # Filter by depth range
        ok = (gt_vals >= args.min_depth) & (gt_vals <= args.vis_max_depth)
        ok &= (pred_vals >= args.min_depth) & (pred_vals <= args.vis_max_depth)
        
        v_coords = v_coords[ok]
        u_coords = u_coords[ok]
        gt_vals = gt_vals[ok]
        pred_vals = pred_vals[ok]
        uv = np.column_stack([u_coords, v_coords])

        if len(uv) < 50:
            print(f'[SKIP] Too few valid pixels for {stem}')
            continue

        # Colorize
        gt_colors = colorize_points(gt_vals, args.min_depth, args.vis_max_depth)
        pred_colors = colorize_points(pred_vals, args.min_depth, args.vis_max_depth)

        # Draw overlays
        gt_overlay = draw_points(rgb.copy(), uv, gt_colors, args.point_radius)
        pred_overlay = draw_points(rgb.copy(), uv, pred_colors, args.point_radius)

        # Apply mask (black out invalid regions)
        if mask_resized is not None:
            black_region = mask_resized == 0
            gt_overlay[black_region] = [0, 0, 0]
            pred_overlay[black_region] = [0, 0, 0]

        # Add labels
        gt_labeled = put_label(gt_overlay, 'GT (LiDAR)')
        pred_labeled = put_label(pred_overlay, 'NPU Prediction')

        # Compose side-by-side: [GT | Pred]
        combined = np.concatenate([gt_labeled, pred_labeled], axis=1)

        # Save
        out_path = output_dir / f'{stem}_comparison.png'
        Image.fromarray(combined).save(out_path)
        processed += 1

        if processed == 1:
            print(f'\n[FIRST SAMPLE]')
            print(f'  Stem: {stem}')
            print(f'  Valid pixels: {len(uv)}')
            print(f'  GT depth range: [{gt_vals.min():.2f}, {gt_vals.max():.2f}]m')
            print(f'  Pred depth range: [{pred_vals.min():.2f}, {pred_vals.max():.2f}]m')
            print(f'  Output: {out_path}')

        if processed % 100 == 0:
            print(f'[PROGRESS] Processed {processed} images...')

    print(f'\n[DONE] Processed {processed} images')
    print(f'Output directory: {output_dir}')


if __name__ == '__main__':
    main()
