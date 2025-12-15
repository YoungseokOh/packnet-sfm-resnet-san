#!/usr/bin/env python3
"""
Visualize NPU depth prediction results with binary mask applied.
Uses pre-computed NPU outputs (integer_sigmoid, fractional_sigmoid) instead of model inference.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def create_custom_colormap(min_depth=0.1, max_depth=15.0):
    """
    Create custom progressive colormap (red=near, blue=far)
    Smooth transitions between colors at specified depth ranges.
    Automatically adjusts to the given min/max depth range.
    """
    depth_range = max_depth - min_depth
    
    # Define color points in absolute depth values
    absolute_depth_points = [
        (0.1,  (1.0, 0.0, 0.0)),    # Pure red
        (0.3,  (1.0, 0.0, 0.0)),    # Red
        (0.4,  (1.0, 0.15, 0.0)),   # Red-orange
        (0.5,  (1.0, 0.35, 0.0)),   # Orange-red
        (0.6,  (1.0, 0.5, 0.0)),    # Orange
        (0.8,  (1.0, 0.55, 0.0)),   # Orange
        (1.0,  (1.0, 0.6, 0.0)),    # Orange
        (1.1,  (1.0, 0.7, 0.0)),    # Orange-yellow
        (1.25, (1.0, 0.85, 0.0)),   # Yellow-orange
        (1.4,  (1.0, 1.0, 0.0)),    # Pure yellow
        (1.8,  (1.0, 1.0, 0.0)),    # Yellow
        (2.2,  (0.9, 1.0, 0.0)),    # Yellow
        (2.4,  (0.7, 1.0, 0.1)),    # Yellow-green
        (2.5,  (0.5, 1.0, 0.2)),    # Green-yellow
        (2.7,  (0.3, 1.0, 0.3)),    # Green
        (3.0,  (0.1, 1.0, 0.4)),    # Green
        (3.3,  (0.0, 1.0, 0.5)),    # Green-cyan
        (3.5,  (0.0, 1.0, 0.7)),    # Cyan-green
        (3.8,  (0.0, 1.0, 0.85)),   # Cyan
        (4.5,  (0.0, 1.0, 1.0)),    # Pure cyan
        (5.5,  (0.0, 0.9, 1.0)),    # Cyan
        (6.5,  (0.0, 0.7, 1.0)),    # Cyan-blue
        (7.0,  (0.0, 0.5, 1.0)),    # Blue-cyan
        (8.0,  (0.0, 0.3, 1.0)),    # Blue
        (10.0, (0.0, 0.15, 1.0)),   # Blue
        (12.0, (0.0, 0.05, 1.0)),   # Deep blue
        (15.0, (0.0, 0.0, 1.0)),    # Pure blue
    ]
    
    # Filter and adjust points to fit within [min_depth, max_depth]
    depth_points = []
    for d, c in absolute_depth_points:
        if d >= min_depth and d <= max_depth:
            depth_points.append((d, c))
    
    # Ensure we have start and end points
    if len(depth_points) == 0 or depth_points[0][0] > min_depth:
        # Find color for min_depth by interpolation
        for i, (d, c) in enumerate(absolute_depth_points):
            if d >= min_depth:
                depth_points.insert(0, (min_depth, c))
                break
    
    if depth_points[-1][0] < max_depth:
        depth_points.append((max_depth, absolute_depth_points[-1][1]))
    
    # Normalize positions to [0, 1]
    positions = [(d - min_depth) / depth_range for d, c in depth_points]
    colors = [c for d, c in depth_points]
    
    # Ensure positions start at 0 and end at 1
    if positions[0] > 0:
        positions[0] = 0.0
    if positions[-1] < 1:
        positions[-1] = 1.0
    
    custom_cmap = LinearSegmentedColormap.from_list('depth_custom', 
                                                     list(zip(positions, colors)), 
                                                     N=512)
    return custom_cmap


def apply_mask_with_spacing(mask: np.ndarray, spacing: int = 4) -> np.ndarray:
    """
    Apply uniform grid spacing to a binary mask.
    """
    H, W = mask.shape
    sparse_mask = np.zeros_like(mask)
    
    y_indices = np.arange(0, H, spacing)
    x_indices = np.arange(0, W, spacing)
    yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
    sparse_mask[yy, xx] = mask[yy, xx]
    
    return sparse_mask


def load_npu_depth(integer_path: Path, fractional_path: Path, max_depth: float) -> np.ndarray:
    """
    Load NPU prediction and compute depth.
    depth = integer_sigmoid * max_depth + fractional_sigmoid
    """
    integer_sig = np.load(str(integer_path))
    fractional_sig = np.load(str(fractional_path))
    
    # Handle different array shapes
    if integer_sig.ndim == 4:  # (1, 1, H, W)
        integer_sig = integer_sig[0, 0]
    elif integer_sig.ndim == 3:  # (1, H, W)
        integer_sig = integer_sig[0]
    
    if fractional_sig.ndim == 4:
        fractional_sig = fractional_sig[0, 0]
    elif fractional_sig.ndim == 3:
        fractional_sig = fractional_sig[0]
    
    depth = integer_sig * max_depth + fractional_sig
    return depth


def visualize_with_mask(
    rgb: np.ndarray,
    pred_depth: np.ndarray,
    mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    point_size: int,
    cmap,
    sample_name: str,
    alpha: float = 1.0,
    bg_mode: str = 'rgb'
) -> plt.Figure:
    """
    Visualize prediction with mask applied.
    """
    H, W = pred_depth.shape
    
    # Get valid mask coordinates
    valid_coords = np.where(mask > 0)
    num_points = len(valid_coords[0])
    
    # Create background based on mode
    if bg_mode == 'black':
        result = np.zeros((H, W, 3), dtype=np.uint8)
    elif bg_mode == 'gray':
        result = np.full((H, W, 3), 40, dtype=np.uint8)
    else:  # rgb
        result = rgb.copy()
    
    # Draw points at mask locations
    for y, x in zip(valid_coords[0], valid_coords[1]):
        depth_val = pred_depth[y, x]
        if depth_val > 0:
            norm_val = np.clip((depth_val - min_depth) / (max_depth - min_depth), 0, 1)
            color = cmap(norm_val)[:3]
            color = tuple(int(c * 255) for c in color)
            
            if point_size == 0:
                result[y, x] = color
            else:
                cv2.circle(result, (x, y), point_size, color, -1, cv2.LINE_AA)
    
    # Alpha blending
    if alpha < 1.0 and bg_mode == 'rgb':
        result = cv2.addWeighted(result, alpha, rgb, 1 - alpha, 0)
    
    # Calculate statistics
    masked_depths = pred_depth[mask > 0]
    valid_depths = masked_depths[masked_depths > 0]
    
    stats = {
        'num_points': num_points,
        'mean': valid_depths.mean() if len(valid_depths) > 0 else 0,
        'std': valid_depths.std() if len(valid_depths) > 0 else 0,
        'min': valid_depths.min() if len(valid_depths) > 0 else 0,
        'max': valid_depths.max() if len(valid_depths) > 0 else 0
    }
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # 1. Original RGB
    axes[0].imshow(rgb)
    axes[0].set_title(f'RGB: {sample_name}', fontsize=12)
    axes[0].axis('off')
    
    # 2. Mask visualization
    mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
    mask_vis[mask > 0] = [255, 255, 255]
    axes[1].imshow(mask_vis)
    axes[1].set_title(f'Mask ({num_points:,} points, {num_points/mask.size*100:.1f}%)', fontsize=12)
    axes[1].axis('off')
    
    # 3. NPU Prediction with mask
    axes[2].imshow(result)
    axes[2].set_title(
        f'NPU Pred (masked): mean={stats["mean"]:.2f}m, range=[{stats["min"]:.2f}, {stats["max"]:.2f}]m',
        fontsize=12
    )
    axes[2].axis('off')
    
    # Colorbar
    plt.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_depth, vmax=max_depth))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Depth (m)', fontsize=11)
    
    return fig, stats


def find_npu_samples(data_dir: Path, npu_subdir: str):
    """
    Find RGB-NPU pairs.
    """
    rgb_dir = data_dir / 'rgb'
    npu_dir = data_dir / 'npu' / npu_subdir
    integer_dir = npu_dir / 'integer_sigmoid'
    fractional_dir = npu_dir / 'fractional_sigmoid'
    
    if not integer_dir.exists() or not fractional_dir.exists():
        print(f"[ERROR] NPU directories not found:")
        print(f"  Integer: {integer_dir}")
        print(f"  Fractional: {fractional_dir}")
        return []
    
    pairs = []
    for rgb_file in sorted(rgb_dir.glob('*.png')):
        sample_name = rgb_file.stem
        integer_file = integer_dir / f"{sample_name}.npy"
        fractional_file = fractional_dir / f"{sample_name}.npy"
        
        if integer_file.exists() and fractional_file.exists():
            pairs.append((rgb_file, integer_file, fractional_file))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Visualize NPU depth prediction with binary mask')
    parser.add_argument('--data_dir', type=str, default='test_set_v2',
                        help='Directory containing rgb/, npu/, GT/ folders')
    parser.add_argument('--npu_subdir', type=str, 
                        default='resnetsan_dual_head_e50_ncdb_v2_640x384_05_to_15m_A6_ES',
                        help='NPU result subdirectory name')
    parser.add_argument('--mask_path', type=str, default='assets/ncdb_640x384_binary_mask_v3.png',
                        help='Path to binary mask')
    parser.add_argument('--output_dir', type=str, default='outputs/npu_masked_prediction',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process')
    parser.add_argument('--random', action='store_true',
                        help='Random sample selection')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help='Minimum depth')
    parser.add_argument('--max_depth', type=float, default=15.0,
                        help='Maximum depth (for dual-head computation)')
    parser.add_argument('--point_size', type=int, default=2,
                        help='Point size (0=single pixel)')
    parser.add_argument('--spacing', type=int, default=5,
                        help='Grid spacing for mask')
    parser.add_argument('--colormap', type=str, default='custom',
                        choices=['turbo', 'turbo_r', 'jet', 'jet_r', 'viridis', 'plasma', 'custom'],
                        help='Colormap')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='Point opacity')
    parser.add_argument('--bg_mode', type=str, default='rgb',
                        choices=['rgb', 'black', 'gray'],
                        help='Background mode')
    
    args = parser.parse_args()
    
    # Load mask
    mask_path = Path(args.mask_path)
    if not mask_path.exists():
        print(f"[ERROR] Mask not found: {mask_path}")
        return
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    print(f"[INFO] Loaded mask: {mask.shape}, unique values: {np.unique(mask)}")
    
    # Apply spacing
    if args.spacing > 0:
        original_points = np.sum(mask > 0)
        mask = apply_mask_with_spacing(mask, args.spacing)
        sparse_points = np.sum(mask > 0)
        print(f"[INFO] Applied spacing={args.spacing}: {original_points:,} -> {sparse_points:,} points")
    
    # Find samples
    data_dir = Path(args.data_dir)
    pairs = find_npu_samples(data_dir, args.npu_subdir)
    print(f"[INFO] Found {len(pairs)} RGB-NPU pairs")
    
    if len(pairs) == 0:
        print("[ERROR] No pairs found!")
        return
    
    # Sample selection
    if args.random:
        if args.seed is not None:
            random.seed(args.seed)
        else:
            random.seed()
        pairs = random.sample(pairs, min(args.num_samples, len(pairs)))
    else:
        pairs = pairs[:args.num_samples]
    
    # Setup output - clear existing
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        existing_files = list(output_dir.glob('*.png'))
        if existing_files:
            print(f"[INFO] Clearing {len(existing_files)} existing files")
            for f in existing_files:
                f.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Colormap
    if args.colormap == 'custom':
        cmap = create_custom_colormap(args.min_depth, args.max_depth)
    else:
        cmap = getattr(plt.cm, args.colormap)
    
    print(f"""
============================================================
NPU MASKED PREDICTION VISUALIZATION
============================================================
Data dir: {data_dir}
NPU subdir: {args.npu_subdir}
Samples: {len(pairs)}
Depth range: {args.min_depth}m - {args.max_depth}m
Colormap: {args.colormap}
Point size: {args.point_size}
Grid spacing: {args.spacing}px
Output: {output_dir}
============================================================
""")
    
    all_stats = []
    
    for i, (rgb_path, integer_path, fractional_path) in enumerate(pairs):
        sample_name = rgb_path.stem
        print(f"[{i+1}/{len(pairs)}] {rgb_path.name}")
        
        # Load RGB
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load NPU depth
        pred_depth = load_npu_depth(integer_path, fractional_path, args.max_depth)
        
        # Resize if needed
        H, W = mask.shape
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))
        if pred_depth.shape != (H, W):
            pred_depth = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Visualize
        fig, stats = visualize_with_mask(
            rgb, pred_depth, mask,
            args.min_depth, args.max_depth,
            args.point_size, cmap, sample_name,
            args.alpha, args.bg_mode
        )
        
        # Save
        output_path = output_dir / f"{sample_name}_npu_masked.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Saved: {output_path.name}")
        
        all_stats.append(stats)
    
    # Summary
    avg_mean = np.mean([s['mean'] for s in all_stats])
    print(f"""
============================================================
SUMMARY
============================================================
Samples processed: {len(all_stats)}
Average pred depth: {avg_mean:.2f}m
Output directory: {output_dir}
============================================================
""")


if __name__ == '__main__':
    main()
