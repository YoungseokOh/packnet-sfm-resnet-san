#!/usr/bin/env python3
"""
Visualize depth prediction with binary mask applied.
This simulates how the mask will look when applied to all predictions.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.models.model_wrapper import ModelWrapper


def create_custom_colormap(min_depth=0.1, max_depth=15.0):
    """
    Create custom progressive colormap (red=near, blue=far)
    Smooth transitions between colors at specified depth ranges.
    
    Color ranges:
    - 0.1 ~ 0.5m:  Red
    - 0.5 ~ 1.25m: Orange  
    - 1.25 ~ 2.5m: Yellow
    - 2.5 ~ 3.5m:  Green
    - 3.5 ~ 7m:    Cyan
    - 7 ~ 15m:     Blue
    """
    depth_range = max_depth - min_depth
    
    # Smooth gradient with transition zones
    depth_points = [
        # Red zone (0.1 ~ 0.5m)
        (0.1,  (1.0, 0.0, 0.0)),    # Pure red
        (0.3,  (1.0, 0.0, 0.0)),    # Red
        # Red -> Orange transition (0.4 ~ 0.6m)
        (0.4,  (1.0, 0.15, 0.0)),   # Red-orange
        (0.5,  (1.0, 0.35, 0.0)),   # Orange-red
        # Orange zone (0.5 ~ 1.25m)
        (0.6,  (1.0, 0.5, 0.0)),    # Orange
        (0.8,  (1.0, 0.55, 0.0)),   # Orange
        (1.0,  (1.0, 0.6, 0.0)),    # Orange
        # Orange -> Yellow transition (1.0 ~ 1.5m)
        (1.1,  (1.0, 0.7, 0.0)),    # Orange-yellow
        (1.25, (1.0, 0.85, 0.0)),   # Yellow-orange
        # Yellow zone (1.25 ~ 2.5m)
        (1.4,  (1.0, 1.0, 0.0)),    # Pure yellow
        (1.8,  (1.0, 1.0, 0.0)),    # Yellow
        (2.2,  (0.9, 1.0, 0.0)),    # Yellow
        # Yellow -> Green transition (2.3 ~ 2.7m)
        (2.4,  (0.7, 1.0, 0.1)),    # Yellow-green
        (2.5,  (0.5, 1.0, 0.2)),    # Green-yellow
        # Green zone (2.5 ~ 3.5m)
        (2.7,  (0.3, 1.0, 0.3)),    # Green
        (3.0,  (0.1, 1.0, 0.4)),    # Green
        (3.3,  (0.0, 1.0, 0.5)),    # Green-cyan
        # Green -> Cyan transition (3.3 ~ 3.7m)
        (3.5,  (0.0, 1.0, 0.7)),    # Cyan-green
        # Cyan zone (3.5 ~ 7m)
        (3.8,  (0.0, 1.0, 0.85)),   # Cyan
        (4.5,  (0.0, 1.0, 1.0)),    # Pure cyan
        (5.5,  (0.0, 0.9, 1.0)),    # Cyan
        (6.5,  (0.0, 0.7, 1.0)),    # Cyan-blue
        # Cyan -> Blue transition (6.5 ~ 7.5m)
        (7.0,  (0.0, 0.5, 1.0)),    # Blue-cyan
        # Blue zone (7 ~ 15m)
        (8.0,  (0.0, 0.3, 1.0)),    # Blue
        (10.0, (0.0, 0.15, 1.0)),   # Blue
        (12.0, (0.0, 0.05, 1.0)),   # Deep blue
        (15.0, (0.0, 0.0, 1.0)),    # Pure blue
    ]
    
    positions = [(d - min_depth) / depth_range for d, c in depth_points]
    colors = [c for d, c in depth_points]
    
    custom_cmap = LinearSegmentedColormap.from_list('depth_custom', 
                                                     list(zip(positions, colors)), 
                                                     N=512)
    return custom_cmap


def load_model(checkpoint_path: str):
    """Load model from checkpoint"""
    print(f"[INFO] Loading model: {checkpoint_path}")
    
    config, state_dict = parse_test_file(checkpoint_path, None)
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_state_dict(state_dict, strict=False)
    model_wrapper = model_wrapper.to('cuda')
    model_wrapper.eval()
    
    # Get max_depth from config
    max_depth = getattr(config.model.params, 'max_depth', 15.0)
    print(f"[INFO] Model loaded. Config max_depth={max_depth}m")
    
    return model_wrapper, max_depth


def depth_to_color(depth: np.ndarray, min_d: float, max_d: float, cmap) -> np.ndarray:
    """Convert depth map to color image"""
    normalized = np.clip((depth - min_d) / (max_d - min_d), 0, 1)
    colored = cmap(normalized)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def apply_mask_with_spacing(mask: np.ndarray, spacing: int = 4) -> np.ndarray:
    """
    Apply uniform grid spacing to a binary mask.
    Only keep pixels at regular grid intervals (no checkerboard pattern).
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        spacing: Grid spacing in pixels (e.g., spacing=4 means every 4th pixel)
    
    Returns:
        Sparse mask with uniform grid pattern
    """
    H, W = mask.shape
    sparse_mask = np.zeros_like(mask)
    
    # Create uniform grid pattern (not checkerboard)
    # Sample every 'spacing' pixels in both x and y directions
    y_indices = np.arange(0, H, spacing)
    x_indices = np.arange(0, W, spacing)
    
    # Use meshgrid to get all grid points
    yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
    
    # Apply mask values at grid points
    sparse_mask[yy, xx] = mask[yy, xx]
    
    return sparse_mask


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
    bg_mode: str = 'rgb'  # 'rgb', 'black', 'gray'
) -> plt.Figure:
    """
    Visualize prediction with mask applied.
    
    Args:
        rgb: RGB image (H, W, 3)
        pred_depth: Predicted depth (H, W)
        mask: Binary mask (H, W), values 0 or 1
        min_depth, max_depth: Depth range
        point_size: Point size for drawing
        cmap: Colormap
        sample_name: Sample name for title
        alpha: Point opacity (1.0 = fully opaque, 0.0 = transparent)
        bg_mode: Background mode - 'rgb' (original), 'black', 'gray'
    
    Returns:
        matplotlib figure
    """
    H, W = pred_depth.shape
    
    # Get valid mask coordinates
    valid_coords = np.where(mask > 0)
    num_points = len(valid_coords[0])
    
    # Create background based on mode
    if bg_mode == 'black':
        result = np.zeros((H, W, 3), dtype=np.uint8)
    elif bg_mode == 'gray':
        result = np.full((H, W, 3), 40, dtype=np.uint8)  # Dark gray
    else:  # rgb
        result = rgb.copy()
    
    # Draw points at mask locations (directly, no blending for LiDAR look)
    for y, x in zip(valid_coords[0], valid_coords[1]):
        depth_val = pred_depth[y, x]
        if depth_val > 0:
            # Normalize and get color
            norm_val = np.clip((depth_val - min_depth) / (max_depth - min_depth), 0, 1)
            color = cmap(norm_val)[:3]
            color = tuple(int(c * 255) for c in color)
            
            if point_size == 0:
                # Single pixel (fastest, most sparse look)
                result[y, x] = color
            else:
                # Circle with given radius
                cv2.circle(result, (x, y), point_size, color, -1, cv2.LINE_AA)
    
    # If alpha < 1.0 and bg_mode is 'rgb', blend with original
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
    
    # Create figure with extra space for colorbar
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
    
    # 3. Prediction with mask
    axes[2].imshow(result)
    axes[2].set_title(
        f'Pred (masked): mean={stats["mean"]:.2f}m, range=[{stats["min"]:.2f}, {stats["max"]:.2f}]m',
        fontsize=12
    )
    axes[2].axis('off')
    
    # Add colorbar with proper spacing
    plt.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_depth, vmax=max_depth))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Depth (m)', fontsize=11)
    
    return fig, stats


def find_sample_pairs(image_dir: Path, depth_subdir: str = 'newest_original_depth_maps'):
    """Find RGB-Depth pairs"""
    rgb_dir = image_dir / 'image_a6'
    depth_dir = image_dir / depth_subdir
    
    pairs = []
    for rgb_file in sorted(rgb_dir.glob('*.jpg')):
        depth_file = depth_dir / rgb_file.name.replace('.jpg', '.png')
        if depth_file.exists():
            pairs.append((rgb_file, depth_file))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Visualize depth prediction with binary mask')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing image_a6 and depth subdirs')
    parser.add_argument('--mask_path', type=str, default='assets/binary_bw_640x384_mask.png',
                        help='Path to binary mask')
    parser.add_argument('--output_dir', type=str, default='outputs/masked_prediction',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process')
    parser.add_argument('--random', action='store_true',
                        help='Random sample selection')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (None = true random each run)')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Minimum depth')
    parser.add_argument('--max_depth', type=float, default=15.0,
                        help='Maximum depth')
    parser.add_argument('--point_size', type=int, default=2,
                        help='Point size for visualization')
    parser.add_argument('--spacing', type=int, default=4,
                        help='Grid spacing for mask (0 = use original mask)')
    parser.add_argument('--colormap', type=str, default='turbo',
                        choices=['turbo', 'turbo_r', 'jet', 'jet_r', 'viridis', 'viridis_r', 'plasma', 'plasma_r', 'custom'],
                        help='Colormap (add _r suffix for reversed)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Point opacity (1.0=opaque, 0.5=semi-transparent)')
    parser.add_argument('--bg_mode', type=str, default='rgb',
                        choices=['rgb', 'black', 'gray'],
                        help='Background mode: rgb (original image), black, gray')
    parser.add_argument('--depth_subdir', type=str, default='newest_original_depth_maps',
                        help='Depth subdirectory name')
    
    args = parser.parse_args()
    
    # Load mask
    mask_path = Path(args.mask_path)
    if not mask_path.exists():
        print(f"[ERROR] Mask not found: {mask_path}")
        return
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    print(f"[INFO] Loaded mask: {mask.shape}, unique values: {np.unique(mask)}")
    
    # Apply spacing if requested
    if args.spacing > 0:
        original_points = np.sum(mask > 0)
        mask = apply_mask_with_spacing(mask, args.spacing)
        sparse_points = np.sum(mask > 0)
        print(f"[INFO] Applied spacing={args.spacing}: {original_points:,} -> {sparse_points:,} points ({sparse_points/mask.size*100:.2f}%)")
    
    # Load model
    model_wrapper, model_max_depth = load_model(args.checkpoint)
    max_depth = args.max_depth if args.max_depth else model_max_depth
    
    # Find samples
    image_dir = Path(args.image_dir)
    pairs = find_sample_pairs(image_dir, args.depth_subdir)
    print(f"[INFO] Found {len(pairs)} RGB-Depth pairs")
    
    if args.random:
        # Use current time as seed for true randomness (or specify --seed)
        seed = args.seed if hasattr(args, 'seed') and args.seed is not None else None
        if seed is not None:
            random.seed(seed)
            print(f"[INFO] Using random seed: {seed}")
        else:
            random.seed()  # True random
            print(f"[INFO] Using random selection (no fixed seed)")
        pairs = random.sample(pairs, min(args.num_samples, len(pairs)))
    else:
        pairs = pairs[:args.num_samples]
    
    # Setup output - clear existing files first
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        # Remove all existing png files
        existing_files = list(output_dir.glob('*.png'))
        if existing_files:
            print(f"[INFO] Clearing {len(existing_files)} existing files in {output_dir}")
            for f in existing_files:
                f.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Colormap
    if args.colormap == 'custom':
        cmap = create_custom_colormap(args.min_depth, max_depth)
    else:
        cmap = getattr(plt.cm, args.colormap)
    
    # Print config
    print(f"""
============================================================
MASKED PREDICTION VISUALIZATION
============================================================
Samples: {len(pairs)}
Depth range: {args.min_depth}m - {max_depth}m
Colormap: {args.colormap}
Point size: {args.point_size}
Grid spacing: {args.spacing}px
Mask points: {np.sum(mask > 0):,} ({np.sum(mask > 0)/mask.size*100:.2f}%)
Output: {output_dir}
============================================================
""")
    
    all_stats = []
    
    for i, (rgb_path, depth_path) in enumerate(pairs):
        sample_name = rgb_path.stem
        print(f"[{i+1}/{len(pairs)}] {rgb_path.name}")
        
        # Load RGB
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        H, W = mask.shape
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))
        
        # Inference - use model directly like visualize_depth.py
        rgb_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        
        with torch.no_grad():
            output = model_wrapper.model({'rgb': rgb_tensor})
            
            if ('integer', 0) in output and ('fractional', 0) in output:
                integer_sig = output[('integer', 0)].cpu().numpy()[0, 0]
                fractional_sig = output[('fractional', 0)].cpu().numpy()[0, 0]
                pred_depth = integer_sig * max_depth + fractional_sig
            else:
                inv_depth = output[('inv_depths', 0)][0].cpu().numpy()[0, 0]
                pred_depth = 1.0 / (inv_depth + 1e-8)
        
        # Resize prediction to match mask size if needed
        if pred_depth.shape != mask.shape:
            pred_depth = cv2.resize(pred_depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Visualize
        fig, stats = visualize_with_mask(
            rgb, pred_depth, mask,
            args.min_depth, max_depth,
            args.point_size, cmap, sample_name,
            args.alpha, args.bg_mode
        )
        
        # Save
        output_path = output_dir / f"{sample_name}_masked.png"
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
