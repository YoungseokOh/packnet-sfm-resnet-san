#!/usr/bin/env python3
"""
Unified Depth Visualization Tool

GT vs Prediction 깊이 시각화를 위한 통합 스크립트.
Sparse (LiDAR 스타일), Dense (colormap), Comparison (side-by-side) 모드 지원.

Modes:
    sparse: GT depth 유효 픽셀 위치 기반 sparse 마스크 (LiDAR 포인트 스타일)
    dense: 전체 픽셀 dense colormap 시각화
    comparison: GT vs Pred side-by-side 비교 (2x2 quad layout)

Usage Examples:
    # Sparse 모드 (LiDAR 스타일) - 20개 랜덤 샘플
    python scripts/visualization/visualize_depth.py \\
        --checkpoint checkpoints/.../epoch=49...ckpt \\
        --image_dir /workspace/data/ncdb-cls-640x384/.../640x384_newest \\
        --output_dir outputs/depth_viz \\
        --mode sparse --num_samples 20 --random \\
        --clip_percentile 15 --subsample_ratio 0.3

    # Dense 모드 - 전체 colormap
    python scripts/visualization/visualize_depth.py \\
        --checkpoint checkpoints/.../epoch=49...ckpt \\
        --image_dir /workspace/data/ncdb-cls-640x384/.../640x384_newest \\
        --output_dir outputs/depth_viz \\
        --mode dense --num_samples 10

    # Comparison 모드 - 2x2 quad 레이아웃
    python scripts/visualization/visualize_depth.py \\
        --checkpoint checkpoints/.../epoch=49...ckpt \\
        --image_dir /workspace/data/ncdb-cls-640x384/.../640x384_newest \\
        --output_dir outputs/depth_viz \\
        --mode comparison --num_samples 5

Author: YoungseokOh
Date: 2025-12-04
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ========== Colormap Utilities ==========

def get_colormap(name: str = 'turbo'):
    """Get matplotlib colormap by name"""
    if name == 'turbo':
        return plt.cm.turbo
    elif name == 'jet':
        return plt.cm.jet
    elif name == 'viridis':
        return plt.cm.viridis
    elif name == 'plasma':
        return plt.cm.plasma
    else:
        return plt.cm.turbo


def depth_to_color(depth: float, min_d: float, max_d: float, cmap) -> Tuple[int, int, int]:
    """Convert single depth value to RGB color"""
    normalized = np.clip((depth - min_d) / (max_d - min_d), 0, 1)
    rgba = cmap(normalized)
    return tuple(int(c * 255) for c in rgba[:3])


def depth_to_colormap(depth: np.ndarray, min_d: float, max_d: float, cmap) -> np.ndarray:
    """Convert depth map to RGB colormap image"""
    normalized = np.clip((depth - min_d) / (max_d - min_d), 0, 1)
    colored = cmap(normalized)[:, :, :3]
    return (colored * 255).astype(np.uint8)


# ========== Sparse Mask Generation ==========

def create_sparse_mask(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    clip_percentile: float = 15.0,
    subsample_ratio: float = 0.3,
    cmap=None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """
    Create sparse mask based on GT depth valid pixel locations.
    
    Args:
        gt_depth: GT depth map (H, W)
        pred_depth: Prediction depth map (H, W)
        min_depth: Minimum depth
        max_depth: Maximum depth
        clip_percentile: Clip top/bottom percentile (0-50)
        subsample_ratio: Ratio of GT points to keep (0-1)
        cmap: Colormap to use
        seed: Random seed
    
    Returns:
        valid_mask, gt_colors, pred_colors, gt_stats, pred_stats
    """
    H, W = gt_depth.shape
    if cmap is None:
        cmap = plt.cm.turbo
    
    # 1. Find valid GT depth pixels
    valid_gt = (gt_depth > 0) & (gt_depth < max_depth * 2)
    
    # 2. Apply percentile clipping
    valid_depths = gt_depth[valid_gt]
    if len(valid_depths) > 0 and clip_percentile > 0:
        low_clip = np.percentile(valid_depths, clip_percentile)
        high_clip = np.percentile(valid_depths, 100 - clip_percentile)
        effective_min = max(min_depth, low_clip)
        effective_max = min(max_depth, high_clip)
    else:
        effective_min = min_depth
        effective_max = max_depth
    
    # 3. Update valid mask with clipping
    valid_mask = valid_gt & (gt_depth >= effective_min) & (gt_depth <= effective_max)
    
    # 4. Subsample if requested
    if subsample_ratio < 1.0:
        np.random.seed(seed)
        valid_coords = np.where(valid_mask)
        num_valid = len(valid_coords[0])
        num_keep = int(num_valid * subsample_ratio)
        
        if num_keep < num_valid:
            indices = np.random.choice(num_valid, num_keep, replace=False)
            subsample_mask = np.zeros((H, W), dtype=bool)
            subsample_mask[valid_coords[0][indices], valid_coords[1][indices]] = True
            valid_mask = subsample_mask
    
    # 5. Generate colors
    gt_colors = np.zeros((H, W, 3), dtype=np.uint8)
    pred_colors = np.zeros((H, W, 3), dtype=np.uint8)
    
    coords = np.where(valid_mask)
    for y, x in zip(coords[0], coords[1]):
        gt_colors[y, x] = depth_to_color(gt_depth[y, x], effective_min, effective_max, cmap)
        pred_colors[y, x] = depth_to_color(pred_depth[y, x], effective_min, effective_max, cmap)
    
    # 6. Statistics
    gt_vals = gt_depth[valid_mask]
    pred_vals = pred_depth[valid_mask]
    
    gt_stats = {
        'num_points': len(gt_vals),
        'mean': gt_vals.mean() if len(gt_vals) > 0 else 0,
        'std': gt_vals.std() if len(gt_vals) > 0 else 0,
        'min': gt_vals.min() if len(gt_vals) > 0 else 0,
        'max': gt_vals.max() if len(gt_vals) > 0 else 0,
        'effective_min': effective_min,
        'effective_max': effective_max
    }
    
    pred_stats = {
        'num_points': len(pred_vals),
        'mean': pred_vals.mean() if len(pred_vals) > 0 else 0,
        'std': pred_vals.std() if len(pred_vals) > 0 else 0,
        'min': pred_vals.min() if len(pred_vals) > 0 else 0,
        'max': pred_vals.max() if len(pred_vals) > 0 else 0
    }
    
    return valid_mask, gt_colors, pred_colors, gt_stats, pred_stats


def draw_sparse_points(
    rgb: np.ndarray, 
    colors: np.ndarray, 
    valid_mask: np.ndarray,
    point_size: int = 3,
    alpha: float = 0.9
) -> np.ndarray:
    """Draw sparse depth points on RGB image"""
    result = rgb.copy()
    overlay = result.copy()
    
    coords = np.where(valid_mask)
    for y, x in zip(coords[0], coords[1]):
        color = tuple(int(c) for c in colors[y, x])
        cv2.circle(overlay, (x, y), point_size, color, -1, cv2.LINE_AA)
    
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    return result


# ========== Visualization Modes ==========

def create_pred_grid_mask(
    pred_depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    top_margin: float = 0.15,
    bottom_margin: float = 0.15,
    grid_spacing: int = 4,
    cmap=None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create grid-sampled mask for prediction in middle region of image.
    
    Args:
        pred_depth: Prediction depth map (H, W)
        min_depth, max_depth: Depth range
        top_margin: Top margin ratio to exclude (0.15 = 15%)
        bottom_margin: Bottom margin ratio to exclude
        grid_spacing: Pixel spacing for grid sampling
        cmap: Colormap
    
    Returns:
        pred_mask: Boolean mask
        pred_colors: RGB colors (H, W, 3)
        pred_stats: Statistics
    """
    H, W = pred_depth.shape
    if cmap is None:
        cmap = plt.cm.turbo
    
    # Calculate vertical bounds (exclude top/bottom margins)
    top_row = int(H * top_margin)
    bottom_row = int(H * (1 - bottom_margin))
    
    # Create grid mask in middle region
    pred_mask = np.zeros((H, W), dtype=bool)
    pred_mask[top_row:bottom_row:grid_spacing, ::grid_spacing] = True
    
    # Filter by valid depth range
    valid_depth = (pred_depth >= min_depth) & (pred_depth <= max_depth)
    pred_mask = pred_mask & valid_depth
    
    # Generate colors
    pred_colors = np.zeros((H, W, 3), dtype=np.uint8)
    coords = np.where(pred_mask)
    for y, x in zip(coords[0], coords[1]):
        pred_colors[y, x] = depth_to_color(pred_depth[y, x], min_depth, max_depth, cmap)
    
    # Statistics
    pred_vals = pred_depth[pred_mask]
    pred_stats = {
        'num_points': len(pred_vals),
        'mean': pred_vals.mean() if len(pred_vals) > 0 else 0,
        'std': pred_vals.std() if len(pred_vals) > 0 else 0,
        'min': pred_vals.min() if len(pred_vals) > 0 else 0,
        'max': pred_vals.max() if len(pred_vals) > 0 else 0
    }
    
    return pred_mask, pred_colors, pred_stats


def visualize_sparse(
    rgb: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    clip_percentile: float,
    subsample_ratio: float,
    point_size: int,
    cmap,
    sample_name: str,
    pred_grid_spacing: int = 4,
    top_margin: float = 0.15,
    bottom_margin: float = 0.15
) -> plt.Figure:
    """Sparse mode: LiDAR 스타일 포인트 시각화
    
    - GT: LiDAR 포인트 위치에서 sparse 샘플링
    - Pred: 이미지 중간 영역(top/bottom margin 제외)에서 grid 샘플링
    """
    
    # GT: LiDAR sparse mask (기존 방식)
    valid_mask, gt_colors, _, gt_stats, _ = create_sparse_mask(
        gt_depth, pred_depth, min_depth, max_depth, clip_percentile, subsample_ratio, cmap
    )
    
    # Pred: Grid sampling in middle region (새 방식)
    eff_min = gt_stats['effective_min']
    eff_max = gt_stats['effective_max']
    pred_mask, pred_colors, pred_stats = create_pred_grid_mask(
        pred_depth, eff_min, eff_max, top_margin, bottom_margin, pred_grid_spacing, cmap
    )
    
    # Draw points - GT는 작은 포인트 (size=1), Pred는 약간 큰 포인트 (size=2)
    gt_sparse = draw_sparse_points(rgb, gt_colors, valid_mask, point_size=1, alpha=0.95)
    pred_sparse = draw_sparse_points(rgb, pred_colors, pred_mask, point_size=2, alpha=0.9)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12), facecolor='#1a1a1a')
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05],
                          hspace=0.12, wspace=0.08,
                          left=0.02, right=0.95, top=0.92, bottom=0.05)
    
    # Top-Left: GT Sparse (LiDAR points)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gt_sparse)
    title1 = f'GT Depth (LiDAR Sparse)\n[{subsample_ratio*100:.0f}% subsample]'
    ax1.set_title(title1, fontsize=12, color='white', fontweight='bold')
    ax1.axis('off')
    stats_text = f"Points: {gt_stats['num_points']:,}\nMean: {gt_stats['mean']:.2f}m"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, color='white', va='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Top-Right: Pred Sparse (Grid sampling in middle region)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred_sparse)
    ax2.set_title(f'Prediction (Grid {pred_grid_spacing}px, mid {int((1-top_margin-bottom_margin)*100)}%)', 
                  fontsize=12, color='white', fontweight='bold')
    ax2.axis('off')
    stats_text = f"Points: {pred_stats['num_points']:,}\nMean: {pred_stats['mean']:.2f}m"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10, color='white', va='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Bottom-Left: GT Dense
    ax3 = fig.add_subplot(gs[1, 0])
    gt_vis = np.clip(gt_depth, eff_min, eff_max)
    gt_vis[gt_depth <= 0] = np.nan
    ax3.imshow(gt_vis, cmap=cmap, vmin=eff_min, vmax=eff_max)
    ax3.set_title('GT Depth (Dense)', fontsize=12, color='white', fontweight='bold')
    ax3.axis('off')
    
    # Bottom-Right: Pred Dense
    ax4 = fig.add_subplot(gs[1, 1])
    im = ax4.imshow(pred_depth, cmap=cmap, vmin=eff_min, vmax=eff_max)
    ax4.set_title('Prediction (Dense)', fontsize=12, color='white', fontweight='bold')
    ax4.axis('off')
    
    # Colorbar
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Depth (m)', fontsize=11, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Title with error
    error = abs(gt_stats['mean'] - pred_stats['mean'])
    error_pct = error / gt_stats['mean'] * 100 if gt_stats['mean'] > 0 else 0
    fig.suptitle(f'{sample_name}\nMean Error: {error:.2f}m ({error_pct:.1f}%)',
                 fontsize=14, color='white', fontweight='bold')
    
    return fig


def visualize_dense(
    rgb: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    cmap,
    sample_name: str
) -> plt.Figure:
    """Dense mode: 전체 픽셀 colormap 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#1a1a1a')
    
    # Compute stats
    valid_mask = (gt_depth > min_depth) & (gt_depth < max_depth)
    gt_mean = gt_depth[valid_mask].mean() if valid_mask.sum() > 0 else 0
    pred_mean = pred_depth[valid_mask].mean() if valid_mask.sum() > 0 else 0
    error_map = np.abs(gt_depth - pred_depth) * valid_mask
    mae = error_map[valid_mask].mean() if valid_mask.sum() > 0 else 0
    
    # Top-Left: RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Input', color='white', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Top-Right: GT Depth
    gt_vis = np.clip(gt_depth, min_depth, max_depth)
    gt_vis[gt_depth <= 0] = np.nan
    im1 = axes[0, 1].imshow(gt_vis, cmap=cmap, vmin=min_depth, vmax=max_depth)
    axes[0, 1].set_title(f'GT Depth (Mean: {gt_mean:.2f}m)', color='white', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom-Left: Pred Depth
    axes[1, 0].imshow(pred_depth, cmap=cmap, vmin=min_depth, vmax=max_depth)
    axes[1, 0].set_title(f'Pred Depth (Mean: {pred_mean:.2f}m)', color='white', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Bottom-Right: Error Map
    im2 = axes[1, 1].imshow(error_map, cmap='hot', vmin=0, vmax=3)
    axes[1, 1].set_title(f'Error Map (MAE: {mae:.2f}m)', color='white', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Colorbars
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label='Depth (m)')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Error (m)')
    
    for ax in axes.flat:
        ax.set_facecolor('#1a1a1a')
    
    fig.suptitle(f'{sample_name}', fontsize=15, color='white', fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_comparison(
    rgb: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    cmap,
    sample_name: str
) -> plt.Figure:
    """Comparison mode: GT vs Pred side-by-side (blended on RGB)"""
    
    H, W = rgb.shape[:2]
    
    # Compute stats
    valid_mask = (gt_depth > min_depth) & (gt_depth < max_depth)
    gt_mean = gt_depth[valid_mask].mean() if valid_mask.sum() > 0 else 0
    pred_mean = pred_depth[valid_mask].mean() if valid_mask.sum() > 0 else 0
    
    # Create blended images
    gt_colored = depth_to_colormap(gt_depth, min_depth, max_depth, cmap)
    pred_colored = depth_to_colormap(pred_depth, min_depth, max_depth, cmap)
    
    alpha = 0.6
    gt_blend = (rgb * (1 - alpha) + gt_colored * alpha).astype(np.uint8)
    pred_blend = (rgb * (1 - alpha) + pred_colored * alpha).astype(np.uint8)
    
    # Mask invalid GT regions
    invalid_gt = gt_depth <= 0
    gt_blend[invalid_gt] = rgb[invalid_gt]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#1a1a1a')
    
    axes[0].imshow(gt_blend)
    axes[0].set_title(f'GT Depth Overlay (Mean: {gt_mean:.2f}m)', color='white', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(pred_blend)
    axes[1].set_title(f'Pred Depth Overlay (Mean: {pred_mean:.2f}m)', color='white', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_depth, vmax=max_depth))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar.set_label('Depth (m)', fontsize=11, color='white')
    cbar.ax.xaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')
    
    error = abs(gt_mean - pred_mean)
    error_pct = error / gt_mean * 100 if gt_mean > 0 else 0
    fig.suptitle(f'{sample_name} | Error: {error:.2f}m ({error_pct:.1f}%)',
                 fontsize=14, color='white', fontweight='bold')
    
    plt.tight_layout()
    return fig


# ========== Model & Data Loading ==========

def load_model(checkpoint_path: str):
    """Load trained Dual-Head model"""
    from packnet_sfm.models.model_wrapper import ModelWrapper
    from packnet_sfm.utils.config import parse_test_file
    
    print(f"[INFO] Loading model: {checkpoint_path}")
    config, state_dict = parse_test_file(checkpoint_path, None)
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_state_dict(state_dict, strict=False)
    model_wrapper = model_wrapper.to('cuda')
    model_wrapper.eval()
    
    max_depth = float(config.model.params.max_depth)
    print(f"[INFO] Model loaded. Config max_depth={max_depth}m")
    
    return model_wrapper, max_depth


def run_inference(model_wrapper, image_path: Path, max_depth: float) -> np.ndarray:
    """Run depth inference on single image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 384), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()
    
    with torch.no_grad():
        output = model_wrapper.model({'rgb': img_tensor})
        
        if ('integer', 0) in output and ('fractional', 0) in output:
            integer_sig = output[('integer', 0)].cpu().numpy()[0, 0]
            fractional_sig = output[('fractional', 0)].cpu().numpy()[0, 0]
            depth = integer_sig * max_depth + fractional_sig
        else:
            inv_depth = output[('inv_depths', 0)][0].cpu().numpy()[0, 0]
            depth = 1.0 / (inv_depth + 1e-8)
    
    return depth


def load_gt_depth(gt_path: Path) -> np.ndarray:
    """Load GT depth (16-bit PNG, value/256 = meters)"""
    gt_img = Image.open(gt_path)
    return np.array(gt_img, dtype=np.float32) / 256.0


def load_rgb_image(rgb_path: Path) -> np.ndarray:
    """Load RGB image"""
    rgb = Image.open(rgb_path).convert('RGB')
    return np.array(rgb.resize((640, 384), Image.BILINEAR))


def find_sample_pairs(image_dir: Path, depth_subdir: str = 'newest_original_depth_maps'):
    """Find RGB-Depth pairs"""
    possible_rgb_dirs = ['image_a6', 'images', 'image', 'rgb']
    rgb_dir = None
    for subdir in possible_rgb_dirs:
        if (image_dir / subdir).exists():
            rgb_dir = image_dir / subdir
            break
    if not rgb_dir:
        print(f"[ERROR] RGB directory not found in {image_dir}")
        return []
    
    depth_dir = image_dir / depth_subdir
    if not depth_dir.exists():
        print(f"[ERROR] Depth directory not found: {depth_dir}")
        return []
    
    print(f"[INFO] RGB dir: {rgb_dir}")
    print(f"[INFO] Depth dir: {depth_dir}")
    
    pairs = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for rgb_file in sorted(rgb_dir.glob(ext)):
            for depth_ext in ['.png', '.npy']:
                depth_file = depth_dir / f'{rgb_file.stem}{depth_ext}'
                if depth_file.exists():
                    pairs.append((rgb_file, depth_file))
                    break
    
    print(f"[INFO] Found {len(pairs)} RGB-Depth pairs")
    return pairs


def find_rgb_pred_pairs(rgb_dir: Path, pred_depth_dir: Path) -> List[Tuple[Path, Path]]:
    """Find RGB files and their corresponding predicted depth PNGs.

    Expected predicted depth filenames: depth_<stem>.png where <stem> is RGB stem.
    Example: RGB 0000000002.jpg -> depth_0000000002.png
    """
    if not rgb_dir.exists():
        print(f"[ERROR] RGB directory not found: {rgb_dir}")
        return []
    if not pred_depth_dir.exists():
        print(f"[ERROR] Pred depth directory not found: {pred_depth_dir}")
        return []

    pairs: List[Tuple[Path, Path]] = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for rgb_file in sorted(rgb_dir.glob(ext)):
            pred_depth = pred_depth_dir / f"depth_{rgb_file.stem}.png"
            if pred_depth.exists():
                pairs.append((rgb_file, pred_depth))
    print(f"[INFO] Found {len(pairs)} RGB-PredDepth pairs")
    return pairs


def load_pred_depth_png(pred_path: Path) -> np.ndarray:
    """Load predicted depth from 16-bit PNG.

    Convention in this repo: uint16 PNG saved in (meters * 256).
    """
    img = np.array(Image.open(pred_path), dtype=np.uint16)
    return img.astype(np.float32) / 256.0


def visualize_pred_only(
    rgb: np.ndarray,
    pred_depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    cmap,
    sample_name: str
) -> plt.Figure:
    """Prediction-only visualization (no GT needed)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#1a1a1a')

    axes[0].imshow(rgb)
    axes[0].set_title('RGB Input', color='white', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(pred_depth, cmap=cmap, vmin=min_depth, vmax=max_depth)
    axes[1].set_title('Pred Depth (Dense)', color='white', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Depth (m)')

    for ax in axes.flat:
        ax.set_facecolor('#1a1a1a')

    fig.suptitle(f'{sample_name}', fontsize=15, color='white', fontweight='bold')
    plt.tight_layout()
    return fig


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(
        description='Unified Depth Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sparse mode (LiDAR style)
  python visualize_depth.py --checkpoint ckpt --image_dir data --mode sparse --num_samples 20 --random
  
  # Dense mode (full colormap)
  python visualize_depth.py --checkpoint ckpt --image_dir data --mode dense --num_samples 10
  
  # Comparison mode (side-by-side overlay)
  python visualize_depth.py --checkpoint ckpt --image_dir data --mode comparison --num_samples 5
        """
    )
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory with RGB images and GT depth')
    parser.add_argument('--output_dir', type=str, default='outputs/depth_viz',
                        help='Output directory')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='sparse',
                        choices=['sparse', 'dense', 'comparison'],
                        help='Visualization mode: sparse (LiDAR), dense (colormap), comparison (overlay)')
    
    # Sampling
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to process (0 = all)')
    parser.add_argument('--random', action='store_true',
                        help='Randomly select samples')
    
    # Depth parameters
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help='Minimum depth (m)')
    parser.add_argument('--max_depth', type=float, default=15.0,
                        help='Maximum depth (m)')
    
    # Sparse mode specific
    parser.add_argument('--clip_percentile', type=float, default=15.0,
                        help='[Sparse] Clip top/bottom percentile (0-50)')
    parser.add_argument('--subsample_ratio', type=float, default=0.3,
                        help='[Sparse] Ratio of GT points to keep (0-1)')
    parser.add_argument('--point_size', type=int, default=3,
                        help='[Sparse] Point size for visualization')
    parser.add_argument('--pred_grid_spacing', type=int, default=4,
                        help='[Sparse] Pred grid sampling spacing (pixels)')
    parser.add_argument('--top_margin', type=float, default=0.15,
                        help='[Sparse] Top margin ratio to exclude (0.15 = 15%%)')
    parser.add_argument('--bottom_margin', type=float, default=0.15,
                        help='[Sparse] Bottom margin ratio to exclude')
    
    # Visualization options
    parser.add_argument('--colormap', type=str, default='turbo',
                        choices=['turbo', 'jet', 'viridis', 'plasma'],
                        help='Colormap to use')
    parser.add_argument('--depth_subdir', type=str, default='newest_original_depth_maps',
                        help='GT depth subdirectory name')

    # Pred-only mode (use existing predicted depth PNGs)
    parser.add_argument('--pred_depth_dir', type=str, default=None,
                        help='If set, skip GT lookup/inference and visualize existing predicted depth PNGs in this directory (expects depth_<stem>.png)')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If pred_depth_dir is provided, do pred-only visualization (no model, no GT)
    if args.pred_depth_dir is not None:
        rgb_dir = Path(args.image_dir) / 'images'
        pred_depth_dir = Path(args.pred_depth_dir)
        pairs = find_rgb_pred_pairs(rgb_dir, pred_depth_dir)
        if not pairs:
            print("[ERROR] No RGB-PredDepth pairs found!")
            return

        model_wrapper = None
        model_max_depth = None
        max_depth = args.max_depth
        print(f"[INFO] Pred-only mode. Using max_depth={max_depth}m")
    else:
        # Load model
        model_wrapper, model_max_depth = load_model(args.checkpoint)
        max_depth = args.max_depth
        print(f"[INFO] Using max_depth={max_depth}m")

        # Find samples (needs GT)
        pairs = find_sample_pairs(Path(args.image_dir), args.depth_subdir)
        if not pairs:
            print("[ERROR] No samples found!")
            return
    
    # Select samples
    if args.num_samples > 0:
        if args.random:
            pairs = random.sample(pairs, min(args.num_samples, len(pairs)))
        else:
            pairs = pairs[:args.num_samples]
    
    # Get colormap
    cmap = get_colormap(args.colormap)
    
    # Print config
    print(f"\n{'='*60}")
    print(f"DEPTH VISUALIZATION")
    print(f"{'='*60}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Samples: {len(pairs)}")
    print(f"Depth range: {args.min_depth}m - {max_depth}m")
    print(f"Colormap: {args.colormap}")
    if args.mode == 'sparse':
        print(f"Clip percentile: {args.clip_percentile}%")
        print(f"Subsample ratio: {args.subsample_ratio*100:.0f}%")
        print(f"Pred grid spacing: {args.pred_grid_spacing}px")
        print(f"Top/Bottom margin: {args.top_margin*100:.0f}% / {args.bottom_margin*100:.0f}%")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Process samples
    all_gt_means = []
    all_pred_means = []
    all_errors = []
    
    for idx, (rgb_path, second_path) in enumerate(pairs):
        print(f"[{idx+1}/{len(pairs)}] {rgb_path.name}")

        # Load RGB
        rgb = load_rgb_image(rgb_path)

        if args.pred_depth_dir is not None:
            # Pred-only: second_path is predicted depth png
            pred_depth = load_pred_depth_png(second_path)
            gt_depth = None
        else:
            # GT-based: second_path is GT depth
            gt_depth = load_gt_depth(second_path)
            if gt_depth.shape != (384, 640):
                gt_depth = cv2.resize(gt_depth, (640, 384), interpolation=cv2.INTER_NEAREST)

            # Inference
            pred_depth = run_inference(model_wrapper, rgb_path, max_depth)
        
        sample_name = rgb_path.stem
        
        # Create visualization based on mode
        if args.pred_depth_dir is not None:
            # Pred-only path: force dense-like output
            fig = visualize_pred_only(
                rgb, pred_depth,
                args.min_depth, max_depth,
                cmap, sample_name
            )
            suffix = 'pred_only'
        else:
            if args.mode == 'sparse':
                fig = visualize_sparse(
                    rgb, gt_depth, pred_depth,
                    args.min_depth, max_depth,
                    args.clip_percentile, args.subsample_ratio,
                    args.point_size, cmap, sample_name,
                    args.pred_grid_spacing, args.top_margin, args.bottom_margin
                )
                suffix = 'sparse'
            elif args.mode == 'dense':
                fig = visualize_dense(
                    rgb, gt_depth, pred_depth,
                    args.min_depth, max_depth,
                    cmap, sample_name
                )
                suffix = 'dense'
            else:  # comparison
                fig = visualize_comparison(
                    rgb, gt_depth, pred_depth,
                    args.min_depth, max_depth,
                    cmap, sample_name
                )
                suffix = 'comparison'
        
        # Save
        output_path = output_dir / f'{sample_name}_{suffix}.png'
        fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        
        # Collect stats when GT exists
        if gt_depth is not None:
            valid_mask = (gt_depth > args.min_depth) & (gt_depth < max_depth)
            if valid_mask.sum() > 0:
                gt_mean = gt_depth[valid_mask].mean()
                pred_mean = pred_depth[valid_mask].mean()
                all_gt_means.append(gt_mean)
                all_pred_means.append(pred_mean)
                all_errors.append(abs(gt_mean - pred_mean))
        
        print(f"    Saved: {output_path.name}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Samples processed: {len(pairs)}")
    print(f"Output directory: {output_dir}")
    if all_gt_means:
        avg_gt = np.mean(all_gt_means)
        avg_pred = np.mean(all_pred_means)
        avg_error = np.mean(all_errors)
        print(f"Average GT depth: {avg_gt:.2f}m")
        print(f"Average Pred depth: {avg_pred:.2f}m")
        print(f"Average error: {avg_error:.2f}m ({avg_error/avg_gt*100:.1f}%)")
    else:
        print("GT not provided (pred-only mode): skipping GT/pred summary statistics")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
