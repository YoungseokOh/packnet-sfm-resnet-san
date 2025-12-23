#!/usr/bin/env python3
"""Visualize depth predictions vs GT for video frames (no JSON needed).

This script directly iterates over image files in a folder, matching them
with corresponding GT depth maps and predicted depth maps by filename stem.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packnet_sfm.visualization.colormaps import create_custom_depth_colormap


def load_depth_png_meters(path: Path) -> np.ndarray:
    """Load NCDB depth PNG (uint16 with value = meters*256)."""
    arr = np.asarray(Image.open(path))
    arr_f = arr.astype(np.float32)
    if arr_f.max() > 255:
        arr_f /= 256.0
    return arr_f


def load_pred_depth_png_meters(path: Path) -> np.ndarray:
    arr16 = np.asarray(Image.open(path), dtype=np.uint16)
    return arr16.astype(np.float32) / 256.0


def _get_cmap(cmap: str, vmin: float, vmax: float):
    if cmap == 'custom':
        return create_custom_depth_colormap(vmin, vmax)
    return plt.get_cmap(cmap)


def colorize_points(values: np.ndarray, vmin: float, vmax: float, cmap: str = 'custom') -> np.ndarray:
    norm = np.clip((values - vmin) / max(1e-6, (vmax - vmin)), 0, 1)
    cm = _get_cmap(cmap, vmin, vmax)
    return (cm(norm)[:, :3] * 255).astype(np.uint8)


def draw_points(rgb: np.ndarray, uv: np.ndarray, colors: np.ndarray, radius: int = 2) -> np.ndarray:
    out = rgb.copy()
    for (u, v), c in zip(uv, colors):
        cv2.circle(out, (int(u), int(v)), radius, (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
    return out


def put_panel_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_uniform_grid_uv(H: int, W: int, step: int) -> np.ndarray:
    step = max(1, int(step))
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb_dir', type=str, required=True, help='RGB images folder')
    ap.add_argument('--gt_depth_dir', type=str, required=True, help='GT depth maps folder')
    ap.add_argument('--pred_depth_dir', type=str, required=True, help='Predicted depth maps folder')
    ap.add_argument('--out_dir', type=str, required=True, help='Output folder for quad images')
    ap.add_argument('--pred_pattern', type=str, default='depth_{stem}.png',
                    help="Pred depth filename pattern (supports '{stem}')")
    ap.add_argument('--image_shape', nargs=2, type=int, default=[384, 640], help='H W')
    ap.add_argument('--min_depth', type=float, default=0.1)
    ap.add_argument('--max_depth', type=float, default=10.0)
    ap.add_argument('--mask_path', type=str, default=None, help='Binary mask (white=valid)')
    ap.add_argument('--point_radius', type=int, default=1)
    ap.add_argument('--dense_step', type=int, default=4, help='Grid step for masked pred panel')
    ap.add_argument('--max_points', type=int, default=30000, help='Max points for overlay')
    ap.add_argument('--num_samples', type=int, default=0, help='0 = all')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    H, W = args.image_shape
    rgb_dir = Path(args.rgb_dir)
    gt_dir = Path(args.gt_depth_dir)
    pred_dir = Path(args.pred_depth_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mask
    dense_mask = None
    if args.mask_path:
        mask_path = Path(args.mask_path)
        if mask_path.exists():
            dense_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if dense_mask.shape != (H, W):
                dense_mask = cv2.resize(dense_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            print(f'[INFO] Loaded mask: {mask_path}, valid_pixels={np.sum(dense_mask > 0)}')

    # Find all RGB images
    rgb_files = sorted(list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png')))
    print(f'[INFO] Found {len(rgb_files)} RGB images')

    if args.num_samples > 0 and args.num_samples < len(rgb_files):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(rgb_files), size=args.num_samples, replace=False)
        rgb_files = [rgb_files[i] for i in sorted(idx)]

    total_saved = 0
    cmap_name = 'custom'

    for rgb_path in rgb_files:
        stem = rgb_path.stem

        # Find GT depth
        gt_path = gt_dir / f'{stem}.png'
        if not gt_path.exists():
            continue

        # Find pred depth
        pred_pattern = args.pred_pattern.format(stem=stem)
        pred_path = pred_dir / pred_pattern
        if not pred_path.exists():
            continue

        # Load images
        rgb = np.asarray(Image.open(rgb_path).convert('RGB').resize((W, H), Image.BILINEAR))
        gt_depth = load_depth_png_meters(gt_path)
        if gt_depth.shape != (H, W):
            gt_depth = cv2.resize(gt_depth, (W, H), interpolation=cv2.INTER_NEAREST)

        pred_depth = load_pred_depth_png_meters(pred_path)
        if pred_depth.shape != (H, W):
            pred_depth = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_NEAREST)

        # Valid pixels from GT
        valid = gt_depth > 0
        uv = np.column_stack(np.where(valid))  # (v, u)
        if uv.shape[0] == 0:
            continue

        # Subsample if needed
        if args.max_points > 0 and uv.shape[0] > args.max_points:
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(uv.shape[0], size=args.max_points, replace=False)
            uv = uv[sel]

        v = uv[:, 0]
        u = uv[:, 1]
        gt_vals = gt_depth[v, u]
        pred_vals = pred_depth[v, u]

        # Filter by depth range
        ok = (gt_vals >= args.min_depth) & (gt_vals <= args.max_depth)
        ok &= (pred_vals >= args.min_depth) & (pred_vals <= args.max_depth)
        if ok.sum() < 50:
            continue

        u = u[ok]
        v = v[ok]
        gt_vals = gt_vals[ok]
        pred_vals = pred_vals[ok]
        uv_ok = np.column_stack([u, v])

        # Panel[1] GT overlay
        gt_c = colorize_points(gt_vals, args.min_depth, args.max_depth, cmap_name)
        gt_overlay = draw_points(rgb, uv_ok, gt_c, radius=args.point_radius)

        # Panel[2] Pred overlay (same GT valid pixels)
        pred_c = colorize_points(pred_vals, args.min_depth, args.max_depth, cmap_name)
        pred_overlay = draw_points(rgb, uv_ok, pred_c, radius=args.point_radius)

        # Panel[3] Masked pred (grid sampling)
        uv_grid = make_uniform_grid_uv(H, W, args.dense_step)
        if dense_mask is not None:
            uu_g = uv_grid[:, 0]
            vv_g = uv_grid[:, 1]
            keep = dense_mask[vv_g, uu_g] > 0
            uv_grid = uv_grid[keep]

        uu_g = uv_grid[:, 0]
        vv_g = uv_grid[:, 1]
        pred_vals_grid = pred_depth[vv_g, uu_g]
        okg = (pred_vals_grid >= args.min_depth) & (pred_vals_grid <= args.max_depth)
        uv_grid = uv_grid[okg]
        pred_vals_grid = pred_vals_grid[okg]
        pred_c_grid = colorize_points(pred_vals_grid, args.min_depth, args.max_depth, cmap_name)
        masked_pred_overlay = draw_points(rgb, uv_grid, pred_c_grid, radius=args.point_radius)

        # Apply mask to panels
        if dense_mask is not None:
            black_region = dense_mask == 0
            gt_overlay[black_region] = [0, 0, 0]
            pred_overlay[black_region] = [0, 0, 0]
            masked_pred_overlay[black_region] = [0, 0, 0]

        # Labels
        rgb_l = put_panel_label(rgb, 'RGB')
        gt_l = put_panel_label(gt_overlay, 'GT Depth (valid px)')
        pred_l = put_panel_label(pred_overlay, 'Pred Depth (valid px)')
        masked_pred_l = put_panel_label(masked_pred_overlay, f'Masked Pred (step={args.dense_step})')

        # Compose quad: [RGB, Masked_Pred] / [GT, Pred]
        quad = np.concatenate([
            np.concatenate([rgb_l, masked_pred_l], axis=1),
            np.concatenate([gt_l, pred_l], axis=1)
        ], axis=0)
        Image.fromarray(quad).save(out_dir / f'{stem}_quad.png')
        total_saved += 1

        if total_saved % 100 == 0:
            print(f'[PROGRESS] Saved {total_saved} quads...')

    print(f'[DONE] Saved {total_saved} quad images to {out_dir}')


if __name__ == '__main__':
    main()
