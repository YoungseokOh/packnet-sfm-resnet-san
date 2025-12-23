#!/usr/bin/env python3
"""Visualize projected depth-map pixels vs predicted depth (same pixel grid).

You said you already have projected outputs; in NCDB indoor loops these live in
folders like:
  - newest_distance_maps/<stem>.png
  - newest_original_depth_maps/<stem>.png

Those are already in image pixel coordinates, so we can:
  1) load projected depth map -> valid pixels where depth>0
  2) sample predicted depth at the same pixels
  3) visualize / compute error on exactly those pixels

This avoids re-projecting raw PCDs entirely.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from packnet_sfm.visualization.colormaps import create_custom_depth_colormap


def load_depth_png_meters(path: Path) -> np.ndarray:
    """Load NCDB depth PNG.

    Convention used in this repo: depth is stored as uint16/int32 with value = meters*256.
    Some files might be uint16, some int32 depending on how they were exported.
    """
    arr = np.asarray(Image.open(path))
    arr_f = arr.astype(np.float32)
    # Heuristic: if values look like fixed-point, undo it
    if arr_f.max() > 255:
        arr_f /= 256.0
    return arr_f


def load_pred_depth_png_meters(path: Path) -> np.ndarray:
    arr16 = np.asarray(Image.open(path), dtype=np.uint16)
    return arr16.astype(np.float32) / 256.0


def _get_cmap(cmap: str, vmin: float, vmax: float):
    """Return a matplotlib colormap.

    Supports built-ins like 'turbo' and also our shared 'custom' depth colormap.
    """
    if cmap == 'custom':
        return create_custom_depth_colormap(vmin, vmax)
    return plt.get_cmap(cmap)


def colorize_points(values: np.ndarray, vmin: float, vmax: float, cmap: str = 'turbo') -> np.ndarray:
    norm = np.clip((values - vmin) / max(1e-6, (vmax - vmin)), 0, 1)
    cm = _get_cmap(cmap, vmin, vmax)
    return (cm(norm)[:, :3] * 255).astype(np.uint8)


def colorize_dense(depth: np.ndarray, vmin: float, vmax: float, cmap: str = 'turbo') -> np.ndarray:
    """Colorize a dense depth map to RGB using a matplotlib colormap."""
    cm = _get_cmap(cmap, vmin, vmax)
    norm = np.clip((depth.astype(np.float32) - vmin) / max(1e-6, (vmax - vmin)), 0, 1)
    rgb = (cm(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def draw_points(rgb: np.ndarray, uv: np.ndarray, colors: np.ndarray, radius: int = 2) -> np.ndarray:
    out = rgb.copy()
    # colors are RGB, cv2 expects BGR for color arg, but since base image is RGB
    # and we save with PIL (RGB), we keep RGB order for the circle color
    for (u, v), c in zip(uv, colors):
        cv2.circle(out, (int(u), int(v)), radius, (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
    return out


def put_panel_label(img: np.ndarray, text: str) -> np.ndarray:
    """Put a small label at the top-left corner of a panel."""
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


def subsample_points_uniform(uv: np.ndarray, step: int) -> np.ndarray:
    """Uniformly subsample pixel coordinates on a grid.

    uv is Nx2 with (u,v). Keeps points where u%step==0 and v%step==0.
    This gives a visually spaced-out sparse overlay.
    """
    if step <= 1:
        return uv
    u = uv[:, 0]
    v = uv[:, 1]
    keep = (u % step == 0) & (v % step == 0)
    return uv[keep]


def make_uniform_grid_uv(H: int, W: int, step: int) -> np.ndarray:
    """Create uniformly spaced (u,v) pixel coordinates over the whole image."""
    step = max(1, int(step))
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)])


def load_test_entries(test_json: Path) -> List[Dict[str, str]]:
    data = json.loads(test_json.read_text())
    out: List[Dict[str, str]] = []
    for e in data:
        if 'new_filename' not in e:
            continue
        out.append({'stem': Path(e['new_filename']).stem, 'dataset_root': e.get('dataset_root', '')})
    return out


def find_existing_projected_depth(dataset_root: Path, stem: str, depth_subdir: str) -> Optional[Path]:
    p = dataset_root / depth_subdir / f'{stem}.png'
    if p.exists():
        return p
    return None


def find_pred_depth_png(pred_dir: Path, stem: str, pattern: str) -> Optional[Path]:
    """Find predicted depth PNG by pattern.

    Pattern supports:
      - '{stem}' placeholder
      - '*' glob

    Examples:
      - 'depth_{stem}.png' (default)
      - '{stem}_depth.png'
      - '*{stem}*depth*.png'
    """
    pat = pattern.format(stem=stem)
    # Fast path: exact file
    p = pred_dir / pat
    if p.exists():
        return p
    # Glob path
    matches = sorted(pred_dir.glob(pat))
    if matches:
        return matches[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_json', type=str, required=True, help='combined_test.json with dataset_root + new_filename')
    ap.add_argument('--rgb_dir', type=str, required=True, help='RGB directory containing <stem>.jpg/png (e.g. test_set/images)')
    ap.add_argument('--pred_depth_dir', type=str, required=True, help='Directory containing depth_<stem>.png predictions')
    ap.add_argument('--pred_pattern', type=str, default='depth_{stem}.png',
                    help="Pred depth filename pattern inside pred_depth_dir (supports '{stem}' and '*').")
    ap.add_argument('--gt_depth_dir', type=str, help='Override GT depth directory (ignores dataset_root in JSON)')
    ap.add_argument('--depth_subdir', type=str, default='newest_original_depth_maps',
                    help='Projected depth map folder within dataset_root (e.g. newest_distance_maps, newest_original_depth_maps, newest_original_distance_maps)')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--image_shape', nargs=2, type=int, default=[384, 640], help='H W')
    ap.add_argument('--min_depth', type=float, default=0.1)
    ap.add_argument('--max_depth', type=float, default=10.0)
    ap.add_argument('--cmap', type=str, default='custom',
                    help="Matplotlib colormap name (ignored; depth panels use shared 'custom' colormap for consistency)")
    ap.add_argument('--mask_path', type=str, default=None,
                    help='Path to binary mask for dense depth panel (white=valid, black=invalid). If not provided, no mask applied.')
    ap.add_argument('--point_radius', type=int, default=1)
    ap.add_argument('--sparse_step', type=int, default=2,
                    help='(legacy) spacing step for overlay points (GT-valid overlays)')
    ap.add_argument('--dense_step', type=int, default=3,
                    help='Grid step for panel[3] masked pred (smaller=denser; e.g. 2~4 is lightly sparse)')
    ap.add_argument('--br_crop_top', type=int, default=0,
                    help='(legacy) kept for backward compatibility; crop is disabled by default')
    ap.add_argument('--save_pred_sparse', action='store_true',
                    help='(legacy) optionally export pred sparse overlays (not part of quad)')
    ap.add_argument('--max_points', type=int, default=30000, help='Subsample valid pixels for speed')
    ap.add_argument('--num_samples', type=int, default=20, help='0 = all')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    H, W = int(args.image_shape[0]), int(args.image_shape[1])
    rgb_dir = Path(args.rgb_dir)
    pred_dir = Path(args.pred_depth_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mask for dense depth panel (BR) if provided
    dense_mask = None
    if args.mask_path:
        mask_path = Path(args.mask_path)
        if mask_path.exists():
            dense_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if dense_mask.shape != (H, W):
                dense_mask = cv2.resize(dense_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            print(f'[INFO] Loaded mask: {mask_path}, shape={dense_mask.shape}, valid_pixels={np.sum(dense_mask > 0)}')
        else:
            print(f'[WARN] Mask not found: {mask_path}, proceeding without mask')

    entries = load_test_entries(Path(args.test_json))
    if args.num_samples > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(entries), size=min(args.num_samples, len(entries)), replace=False)
        entries = [entries[i] for i in sorted(idx)]

    total_saved = 0

    for e in entries:
        stem = e['stem']
        dataset_root = Path(e['dataset_root']) if e['dataset_root'] else None

        # rgb
        rgb_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            cand = rgb_dir / f'{stem}{ext}'
            if cand.exists():
                rgb_path = cand
                break
        if rgb_path is None:
            continue

        # projected depth map
        if args.gt_depth_dir:
            proj_path = Path(args.gt_depth_dir) / f'{stem}.png'
            if not proj_path.exists():
                proj_path = None
        elif dataset_root is not None:
            proj_path = find_existing_projected_depth(dataset_root, stem, args.depth_subdir)
        else:
            proj_path = None

        if proj_path is None:
            continue

        # predicted depth png
        pred_path = find_pred_depth_png(pred_dir, stem, args.pred_pattern)
        if pred_path is None:
            continue

        rgb = np.asarray(Image.open(rgb_path).convert('RGB').resize((W, H), Image.BILINEAR))
        proj_depth = load_depth_png_meters(proj_path)
        if proj_depth.shape != (H, W):
            proj_depth = cv2.resize(proj_depth, (W, H), interpolation=cv2.INTER_NEAREST)

        pred_depth = load_pred_depth_png_meters(pred_path)
        if pred_depth.shape != (H, W):
            pred_depth = cv2.resize(pred_depth, (W, H), interpolation=cv2.INTER_NEAREST)

        # valid pixels from projected depth
        valid = proj_depth > 0
        uv = np.column_stack(np.where(valid))  # (v,u)
        if uv.shape[0] == 0:
            continue

        # optional subsample
        if args.max_points > 0 and uv.shape[0] > args.max_points:
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(uv.shape[0], size=args.max_points, replace=False)
            uv = uv[sel]

        v = uv[:, 0]
        u = uv[:, 1]
        proj_vals = proj_depth[v, u]
        pred_vals = pred_depth[v, u]

        # clamp and filter
        ok = (proj_vals >= args.min_depth) & (proj_vals <= args.max_depth)
        ok &= (pred_vals >= args.min_depth) & (pred_vals <= args.max_depth)
        if ok.sum() < 50:
            # too few points -> won't be informative
            continue
        u = u[ok]
        v = v[ok]
        proj_vals = proj_vals[ok]
        pred_vals = pred_vals[ok]
        uv_ok = np.column_stack([u, v])

        # ========== COLORMAP IS ALWAYS 'custom' ==========
        cmap_name = 'custom'

        # Panel[1] GT overlay: ALL valid GT pixels (no subsampling)
        proj_c = colorize_points(proj_vals, args.min_depth, args.max_depth, cmap_name)
        gt_overlay = draw_points(rgb, uv_ok, proj_c, radius=args.point_radius)

        # Panel[2] Pred overlay: same valid GT pixel coords, pred values
        pred_c = colorize_points(pred_vals, args.min_depth, args.max_depth, cmap_name)
        pred_overlay = draw_points(rgb, uv_ok, pred_c, radius=args.point_radius)

        # Panel[3] Masked pred: lightly-sparse grid sampling over the whole image, then mask out invalid
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

        # Apply mask to panels 1, 2, 3: set invalid regions (mask==0) to black
        if dense_mask is not None:
            black_region = dense_mask == 0
            gt_overlay[black_region] = [0, 0, 0]
            pred_overlay[black_region] = [0, 0, 0]
            masked_pred_overlay[black_region] = [0, 0, 0]

        # Labels
        rgb_l = put_panel_label(rgb, 'RGB')
        gt_l = put_panel_label(gt_overlay, 'GT Depth (valid px)')
        pred_l = put_panel_label(pred_overlay, 'Pred Depth (valid px)')
        masked_pred_l = put_panel_label(masked_pred_overlay, f'Masked Pred (grid step={int(args.dense_step)})')

        # Compose quad: [RGB, Masked_Pred] / [GT, Pred]
        quad = np.concatenate([
            np.concatenate([rgb_l, masked_pred_l], axis=1),
            np.concatenate([gt_l, pred_l], axis=1)
        ], axis=0)
        Image.fromarray(quad).save(out_dir / f'{stem}_quad.png')
        total_saved += 1

    print(f'[DONE] Saved {total_saved} samples to {out_dir}')


if __name__ == '__main__':
    main()
