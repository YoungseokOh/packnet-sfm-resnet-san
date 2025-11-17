#!/usr/bin/env python3
"""
Enhanced visualization with:
- RGB image
- Inverse depth (1/depth) visualization with colormap
- GT depth overlaid on RGB (colored by depth with transparency)
- FP32 vs NPU comparison (inverse depth)

This creates comprehensive visualizations for Fin_Test_Set_ncdb/viz and viz_npu.

Usage:
  python scripts/visualize_with_inverse_depth_and_gt_overlay.py \\
    --test_file /workspace/data/ncdb-cls-640x384/splits/combined_test.json \\
    --dataset_root /workspace/data/ncdb-cls-640x384 \\
    --gt_dir Fin_Test_Set_ncdb/GT \\
    --rgb_dir Fin_Test_Set_ncdb/RGB \\
    --fp32_dir Fin_Test_Set_ncdb/fp32 \\
    --npu_dir Fin_Test_Set_ncdb/npu \\
    --out_dir Fin_Test_Set_ncdb/viz_enhanced \\
    --mode both
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cv2


def read_test_file(test_file_path):
    test_file_path = Path(test_file_path)
    if test_file_path.suffix == '.json':
        with open(test_file_path, 'r') as f:
            data = json.load(f)
        results = []
        for entry in data:
            if 'new_filename' in entry:
                results.append({'filename': entry['new_filename'], 'dataset_root_override': entry.get('dataset_root')})
            elif 'image_path' in entry:
                filename = Path(entry['image_path']).stem
                if 'synced_data' in Path(entry['image_path']).parts:
                    idx = Path(entry['image_path']).parts.index('synced_data')
                    dataset_root_override = str(Path(*Path(entry['image_path']).parts[:idx+1]))
                else:
                    dataset_root_override = None
                results.append({'filename': filename, 'dataset_root_override': dataset_root_override, 'image_path': entry.get('image_path')})
        return results
    # fallback txt
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '.png' in line:
            filename = line.split('.png')[0].split('/')[-1]
        else:
            parts = line.split()
            filename = parts[-1] if parts else line
        results.append({'filename': filename, 'dataset_root_override': None})
    return results


def load_depth_png(path: Path):
    """Load depth from 16-bit PNG (value / 256 = meters)"""
    if not path.exists():
        return None
    img = np.array(Image.open(path), dtype=np.uint16)
    return img.astype(np.float32) / 256.0


def load_depth_npy(path: Path):
    """Load depth from npy (already in meters)"""
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def create_inverse_depth_colormap(depth, min_depth=0.5, max_depth=15.0, cmap='plasma'):
    """
    Create inverse depth (1/depth) visualization with colormap.
    Args:
        depth: depth array in meters
        min_depth, max_depth: clipping range
        cmap: matplotlib colormap name
    Returns:
        RGB image (H, W, 3) uint8
    """
    # Clip depth to valid range
    depth_clipped = np.clip(depth, min_depth, max_depth)
    
    # Compute inverse depth
    inverse_depth = 1.0 / depth_clipped
    
    # Normalize inverse depth to [0, 1] for colormap
    inv_min = 1.0 / max_depth
    inv_max = 1.0 / min_depth
    inverse_depth_norm = (inverse_depth - inv_min) / (inv_max - inv_min)
    inverse_depth_norm = np.clip(inverse_depth_norm, 0, 1)
    
    # Apply colormap
    colormap = plt.get_cmap(cmap)
    colored = colormap(inverse_depth_norm)[:, :, :3]  # Drop alpha
    colored_uint8 = (colored * 255).astype(np.uint8)
    
    return colored_uint8


def overlay_gt_depth_on_rgb(rgb_img, gt_depth, min_depth=0.5, max_depth=15.0, alpha=0.5, cmap='jet'):
    """
    Overlay GT depth on RGB image with transparency.
    Args:
        rgb_img: PIL Image or numpy array (H, W, 3)
        gt_depth: depth array (H, W) in meters
        min_depth, max_depth: depth range for colormap
        alpha: overlay transparency (0=transparent, 1=opaque)
        cmap: matplotlib colormap
    Returns:
        PIL Image
    """
    if isinstance(rgb_img, Image.Image):
        rgb_arr = np.array(rgb_img)
    else:
        rgb_arr = rgb_img
    
    # Create mask for valid depth
    valid_mask = (gt_depth >= min_depth) & (gt_depth <= max_depth)
    
    # Normalize depth to [0, 1]
    depth_norm = np.clip((gt_depth - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Apply colormap
    colormap = plt.get_cmap(cmap)
    depth_colored = colormap(depth_norm)[:, :, :3]  # Drop alpha
    depth_colored_uint8 = (depth_colored * 255).astype(np.uint8)
    
    # Blend with RGB where depth is valid
    overlay = rgb_arr.copy()
    overlay[valid_mask] = (alpha * depth_colored_uint8[valid_mask] + (1 - alpha) * rgb_arr[valid_mask]).astype(np.uint8)
    
    return Image.fromarray(overlay)


def visualize_fp32_only(filename, rgb_path, gt_path, fp32_path, out_path):
    """
    Create visualization for FP32 only:
    Row 1: RGB, RGB+GT overlay, Inverse depth (GT)
    Row 2: Inverse depth (FP32), Absolute diff (GT-FP32), Relative error map
    """
    # Load data
    rgb_img = Image.open(rgb_path) if rgb_path and rgb_path.exists() else None
    gt_depth = load_depth_png(gt_path) if gt_path and gt_path.exists() else None
    fp32_depth = load_depth_npy(fp32_path) if fp32_path and fp32_path.exists() else None
    
    if rgb_img is None or gt_depth is None or fp32_depth is None:
        return False
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1, Col 1: RGB
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('RGB')
    axes[0, 0].axis('off')
    
    # Row 1, Col 2: RGB + GT overlay
    rgb_gt_overlay = overlay_gt_depth_on_rgb(rgb_img, gt_depth, alpha=0.4, cmap='jet')
    axes[0, 1].imshow(rgb_gt_overlay)
    axes[0, 1].set_title('RGB + GT Depth Overlay')
    axes[0, 1].axis('off')
    
    # Row 1, Col 3: Inverse depth (GT)
    inv_gt = create_inverse_depth_colormap(gt_depth, cmap='plasma')
    axes[0, 2].imshow(inv_gt)
    axes[0, 2].set_title('Inverse Depth (GT)')
    axes[0, 2].axis('off')
    
    # Row 2, Col 1: Inverse depth (FP32)
    inv_fp32 = create_inverse_depth_colormap(fp32_depth, cmap='plasma')
    axes[1, 0].imshow(inv_fp32)
    axes[1, 0].set_title('Inverse Depth (FP32)')
    axes[1, 0].axis('off')
    
    # Row 2, Col 2: Absolute diff (GT - FP32)
    abs_diff = np.abs(gt_depth - fp32_depth)
    im_diff = axes[1, 1].imshow(abs_diff, cmap='viridis', vmin=0, vmax=2.0)
    axes[1, 1].set_title('Abs Diff (GT - FP32)')
    axes[1, 1].axis('off')
    plt.colorbar(im_diff, ax=axes[1, 1], fraction=0.046)
    
    # Row 2, Col 3: Relative error (for valid pixels)
    valid_mask = (gt_depth >= 0.5) & (gt_depth <= 15.0)
    rel_error = np.zeros_like(gt_depth)
    rel_error[valid_mask] = np.abs(gt_depth[valid_mask] - fp32_depth[valid_mask]) / gt_depth[valid_mask]
    im_rel = axes[1, 2].imshow(rel_error, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 2].set_title('Relative Error (GT-FP32)/GT')
    axes[1, 2].axis('off')
    plt.colorbar(im_rel, ax=axes[1, 2], fraction=0.046)
    
    plt.suptitle(f'{filename} - FP32 Evaluation', fontsize=16)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def visualize_fp32_vs_npu(filename, rgb_path, gt_path, fp32_path, npu_path, out_path):
    """
    Create visualization for FP32 vs NPU:
    Row 1: RGB, RGB+GT overlay, Inverse depth (GT)
    Row 2: Inverse depth (FP32), Inverse depth (NPU), Inverse depth diff (FP32-NPU)
    Row 3: GT-FP32 abs diff, GT-NPU abs diff, FP32-NPU abs diff
    """
    # Load data
    rgb_img = Image.open(rgb_path) if rgb_path and rgb_path.exists() else None
    gt_depth = load_depth_png(gt_path) if gt_path and gt_path.exists() else None
    fp32_depth = load_depth_npy(fp32_path) if fp32_path and fp32_path.exists() else None
    
    # Load NPU (composed from integer + fractional)
    npu_int_path = npu_path.parent.parent / 'integer_sigmoid' / f'{filename}.npy'
    npu_frac_path = npu_path.parent.parent / 'fractional_sigmoid' / f'{filename}.npy'
    if npu_int_path.exists() and npu_frac_path.exists():
        npu_int = load_depth_npy(npu_int_path)
        npu_frac = load_depth_npy(npu_frac_path)
        npu_depth = npu_int * 15.0 + npu_frac
    else:
        # Try loading composed png
        npu_png_path = npu_path.parent.parent / 'png' / f'{filename}.png'
        npu_depth = load_depth_png(npu_png_path) if npu_png_path.exists() else None
    
    if rgb_img is None or gt_depth is None or fp32_depth is None or npu_depth is None:
        return False
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Row 1, Col 1: RGB
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('RGB')
    axes[0, 0].axis('off')
    
    # Row 1, Col 2: RGB + GT overlay
    rgb_gt_overlay = overlay_gt_depth_on_rgb(rgb_img, gt_depth, alpha=0.4, cmap='jet')
    axes[0, 1].imshow(rgb_gt_overlay)
    axes[0, 1].set_title('RGB + GT Depth Overlay')
    axes[0, 1].axis('off')
    
    # Row 1, Col 3: Inverse depth (GT)
    inv_gt = create_inverse_depth_colormap(gt_depth, cmap='plasma')
    axes[0, 2].imshow(inv_gt)
    axes[0, 2].set_title('Inverse Depth (GT)')
    axes[0, 2].axis('off')
    
    # Row 2, Col 1: Inverse depth (FP32)
    inv_fp32 = create_inverse_depth_colormap(fp32_depth, cmap='plasma')
    axes[1, 0].imshow(inv_fp32)
    axes[1, 0].set_title('Inverse Depth (FP32)')
    axes[1, 0].axis('off')
    
    # Row 2, Col 2: Inverse depth (NPU)
    inv_npu = create_inverse_depth_colormap(npu_depth, cmap='plasma')
    axes[1, 1].imshow(inv_npu)
    axes[1, 1].set_title('Inverse Depth (NPU)')
    axes[1, 1].axis('off')
    
    # Row 2, Col 3: Inverse depth diff (FP32 - NPU) in inverse space
    inv_fp32_arr = 1.0 / np.clip(fp32_depth, 0.5, 15.0)
    inv_npu_arr = 1.0 / np.clip(npu_depth, 0.5, 15.0)
    inv_diff = np.abs(inv_fp32_arr - inv_npu_arr)
    im_inv_diff = axes[1, 2].imshow(inv_diff, cmap='viridis', vmin=0, vmax=0.1)
    axes[1, 2].set_title('Inverse Depth Diff (|FP32-NPU|)')
    axes[1, 2].axis('off')
    plt.colorbar(im_inv_diff, ax=axes[1, 2], fraction=0.046)
    
    # Row 3, Col 1: GT - FP32 abs diff
    abs_diff_fp32 = np.abs(gt_depth - fp32_depth)
    im_diff_fp32 = axes[2, 0].imshow(abs_diff_fp32, cmap='hot', vmin=0, vmax=2.0)
    axes[2, 0].set_title('Abs Diff (GT - FP32)')
    axes[2, 0].axis('off')
    plt.colorbar(im_diff_fp32, ax=axes[2, 0], fraction=0.046)
    
    # Row 3, Col 2: GT - NPU abs diff
    abs_diff_npu = np.abs(gt_depth - npu_depth)
    im_diff_npu = axes[2, 1].imshow(abs_diff_npu, cmap='hot', vmin=0, vmax=2.0)
    axes[2, 1].set_title('Abs Diff (GT - NPU)')
    axes[2, 1].axis('off')
    plt.colorbar(im_diff_npu, ax=axes[2, 1], fraction=0.046)
    
    # Row 3, Col 3: FP32 - NPU abs diff
    abs_diff_fp32_npu = np.abs(fp32_depth - npu_depth)
    im_diff_fp32_npu = axes[2, 2].imshow(abs_diff_fp32_npu, cmap='viridis', vmin=0, vmax=1.0)
    axes[2, 2].set_title('Abs Diff (FP32 - NPU)')
    axes[2, 2].axis('off')
    plt.colorbar(im_diff_fp32_npu, ax=axes[2, 2], fraction=0.046)
    
    plt.suptitle(f'{filename} - FP32 vs NPU Evaluation', fontsize=16)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--rgb_dir', required=True)
    parser.add_argument('--fp32_dir', required=True)
    parser.add_argument('--npu_dir', default=None, help='NPU outputs dir (if comparing FP32 vs NPU)')
    parser.add_argument('--out_dir', default='Fin_Test_Set_ncdb/viz_enhanced')
    parser.add_argument('--mode', choices=['fp32', 'npu', 'both'], default='both', help='Visualization mode')
    parser.add_argument('--n', type=int, default=None, help='Limit number of images')
    args = parser.parse_args()
    
    entries = read_test_file(args.test_file)
    if args.n:
        entries = entries[:args.n]
    
    gt_dir = Path(args.gt_dir)
    rgb_dir = Path(args.rgb_dir)
    fp32_dir = Path(args.fp32_dir)
    npu_dir = Path(args.npu_dir) if args.npu_dir else None
    out_dir = Path(args.out_dir)
    
    count = 0
    for entry in entries:
        filename = entry['filename']
        
        gt_path = gt_dir / f'{filename}.png'
        rgb_path = rgb_dir / f'{filename}.png'
        if not rgb_path.exists():
            # Try jpg
            rgb_path = rgb_dir / f'{filename}.jpg'
        fp32_path = fp32_dir / f'{filename}.npy'
        
        # Debug: check if files exist
        exists_status = f"GT:{gt_path.exists()} RGB:{rgb_path.exists()} FP32:{fp32_path.exists()}"
        if not (gt_path.exists() and rgb_path.exists() and fp32_path.exists()):
            print(f'Skipping {filename}: {exists_status}')
            continue
        
        if args.mode == 'fp32' or (args.mode == 'both' and npu_dir is None):
            out_path = out_dir / 'fp32' / f'{filename}_fp32_viz.png'
            success = visualize_fp32_only(filename, rgb_path, gt_path, fp32_path, out_path)
            if success:
                count += 1
                print(f'Created FP32 viz: {out_path}')
        
        if (args.mode == 'npu' or args.mode == 'both') and npu_dir is not None:
            npu_path = npu_dir / 'png' / f'{filename}.png'  # placeholder, actual loading in function
            out_path = out_dir / 'npu' / f'{filename}_npu_viz.png'
            success = visualize_fp32_vs_npu(filename, rgb_path, gt_path, fp32_path, npu_path, out_path)
            if success:
                count += 1
                print(f'Created NPU viz: {out_path}')
    
    print(f'\nTotal visualizations created: {count}')
    
    # Create index HTML for each subfolder
    for subdir in ['fp32', 'npu']:
        subdir_path = out_dir / subdir
        if subdir_path.exists():
            create_html_index(subdir_path, f'{subdir.upper()} Visualizations')


def create_html_index(viz_dir: Path, title: str):
    """Create a simple HTML index for visualizations."""
    images = sorted(viz_dir.glob('*.png'))
    if not images:
        return
    
    html_lines = [
        '<html><head><meta charset="utf-8">',
        f'<title>{title}</title>',
        '<style>body{font-family:Arial;padding:20px} .gallery{display:flex;flex-wrap:wrap} .item{margin:10px;text-align:center} img{max-width:400px}</style>',
        '</head><body>',
        f'<h1>{title}</h1>',
        f'<p>Total images: {len(images)}</p>',
        '<div class="gallery">'
    ]
    
    for img in images:
        html_lines.append(
            f'<div class="item"><a href="{img.name}" target="_blank"><img src="{img.name}"></a><br>{img.stem}</div>'
        )
    
    html_lines.append('</div></body></html>')
    
    index_path = viz_dir / 'index.html'
    with open(index_path, 'w') as f:
        f.write('\n'.join(html_lines))
    
    print(f'Created index: {index_path}')


if __name__ == '__main__':
    main()
