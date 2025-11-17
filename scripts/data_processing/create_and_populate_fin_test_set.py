#!/usr/bin/env python3
"""
Create and populate Fin_Test_Set_ncdb with GT, RGB, FP32 outputs and visualizations.

Creates the following structure by copying from dataset and outputs:
Fin_Test_Set_ncdb/
  GT/
  RGB/
  fp32/
  viz/

Usage:
  python scripts/create_and_populate_fin_test_set.py --test_file /workspace/data/ncdb-cls-640x384/splits/combined_test.json --dataset_root /workspace/data/ncdb-cls-640x384 --fp32_dir outputs/ncdb_test_fp32_full_separated/resnetsan01_fp32/depth_fp32 --out_root Fin_Test_Set_ncdb

"""

import argparse
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image
import os

from visualize_fp32_vs_npu_vs_gt import read_test_file, load_gt_depth


def copy_file_safe(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copyfile(str(src), str(dst))


def load_fp32_npy(fp32_dir: Path, filename: str):
    p = fp32_dir / f'{filename}.npy'
    if p.exists():
        arr = np.load(p)
        if arr.ndim == 4:
            arr = arr[0,0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr
    return None


def save_depth_as_png(depth_arr: np.ndarray, dst: Path):
    # Convert depth (float) to uint16 by multiplying by 256
    dst.parent.mkdir(parents=True, exist_ok=True)
    depth_uint16 = (depth_arr * 256.0).astype(np.uint16)
    Image.fromarray(depth_uint16).save(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--fp32_dir', required=True)
    parser.add_argument('--fp32_model_name', default='resnetsan01_fp32')
    parser.add_argument('--out_root', default='Fin_Test_Set_ncdb')
    parser.add_argument('--n', default=None, type=int, help='If set, limit the number of images')
    args = parser.parse_args()

    out_root = Path(args.out_root)
    gt_out = out_root / 'GT'
    rgb_out = out_root / 'RGB'
    fp32_out = out_root / 'fp32'
    viz_out = out_root / 'viz'

    # read test entries
    entries = read_test_file(args.test_file)
    if args.n:
        entries = entries[:args.n]

    copied = 0
    missing = []
    for e in entries:
        filename = e['filename']
        dataset_root_override = e.get('dataset_root_override') or e.get('dataset_root') or args.dataset_root

        # Copy GT
        src_gt = Path(dataset_root_override) / 'newest_depth_maps' / f'{filename}.png'
        if not src_gt.exists():
            # try deeper search
            src_gt = Path(args.dataset_root) / 'depth_selection' / 'val_selection_cropped' / 'groundtruth_depth' / f'{filename}.png'
        if not src_gt.exists():
            missing.append((filename, 'GT'))
            continue
        dst_gt = gt_out / f'{filename}.png'
        copy_file_safe(src_gt, dst_gt)

        # Copy RGB — attempt to find in dataset by searching for 'image_a6' or 'image_a5'
        # Look for exact image_path if present in entry
        # Try a6 then a5 fallback
        src_rgb = None
        # if entry has image_path
        if 'image_path' in e and e['image_path']:
            p = Path(e['image_path'])
            if p.exists():
                src_rgb = p
        if src_rgb is None:
            # check common folders under dataset root
            p1 = Path(dataset_root_override) / 'synced_data' / 'image_a6' / f'{filename}.png'
            p2 = Path(dataset_root_override) / 'synced_data' / 'image_a5' / f'{filename}.png'
            p3 = Path(dataset_root_override) / 'synced_data' / 'image_a6' / f'{filename}.jpg'
            p4 = Path(dataset_root_override) / 'image_a6' / f'{filename}.png'
            for p in [p1, p2, p3, p4]:
                if p.exists():
                    src_rgb = p
                    break
        if src_rgb is None:
            missing.append((filename, 'RGB'))
            continue
        dst_rgb = rgb_out / src_rgb.name
        copy_file_safe(src_rgb, dst_rgb)

        # Copy FP32 npy
        src_fp32 = Path(args.fp32_dir) / f'{filename}.npy'
        if not src_fp32.exists():
            # maybe depth_png output in outputs/ncdb_test_fp32_full/<filename>.png
            fallback_png = Path('outputs/ncdb_test_fp32_full') / f'{filename}.png'
            if fallback_png.exists():
                # copy png to fp32_out
                dst_fp32_png = fp32_out / f'{filename}.png'
                copy_file_safe(fallback_png, dst_fp32_png)
            else:
                missing.append((filename, 'FP32'))
                continue
        else:
            dst_fp32 = fp32_out / src_fp32.name
            copy_file_safe(src_fp32, dst_fp32)

        # Save visualization for this sample: RGB / GT / FP32 depth + diff
        # FP32 loaded as npy or png
        # Load rgb PIL
        rgb_img = Image.open(dst_rgb)
        gt_depth = np.array(Image.open(dst_gt), dtype=np.uint16).astype(np.float32) / 256.0
        # fp32: if npy
        if (fp32_out / f'{filename}.npy').exists():
            fp32_depth = np.load(fp32_out / f'{filename}.npy')
            if fp32_depth.ndim == 4:
                fp32_depth = fp32_depth[0,0]
            elif fp32_depth.ndim == 3 and fp32_depth.shape[0] == 1:
                fp32_depth = fp32_depth[0]
        else:
            # png case
            fp32_png_path = fp32_out / f'{filename}.png'
            if fp32_png_path.exists():
                fp32_depth = np.array(Image.open(fp32_png_path), dtype=np.uint16).astype(np.float32) / 256.0
            else:
                missing.append((filename, 'FP32-load'))
                continue

        # Build viz: show RGB, GT depth, FP32 depth, and diff GT-FP32
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        # Subplot 1: RGB
        ax = plt.subplot(1, 4, 1)
        ax.imshow(rgb_img)
        ax.set_title('RGB')
        ax.axis('off')
        # Subplot 2: GT depth
        ax2 = plt.subplot(1, 4, 2)
        im2 = ax2.imshow(gt_depth, cmap='magma', vmin=0.5, vmax=15)
        ax2.set_title('GT depth')
        plt.colorbar(im2, ax=ax2, fraction=0.05)
        ax2.axis('off')
        # Subplot 3: FP32 depth
        ax3 = plt.subplot(1, 4, 3)
        im3 = ax3.imshow(fp32_depth, cmap='magma', vmin=0.5, vmax=15)
        ax3.set_title('FP32 depth')
        plt.colorbar(im3, ax=ax3, fraction=0.05)
        ax3.axis('off')
        # Subplot 4: Abs diff GT-FP32
        d = np.abs(gt_depth - fp32_depth)
        ax4 = plt.subplot(1, 4, 4)
        im4 = ax4.imshow(d, cmap='viridis')
        ax4.set_title('Absolute diff')
        plt.colorbar(im4, ax=ax4, fraction=0.05)
        ax4.axis('off')

        plt.suptitle(filename)
        viz_path = viz_out / f'{filename}_rgb_gt_fp32.png'
        viz_out.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path, dpi=150)
        plt.close()

        copied += 1

    # End for each sample
    print(f'Copied/visualized: {copied} samples, missing entries: {len(missing)}')
    if missing:
        print('Missing details (first 10):')
        print('\n'.join([f'{m[0]}: {m[1]}' for m in missing[:10]]))

    # Summaries: counts
    print('GT count:', len(list(gt_out.glob('*'))))
    print('RGB count:', len(list(rgb_out.glob('*'))))
    print('FP32 count:', len(list(fp32_out.glob('*'))))
    print('Viz count:', len(list(viz_out.glob('*'))))


if __name__ == '__main__':
    main()
