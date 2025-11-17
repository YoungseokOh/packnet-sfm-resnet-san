#!/usr/bin/env python3
"""
Visualize FP32 vs NPU (Dual-Head) vs GT depth maps.

This script will generate per-image comparisons showing:
  - GT depth
  - FP32 composed depth
  - NPU composed depth
  - Absolute diffs: GT-FP32, GT-NPU, FP32-NPU

Usage:
  python scripts/visualize_fp32_vs_npu_vs_gt.py --comp_json outputs/npu_vs_fp32_component_and_gt_eval.json --n --top_type npu

Supported --top_type: npu | fp32 | component | list
If --list is used, pass image IDs comma separated via --images 0000000038,0000000056
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from typing import List, Optional


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
                img_path = Path(entry['image_path'])
                filename = img_path.stem
                if 'synced_data' in img_path.parts:
                    idx = img_path.parts.index('synced_data')
                    dataset_root_override = str(Path(*img_path.parts[:idx+1]))
                else:
                    dataset_root_override = None
                results.append({'filename': filename, 'dataset_root_override': dataset_root_override})
        return results
    else:
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
                filename = parts[-1]
            results.append({'filename': filename, 'dataset_root_override': None})
        return results


def load_gt_depth(dataset_root, filename, dataset_root_override=None):
    base = Path(dataset_root_override) if dataset_root_override else Path(dataset_root)
    depth_path = base / 'newest_depth_maps' / f'{filename}.png'
    if not depth_path.exists():
        # fallback to KITTI style
        depth_path = Path(dataset_root) / 'depth_selection' / 'val_selection_cropped' / 'groundtruth_depth' / f'{filename}.png'
    if not depth_path.exists():
        return None
    depth_png = np.array(Image.open(depth_path), dtype=np.uint16)
    return depth_png.astype(np.float32) / 256.0


def load_fp32_components(fp32_dir: Path, model_name: str, filename: str, precision: str = 'fp32'):
    base_dir = Path(fp32_dir) / model_name
    integer_path = base_dir / f'integer_{precision}' / f'{filename}.npy'
    fractional_path = base_dir / f'fractional_{precision}' / f'{filename}.npy'
    depth_path = base_dir / f'depth_{precision}' / f'{filename}.npy'
    if depth_path.exists():
        d = np.load(depth_path)
        if d.ndim == 4:
            d = d[0, 0]
        elif d.ndim == 3:
            d = d[0]
        return d, None, None
    if integer_path.exists() and fractional_path.exists():
        i = np.load(integer_path)
        f = np.load(fractional_path)
        if i.ndim == 3 and i.shape[0] == 1:
            i = i[0]
            f = f[0]
        elif i.ndim == 4 and i.shape[0] == 1 and i.shape[1] == 1:
            i = i[0,0]
            f = f[0,0]
        return None, i, f
    # fallback: try npz file
    npz = Path(fp32_dir) / f'{filename}.npz'
    if npz.exists():
        data = np.load(npz)
        if 'depth' in data:
            d = data['depth']
            if d.ndim == 4:
                d = d[0,0]
            elif d.ndim == 3:
                d = d[0]
            i = data.get('integer_sigmoid')
            f = data.get('fractional_sigmoid')
            if i is not None and i.ndim == 3 and i.shape[0] == 1:
                i = i[0]
                f = f[0]
            return d, i, f
    return None, None, None


def load_npu_components(npu_dir: Path, filename: str):
    i_path = Path(npu_dir) / 'integer_sigmoid' / f'{filename}.npy'
    f_path = Path(npu_dir) / 'fractional_sigmoid' / f'{filename}.npy'
    if i_path.exists() and f_path.exists():
        i = np.load(i_path)
        f = np.load(f_path)
        if i.ndim == 3 and i.shape[0] == 1:
            i = i[0]
            f = f[0]
        elif i.ndim == 4 and i.shape[0] == 1 and i.shape[1] == 1:
            i = i[0,0]
            f = f[0,0]
        return i, f
    return None, None


def compose_depth_from_components(i, f, max_depth=15.0):
    return i * max_depth + f


def compute_metrics(gt, pred, min_depth=0.5, max_depth=15.0):
    mask = (gt >= min_depth) & (gt <= max_depth)
    if mask.sum() == 0:
        return None
    gt_m = gt[mask]
    pred_m = pred[mask]
    thresh = np.maximum(gt_m/pred_m, pred_m/gt_m)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()
    abs_rel = np.mean(np.abs(gt_m - pred_m) / gt_m)
    sq_rel = np.mean(((gt_m - pred_m)**2) / gt_m)
    rmse = np.sqrt(np.mean((gt_m - pred_m)**2))
    rmse_log = np.sqrt(np.mean((np.log(gt_m) - np.log(pred_m))**2))
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


def plot_comparison(gt, fp32_depth, npu_depth, fp32_metrics, npu_metrics, out_path: Path, title: Optional[str] = None):
    # Display an image with colorbar and title
    plt.figure(figsize=(16, 8))
    vmin, vmax = 0.5, 15.0
    # Row 1: GT, FP32, NPU
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(gt, cmap='magma', vmin=vmin, vmax=vmax)
    ax1.set_title('GT Depth')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(fp32_depth, cmap='magma', vmin=vmin, vmax=vmax)
    ax2.set_title('FP32 Depth')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(npu_depth, cmap='magma', vmin=vmin, vmax=vmax)
    ax3.set_title('NPU Depth')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Row 2: GT-FP32, GT-NPU, FP32-NPU
    d1 = np.abs(gt - fp32_depth)
    d2 = np.abs(gt - npu_depth)
    d3 = np.abs(fp32_depth - npu_depth)
    vmin_d = 0.0
    vmax_d = max(np.nanmax(d1), np.nanmax(d2), np.nanmax(d3))

    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(d1, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    ax4.set_title('GT - FP32 (abs)')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(d2, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    ax5.set_title('GT - NPU (abs)')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(d3, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    ax6.set_title('FP32 - NPU (abs)')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # Metrics text – include more metrics: abs_rel, sq_rel, rmse, rmse_log, a1,a2,a3
    def fmt(metrics_):
        keys = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        return ', '.join([f"{k}: {metrics_.get(k, 0):.4f}" for k in keys])
    def deltas(fp, nu):
        keys = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        return ', '.join([f"{k}: {nu.get(k,0)-fp.get(k,0):+.4f}" for k in keys])

    text = f"FP32: {fmt(fp32_metrics)}\nNPU : {fmt(npu_metrics)}\nDelta: {deltas(fp32_metrics, npu_metrics)}"
    plt.suptitle(title if title else '', fontsize=14)
    plt.figtext(0.5, 0.01, text, ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_thumbnail(image_path: Path, thumb_path: Path, size=(512, 320)):
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    im = PILImage.open(image_path)
    im.thumbnail(size)
    im.save(thumb_path)


def write_index_html(out_dir: Path, categories: dict, index_path: Path, metadata: dict = None):
    # categories: {category_name: [file1, file2, ...]}
    html_lines = ["<html><head><meta charset='utf-8'><title>FP32 vs NPU Visualizations</title></head><body>"]
    html_lines.append("<h1>FP32 vs NPU Visualizations</h1>")
    for cat, imgs in categories.items():
        html_lines.append(f"<h2>{cat}</h2>")
        html_lines.append("<div style='display:flex;flex-wrap:wrap'>")
        for p in imgs:
            thumb = Path(p).name
            # assume thumbnail located in same folder and prefixed with 'thumb_'
            thumb_name = 'thumb_' + thumb
            # add metrics if provided
            m = metadata.get(p) if metadata else None
            metrics_html = ''
            if m:
                fp = m.get('fp32_metrics', {})
                nu = m.get('npu_metrics', {})
                metrics_html = f"<div style='font-size:11px;line-height:1.2em'>FP32 abs_rel: {fp.get('abs_rel',0):.4f}, RMSE: {fp.get('rmse',0):.3f}<br>NPU abs_rel: {nu.get('abs_rel',0):.4f}, RMSE: {nu.get('rmse',0):.3f}</div>"
            html_lines.append(
                f"<div style='margin:10px;display:flex;flex-direction:column;align-items:center'>"
                f"<a href='{p}' target='_blank'><img src='{Path(p).parent.name}/{thumb_name}' style='max-width:240px;'/></a>"
                f"<div style='font-size:12px'>{Path(p).name}</div>"
                f"{metrics_html}"
                f"</div>"
            )
        html_lines.append("</div>")
    html_lines.append("</body></html>")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'w') as f:
        f.write('\n'.join(html_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp32_dir', required=True, help='FP32 outputs dir (separated layout)')
    parser.add_argument('--fp32_model_name', required=True)
    parser.add_argument('--npu_dir', required=True, help='NPU outputs dir (integer_sigmoid, fractional_sigmoid)')
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--comp_json', default='outputs/npu_vs_fp32_component_and_gt_eval.json', help='Comparison JSON (contains top lists)')
    parser.add_argument('--top_types', default='npu', help='Comma-separated top types: npu,fp32,component,overlap,union,list')
    parser.add_argument('--images', default=None, help='Comma separated list of image IDs if top_type==list')
    parser.add_argument('--n', type=int, default=6, help='Top N images to visualize')
    parser.add_argument('--out_dir', default='outputs/visualizations/fp32_vs_npu', help='Output dir for PNGs')
    parser.add_argument('--dual_head_max_depth', type=float, default=15.0)
    args = parser.parse_args()

    test_entries = read_test_file(args.test_file)
    test_map = {e['filename']: e['dataset_root_override'] for e in test_entries}

    # Load comparison JSON to get top lists
    comp = json.load(open(args.comp_json))
    top_types = [t.strip() for t in args.top_types.split(',') if t.strip()]
    top_lists = {}
    for t in top_types:
        if t == 'npu':
            top_lists['npu'] = comp.get('top10_npu_abs_rel', [])[:args.n]
        elif t == 'fp32':
            top_lists['fp32'] = comp.get('top10_fp32_abs_rel', [])[:args.n]
        elif t == 'component':
            top_lists['component'] = comp.get('top10_component_depth_mean', [])[:args.n]
        elif t == 'overlap':
            # overlap between component and npu
            comp_list = set(comp.get('top10_component_depth_mean', []))
            npu_list = set(comp.get('top10_npu_abs_rel', []))
            overlap = list(comp_list.intersection(npu_list))[:args.n]
            top_lists['overlap'] = overlap
        elif t == 'union':
            union = list(set(comp.get('top10_component_depth_mean', [])).union(set(comp.get('top10_npu_abs_rel', []))).union(set(comp.get('top10_fp32_abs_rel', []))))[:args.n]
            top_lists['union'] = union
        elif t == 'list':
            if args.images:
                top_lists['list'] = [i.strip() for i in args.images.split(',') if i.strip()][:args.n]
            else:
                top_lists['list'] = []
        else:
            top_lists[t] = comp.get('top10_npu_abs_rel', [])[:args.n]

    # top_lists already sliced above; no need to re-slice

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = {}
    metadata = {}
    for cat, top_list in top_lists.items():
        categories[cat] = []
        for idx, image_id in enumerate(top_list, 1):
            ds_root_override = test_map.get(image_id)
            gt_depth = load_gt_depth(args.dataset_root, image_id, ds_root_override)
            if gt_depth is None:
                print(f'GT not found for {image_id}, skipping')
                continue

            fp32_depth, fp32_i, fp32_f = load_fp32_components(Path(args.fp32_dir), args.fp32_model_name, image_id)
            if fp32_depth is None and (fp32_i is None or fp32_f is None):
                print(f'FP32 depth not found for {image_id}, skipping')
                continue
            if fp32_depth is None:
                fp32_depth = compose_depth_from_components(fp32_i, fp32_f, args.dual_head_max_depth)

            npu_i, npu_f = load_npu_components(Path(args.npu_dir), image_id)
            if npu_i is None or npu_f is None:
                print(f'NPU outputs not found for {image_id}, skipping')
                continue
            npu_depth = compose_depth_from_components(npu_i, npu_f, args.dual_head_max_depth)

            # Compute metrics
            fp32_metrics = compute_metrics(gt_depth, fp32_depth)
            npu_metrics = compute_metrics(gt_depth, npu_depth)
            # ensure metrics present
            if fp32_metrics is None:
                fp32_metrics = {'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'rmse_log': 0.0, 'a1': 0.0, 'a2': 0.0, 'a3': 0.0}
            if npu_metrics is None:
                npu_metrics = {'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'rmse_log': 0.0, 'a1': 0.0, 'a2': 0.0, 'a3': 0.0}
            out_path = out_dir / f'{image_id}_fp32_vs_npu_vs_gt.png'
            title = f'{image_id} | FP32 vs NPU vs GT'
            plot_comparison(gt_depth, fp32_depth, npu_depth, fp32_metrics, npu_metrics, out_path, title)
            print(f'Saved: {out_path}')
            categories[cat].append(str(out_path))
            # collect metadata
            metadata[str(out_path)] = {
                'image_id': image_id,
                'fp32_metrics': fp32_metrics,
                'npu_metrics': npu_metrics,
                'gt_path': ds_root_override
            }
            # Save thumbnail
            thumb_path = out_dir / ('thumb_' + out_path.name)
            make_thumbnail(out_path, thumb_path)

    # Write HTML index
    index_path = out_dir / 'index.html'
    write_index_html(out_dir, categories, index_path, metadata)
    print('Index written to:', index_path)
    print('Done')


if __name__ == '__main__':
    main()
