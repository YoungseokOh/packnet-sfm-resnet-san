#!/usr/bin/env python3
"""
Compare Dual-Head FP32 vs NPU per-component (integer & fractional)

Usage (example):
    python scripts/compare_dual_head_components.py \
        --fp32_dir outputs/ncdb_test_fp32_full_separated \
        --fp32_model_name resnetsan01_fp32 \
        --npu_dir outputs/resnetsan_dual_head_seperate_static \
        --test_file /workspace/data/ncdb-cls-640x384/splits/combined_test.json \
        --output_json outputs/npu_vs_fp32_component_comparison.json

This tool compares integer/fractional components exactly (raw) and after 1/256 quantized rounding,
and gives per-image and aggregated statistics.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image


def read_test_file(test_file_path):
    test_file_path = Path(test_file_path)
    # JSON (NCDB style)
    if test_file_path.suffix == '.json':
        import json
        with open(test_file_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError('JSON test file must be a list')
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
                results.append({'filename': filename, 'dataset_root_override': dataset_root_override})
        return results
    # TXT (KITTI style)
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
            filename = parts[-1] if len(parts) >= 2 else line
        results.append({'filename': filename, 'dataset_root_override': None})
    return results


def load_npy_file(path: Path):
    if not path.exists():
        return None
    arr = np.load(path)
    # Collapse leading channel if present
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
        arr = arr[0, 0]
    return arr


def load_gt_depth(dataset_root, filename, dataset_root_override=None):
    base_path = Path(dataset_root_override) if dataset_root_override else Path(dataset_root)
    depth_path = base_path / 'newest_depth_maps' / f'{filename}.png'
    if not depth_path.exists():
        depth_path = Path(dataset_root) / 'depth_selection' / 'val_selection_cropped' / 'groundtruth_depth' / f'{filename}.png'
    if not depth_path.exists():
        return None
    depth_png = np.array(Image.open(depth_path), dtype=np.uint16)
    gt_depth = depth_png.astype(np.float32) / 256.0
    return gt_depth


def load_fp32_components(fp32_dir: Path, model_name: str, filename: str, precision: str = 'fp32'):
    # Preferred layout: fp32_dir/<model_name>/integer_fp32/<file>.npy and fractional_fp32
    integer_path = fp32_dir / model_name / f'integer_{precision}' / f'{filename}.npy'
    fractional_path = fp32_dir / model_name / f'fractional_{precision}' / f'{filename}.npy'
    if integer_path.exists() and fractional_path.exists():
        i = load_npy_file(integer_path)
        f = load_npy_file(fractional_path)
        return i, f
    # Fallback: try npz file containing integer_sigmoid/fractional_sigmoid
    npz_path = fp32_dir / f'{filename}.npz'
    if npz_path.exists():
        data = np.load(npz_path)
        i = data.get('integer_sigmoid', data.get('integer'))
        f = data.get('fractional_sigmoid', data.get('fractional'))
        if i is not None and f is not None:
            if i.ndim == 4:
                i = i[0, 0]
                f = f[0, 0]
            elif i.ndim == 3:
                i = i[0]
                f = f[0]
            return i, f
    # Not found
    return None, None


def load_npu_components(npu_dir: Path, filename: str, integer_dir: Path = None, fractional_dir: Path = None):
    # If explicit dirs provided
    if integer_dir and fractional_dir:
        i_path = integer_dir / f'{filename}.npy'
        f_path = fractional_dir / f'{filename}.npy'
        if i_path.exists() and f_path.exists():
            return load_npy_file(i_path), load_npy_file(f_path)
    # Default layout: npu_dir/integer_sigmoid/<file>.npy and fractional_sigmoid
    i_path = npu_dir / 'integer_sigmoid' / f'{filename}.npy'
    f_path = npu_dir / 'fractional_sigmoid' / f'{filename}.npy'
    if i_path.exists() and f_path.exists():
        return load_npy_file(i_path), load_npy_file(f_path)
    # Not found
    return None, None


def stats_from_diffs(diff):
    return {
        'mean': float(np.mean(diff)),
        'median': float(np.median(diff)),
        'std': float(np.std(diff)),
        'max': float(np.max(diff)),
        'p90': float(np.percentile(diff, 90)),
        'p99': float(np.percentile(diff, 99)),
    }


def compute_depth_metrics(gt, pred, min_depth=1e-3, max_depth=80.0):
    # Drop invalids and compute metrics; also return valid pixel count
    mask = (gt >= min_depth) & (gt <= max_depth)
    valid = int(np.sum(mask))
    if valid == 0:
        return None, 0
    gt_m = gt[mask]
    pred_m = pred[mask]
    thresh = np.maximum((gt_m / pred_m), (pred_m / gt_m))
    a1 = float((thresh < 1.25).mean())
    a2 = float((thresh < 1.25 ** 2).mean())
    a3 = float((thresh < 1.25 ** 3).mean())
    abs_rel = float(np.mean(np.abs(gt_m - pred_m) / gt_m))
    sq_rel = float(np.mean(((gt_m - pred_m) ** 2) / gt_m))
    rmse = float(np.sqrt(np.mean((gt_m - pred_m) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(gt_m) - np.log(pred_m)) ** 2)))
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }, valid


def compare_components(fp32_i, fp32_f, npu_i, npu_f, dual_head_max_depth=15.0):
    # Ensure shapes match
    if fp32_i is None or fp32_f is None or npu_i is None or npu_f is None:
        return None
    # Squeeze leading dims
    if fp32_i.ndim == 3 and fp32_i.shape[0] == 1:
        fp32_i = fp32_i[0]
    if fp32_f.ndim == 3 and fp32_f.shape[0] == 1:
        fp32_f = fp32_f[0]
    if npu_i.ndim == 3 and npu_i.shape[0] == 1:
        npu_i = npu_i[0]
    if npu_f.ndim == 3 and npu_f.shape[0] == 1:
        npu_f = npu_f[0]
    if fp32_i.shape != npu_i.shape:
        raise ValueError(f'Shape mismatch: fp32_i {fp32_i.shape} vs npu_i {npu_i.shape}')
    # Differences
    diff_i = np.abs(fp32_i - npu_i)
    diff_f = np.abs(fp32_f - npu_f)
    # scaled/quantized difference: round to 1/256 steps
    fp32_i_scaled = np.round(fp32_i * 256).astype(np.int32)
    npu_i_scaled = np.round(npu_i * 256).astype(np.int32)
    fp32_f_scaled = np.round(fp32_f * 256).astype(np.int32)
    npu_f_scaled = np.round(npu_f * 256).astype(np.int32)
    diff_i_scaled = np.abs(fp32_i_scaled - npu_i_scaled)
    diff_f_scaled = np.abs(fp32_f_scaled - npu_f_scaled)
    # equality ratios
    equal_i_ratio = float((fp32_i_scaled == npu_i_scaled).mean())
    equal_f_ratio = float((fp32_f_scaled == npu_f_scaled).mean())
    # Compose depth and compute depth diff
    depth_fp32 = fp32_i * dual_head_max_depth + fp32_f
    depth_npu = npu_i * dual_head_max_depth + npu_f
    depth_diff = np.abs(depth_fp32 - depth_npu)
    return {
        'integer': stats_from_diffs(diff_i),
        'fractional': stats_from_diffs(diff_f),
        'integer_scaled': stats_from_diffs(diff_i_scaled),
        'fractional_scaled': stats_from_diffs(diff_f_scaled),
        'integer_equal_ratio': equal_i_ratio,
        'fractional_equal_ratio': equal_f_ratio,
        'depth': stats_from_diffs(depth_diff),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp32_dir', type=str, required=True, help='FP32 outputs dir (separated layout)')
    parser.add_argument('--fp32_model_name', type=str, required=True)
    parser.add_argument('--fp32_precision', type=str, default='fp32', choices=['fp32', 'int8'])
    parser.add_argument('--npu_dir', type=str, required=True, help='NPU outputs dir (default layout: integer_sigmoid/, fractional_sigmoid/)')
    parser.add_argument('--npu_integer_dir', type=str, default=None, help='Optional explicit integer dir for NPU outputs')
    parser.add_argument('--npu_fractional_dir', type=str, default=None, help='Optional explicit fractional dir for NPU outputs')
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default=None, help='Dataset root (used to load GT newest_depth_maps)')
    parser.add_argument('--evaluate_vs_gt', action='store_true', help='Also evaluate composed depths vs GT metrics (abs_rel, rmse, etc)')
    parser.add_argument('--min_depth', type=float, default=0.5, help='Minimum depth for evaluation mask')
    parser.add_argument('--eval_max_depth', type=float, default=15.0, help='Maximum depth for evaluation mask')
    parser.add_argument('--dual_head_max_depth', type=float, default=15.0)
    parser.add_argument('--output_json', type=str, default='outputs/npu_vs_fp32_component_comparison.json')
    parser.add_argument('--limit', type=int, default=None, help='Limit to first N samples')
    args = parser.parse_args()

    fp32_dir = Path(args.fp32_dir)
    npu_dir = Path(args.npu_dir)
    npu_integer_dir = Path(args.npu_integer_dir) if args.npu_integer_dir else None
    npu_fractional_dir = Path(args.npu_fractional_dir) if args.npu_fractional_dir else None

    test_entries = read_test_file(args.test_file)
    if args.limit:
        test_entries = test_entries[:args.limit]

    per_image = []
    missing = 0
    for entry in tqdm(test_entries):
        filename = entry['filename']
        fp32_i, fp32_f = load_fp32_components(fp32_dir, args.fp32_model_name, filename, args.fp32_precision)
        npu_i, npu_f = load_npu_components(npu_dir, filename, npu_integer_dir, npu_fractional_dir)
        if fp32_i is None or fp32_f is None or npu_i is None or npu_f is None:
            missing += 1
            per_image.append({'image_id': filename, 'status': 'missing'})
            continue
        try:
            stats = compare_components(fp32_i, fp32_f, npu_i, npu_f, args.dual_head_max_depth)
        except Exception as e:
            per_image.append({'image_id': filename, 'status': 'error', 'error': str(e)})
            continue
        entry_out = {'image_id': filename, 'status': 'ok', 'stats': stats}
        # Also compute composed depth and GT metrics if requested
        if args.evaluate_vs_gt and args.dataset_root:
            gt_depth = load_gt_depth(args.dataset_root, filename, entry.get('dataset_root_override'))
            if gt_depth is not None:
                # compose
                fp32_depth = (fp32_i if fp32_i.ndim==2 else fp32_i[0]) * args.dual_head_max_depth + (fp32_f if fp32_f.ndim==2 else fp32_f[0])
                npu_depth = (npu_i if npu_i.ndim==2 else npu_i[0]) * args.dual_head_max_depth + (npu_f if npu_f.ndim==2 else npu_f[0])
                met_fp32, valid_fp32 = compute_depth_metrics(gt_depth, fp32_depth, args.min_depth, args.eval_max_depth)
                met_npu, valid_npu = compute_depth_metrics(gt_depth, npu_depth, args.min_depth, args.eval_max_depth)
                entry_out['metrics_fp32_vs_gt'] = met_fp32
                entry_out['metrics_npu_vs_gt'] = met_npu
                entry_out['valid_pixels'] = int(valid_fp32)

        per_image.append(entry_out)

    # aggregate
    valid_entries = [x for x in per_image if x.get('status') == 'ok']
    # compute mean of depth mean values etc by averaging per-image means (simple)
    agg = {}
    if len(valid_entries) > 0:
        def agg_key(key_path):
            vals = []
            for e in valid_entries:
                s = e['stats']
                # fold on depth or integer.*field
                # key_path example: 'integer.mean' or 'fractional_scaled.max'
                parts = key_path.split('.')
                current = s
                for p in parts:
                    current = current[p]
                vals.append(current)
            return float(np.mean(vals))
        # some default keys
        keys_to_agg = [
            'integer.mean', 'integer.median', 'integer.std', 'integer.max',
            'fractional.mean', 'fractional.median', 'fractional.std', 'fractional.max',
            'integer_scaled.mean', 'integer_scaled.median', 'integer_scaled.max',
            'fractional_scaled.mean', 'fractional_scaled.median', 'fractional_scaled.max',
            'integer_equal_ratio', 'fractional_equal_ratio',
            'depth.mean', 'depth.median', 'depth.std', 'depth.max'
        ]
        for k in keys_to_agg:
            agg[k] = agg_key(k)
    # If GT evaluation requested, aggregate metrics for FP32 & NPU vs GT
    agg_fp32 = None
    agg_npu = None
    if args.evaluate_vs_gt and args.dataset_root:
        # pixel-weighted aggregation
        total_valid = 0
        sums_fp32 = {}
        sums_npu = {}
        for e in valid_entries:
            valid = int(e.get('valid_pixels', 0))
            if valid == 0:
                continue
            total_valid += valid
            met_fp32 = e.get('metrics_fp32_vs_gt')
            met_npu = e.get('metrics_npu_vs_gt')
            if met_fp32 is None or met_npu is None:
                continue
            for k, v in met_fp32.items():
                sums_fp32[k] = sums_fp32.get(k, 0.0) + v * valid
            for k, v in met_npu.items():
                sums_npu[k] = sums_npu.get(k, 0.0) + v * valid
        if total_valid > 0:
            agg_fp32 = {k: sums_fp32[k] / total_valid for k in sums_fp32}
            agg_npu = {k: sums_npu[k] / total_valid for k in sums_npu}
        else:
            agg_fp32 = None
            agg_npu = None

    # derive Top-10 lists and overlaps
    # Top-10 component depth diff by mean
    top10_comp_by_depth = [e['image_id'] for e in sorted(valid_entries, key=lambda x: x['stats']['depth']['mean'], reverse=True)[:10]]
    top10_dep_depth_mean = [e['image_id'] for e in sorted(valid_entries, key=lambda x: x.get('metrics_npu_vs_gt', {'abs_rel': 0}).get('abs_rel', 0) - x.get('metrics_fp32_vs_gt', {'abs_rel': 0}).get('abs_rel', 0), reverse=True)[:10]]
    # Top-10 NPU worst by abs_rel
    top10_npu_abs_rel = [e['image_id'] for e in sorted(valid_entries, key=lambda x: x.get('metrics_npu_vs_gt', {'abs_rel': 0}).get('abs_rel', 0), reverse=True)[:10]]
    # Top-10 FP32 worst by abs_rel
    top10_fp32_abs_rel = [e['image_id'] for e in sorted(valid_entries, key=lambda x: x.get('metrics_fp32_vs_gt', {'abs_rel': 0}).get('abs_rel', 0), reverse=True)[:10]]
    overlap_comp_vs_depth = list(set(top10_comp_by_depth).intersection(set(top10_dep_depth_mean)))
    overlap_comp_vs_npu = list(set(top10_comp_by_depth).intersection(set(top10_npu_abs_rel)))
    results = {
        'fp32_dir': str(fp32_dir), 'fp32_model_name': args.fp32_model_name,
        'npu_dir': str(npu_dir), 'npu_integer_dir': str(npu_integer_dir) if npu_integer_dir else None,
        'npu_fractional_dir': str(npu_fractional_dir) if npu_fractional_dir else None,
        'num_test_entries': len(test_entries), 'num_missing': missing, 'num_valid': len(valid_entries),
        'aggregated': agg, 'per_image': per_image
    }
    if agg_fp32 is not None:
        results['aggregated_fp32_vs_gt'] = agg_fp32
    if agg_npu is not None:
        results['aggregated_npu_vs_gt'] = agg_npu
    results['top10_component_depth_mean'] = top10_comp_by_depth
    results['top10_npu_abs_rel'] = top10_npu_abs_rel
    results['top10_fp32_abs_rel'] = top10_fp32_abs_rel
    results['top10_composed_abs_rel_delta'] = top10_dep_depth_mean
    results['overlap_comp_vs_depth'] = overlap_comp_vs_depth
    results['overlap_comp_vs_npu'] = overlap_comp_vs_npu
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results written to: {args.output_json}')


if __name__ == '__main__':
    main()
