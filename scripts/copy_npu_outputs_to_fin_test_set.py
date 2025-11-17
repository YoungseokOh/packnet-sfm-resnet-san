#!/usr/bin/env python3
"""
Copy NPU dual-head results (integer_sigmoid, fractional_sigmoid) into a Fin_Test_Set folder
and optionally create 16-bit PNG composed depths for visual inspection.

This script will:
  - Read the test split (combined_test.json) to identify sample list
  - Copy integer/fractional .npy from the source NPU dir to Fin_Test_Set_ncdb/npu/<component>/
  - Compose depth = integer * DUAL_HEAD_MAX_DEPTH + fractional and save as 16-bit PNG
  - Write a small manifest JSON listing copied files

Usage:
  python scripts/copy_npu_outputs_to_fin_test_set.py --npu_dir outputs/resnetsan_dual_head_seperate_static --test_file /workspace/data/ncdb-cls-640x384/splits/combined_test.json --out_root Fin_Test_Set_ncdb --dual_head_max_depth 15.0
"""

import argparse
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image


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
                results.append({'filename': filename, 'dataset_root_override': dataset_root_override})
        return results
    # fallback to text list
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


def save_depth_as_png(depth_arr: np.ndarray, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    depth_uint16 = (depth_arr * 256.0).astype(np.uint16)
    Image.fromarray(depth_uint16).save(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npu_dir', required=True)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--out_root', default='Fin_Test_Set_ncdb')
    parser.add_argument('--dual_head_max_depth', default=15.0, type=float)
    parser.add_argument('--n', default=None, type=int, help='Optional limit for number of samples')
    args = parser.parse_args()

    npu_dir = Path(args.npu_dir)
    test_file = Path(args.test_file)
    out_root = Path(args.out_root)
    out_npu = out_root / 'npu'
    integer_out = out_npu / 'integer_sigmoid'
    fractional_out = out_npu / 'fractional_sigmoid'
    png_out = out_npu / 'png'
    integer_out.mkdir(parents=True, exist_ok=True)
    fractional_out.mkdir(parents=True, exist_ok=True)
    png_out.mkdir(parents=True, exist_ok=True)

    # Read test entries
    entries = read_test_file(test_file)
    if args.n:
        entries = entries[:args.n]

    copied = 0
    missing = []
    manifest = []
    for e in entries:
        fname = e['filename']
        i_src = npu_dir / 'integer_sigmoid' / f'{fname}.npy'
        f_src = npu_dir / 'fractional_sigmoid' / f'{fname}.npy'
        if not i_src.exists() or not f_src.exists():
            missing.append((fname, 'missing NPU file'))
            continue
        i_dst = integer_out / f'{fname}.npy'
        f_dst = fractional_out / f'{fname}.npy'
        shutil.copyfile(i_src, i_dst)
        shutil.copyfile(f_src, f_dst)

        # compose and save png
        i_arr = np.load(i_src)
        f_arr = np.load(f_src)
        # squeeze if needed
        if i_arr.ndim == 4:
            i_arr = i_arr[0,0]
            f_arr = f_arr[0,0]
        elif i_arr.ndim == 3 and i_arr.shape[0] == 1:
            i_arr = i_arr[0]
            f_arr = f_arr[0]
        depth = i_arr * float(args.dual_head_max_depth) + f_arr
        save_depth_as_png(depth, png_out / f'{fname}.png')

        manifest.append({'image_id': fname, 'integer_src': str(i_src), 'fractional_src': str(f_src), 'integer_dst': str(i_dst), 'fractional_dst': str(f_dst), 'png': str(png_out / f'{fname}.png')})
        copied += 1

    print(f'Copied NPU outputs: {copied}, Missing: {len(missing)}')
    if missing:
        print('First missing:', missing[:10])

    # write manifest
    import json
    (out_npu / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Manifest written to', out_npu / 'manifest.json')


if __name__ == '__main__':
    main()
