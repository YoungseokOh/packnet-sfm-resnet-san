#!/usr/bin/env python3
"""
Convert dual-head NPZ files (integer+fractional) to per-model separate dirs structure:
  <dst_dir>/<model_name>/integer_<precision>/<filename>.npy
  <dst_dir>/<model_name>/fractional_<precision>/<filename>.npy
  <dst_dir>/<model_name>/depth_<precision>/<filename>.npy

Example:
  python scripts/convert_npz_to_separate_dirs.py --src_dir outputs/ncdb_test_fp32_full --dst_dir outputs/ncdb_test_fp32_full_separated --model_name resnetsan01_fp32 --precision fp32
"""

import argparse
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Convert NPZ dual-head outputs to separate dirs')
    parser.add_argument('--src_dir', type=str, required=True, help='Source directory containing .npz files')
    parser.add_argument('--dst_dir', type=str, required=True, help='Destination base directory')
    parser.add_argument('--model_name', type=str, required=False, default=None, help='Model name to use for subdir (if not provided, use src_dir basename)')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'int8'], help='Precision tag for dirs')
    parser.add_argument('--max_depth', type=float, default=15.0, help='Max depth for composing (default: 15.0)')
    parser.add_argument('--pattern', type=str, default='*.npz', help='Glob pattern to select files (default: *.npz)')
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or Path(args.src_dir).stem

    integer_dir = dst / model_name / f'integer_{args.precision}'
    fractional_dir = dst / model_name / f'fractional_{args.precision}'
    depth_dir = dst / model_name / f'depth_{args.precision}'
    integer_dir.mkdir(parents=True, exist_ok=True)
    fractional_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(args.pattern))
    count = 0
    for f in files:
        try:
            data = np.load(f)
            if 'integer_sigmoid' in data and 'fractional_sigmoid' in data:
                integer = data['integer_sigmoid']
                fractional = data['fractional_sigmoid']
                # handle shapes
                if integer.ndim == 4:
                    integer = integer[0, 0]
                    fractional = fractional[0, 0]
                elif integer.ndim == 3:
                    integer = integer[0]
                    fractional = fractional[0]

                # Compose depth
                depth = integer * args.max_depth + fractional

                fname = f.stem
                np.save(integer_dir / f'{fname}.npy', integer)
                np.save(fractional_dir / f'{fname}.npy', fractional)
                np.save(depth_dir / f'{fname}.npy', depth)
                count += 1
        except Exception as e:
            print(f"Failed to convert {f}: {e}")
            continue

    print(f"Converted {count} files to {dst} using model_name='{model_name}', precision='{args.precision}'")


if __name__ == '__main__':
    main()
