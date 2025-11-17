#!/usr/bin/env python3
"""
Convert FP32 predicted depths saved as .npy to 16-bit PNG images under specified output directory.
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def npy_to_png(npy_path: Path, out_path: Path):
    arr = np.load(npy_path)
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uint16 = (arr * 256.0).astype(np.uint16)
    Image.fromarray(uint16).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_dir', default='Fin_Test_Set_ncdb/fp32')
    parser.add_argument('--out_dir', default='Fin_Test_Set_ncdb/fp32_png')
    args = parser.parse_args()
    pdir = Path(args.npy_dir)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for f in pdir.glob('*.npy'):
        outp = outdir / f'{f.stem}.png'
        try:
            npy_to_png(f, outp)
            converted += 1
        except Exception as e:
            print('Error converting', f, e)
    print('Converted', converted, 'files')


if __name__ == '__main__':
    main()
