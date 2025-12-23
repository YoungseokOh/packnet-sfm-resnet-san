#!/usr/bin/env python3
"""Copy test images referenced in a split JSON into a flat folder.

The split JSON items are expected like:
  {"dataset_root": "/path/to/loop_x_640x384_newest", "new_filename": "..."}

This script copies:
  <dataset_root>/image_a6/<new_filename>.jpg (or .png)
into:
  <output_dir>/images/

It writes a manifest.json with resolved source->dest mappings.
"""

import argparse
import json
import shutil
from pathlib import Path


def find_image_path(dataset_root: Path, stem: str):
    img_dir = dataset_root / 'image_a6'
    for ext in ('.jpg', '.png', '.jpeg'):
        p = img_dir / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description='Copy test images from combined_test.json')
    ap.add_argument('--split', required=True, help='Path to combined_test.json')
    ap.add_argument('--output', required=True, help='Output root (will create images/)')
    ap.add_argument('--max', type=int, default=None, help='Optional max number of images to copy')
    args = ap.parse_args()

    split_path = Path(args.split)
    out_root = Path(args.output)
    out_images = out_root / 'images'
    out_images.mkdir(parents=True, exist_ok=True)

    data = json.loads(split_path.read_text(encoding='utf-8'))

    manifest = []
    copied = 0
    missing = 0

    for item in data:
        if args.max is not None and copied >= args.max:
            break

        dataset_root = Path(item['dataset_root'])
        stem = item['new_filename']

        src = find_image_path(dataset_root, stem)
        if src is None:
            missing += 1
            continue

        dst = out_images / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        manifest.append({'src': str(src), 'dst': str(dst)})
        copied += 1

    (out_root / 'manifest.json').write_text(json.dumps({
        'split': str(split_path),
        'output': str(out_root),
        'copied': copied,
        'missing': missing,
        'items': manifest,
    }, indent=2), encoding='utf-8')

    print(f'✅ Copied: {copied}')
    print(f'⚠️ Missing: {missing}')
    print(f'📁 Output: {out_root}')


if __name__ == '__main__':
    main()
