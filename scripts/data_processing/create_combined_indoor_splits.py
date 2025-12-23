#!/usr/bin/env python3
"""
Indoor loops combined split generator
- Scans multiple loop_*_640x384_newest folders under a root
- Collects image files from image_a6 (png/jpg)
- Produces train/val/test JSON splits (default 80/10/10)
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def scan_loop(loop_path: Path):
    """Scan one loop folder for image_a6 png/jpg files."""
    image_dir = loop_path / 'image_a6'
    if not image_dir.exists():
        print(f"⚠️  skip (no image_a6): {loop_path}")
        return []
    files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    samples = []
    for p in files:
        samples.append({
            "dataset_root": str(loop_path),
            "new_filename": p.stem
        })
    return samples

def create_combined_splits(loop_dirs, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    print("\n" + "="*60)
    print("📂 Scanning indoor loops...")
    print("="*60)

    all_samples = []
    for loop in tqdm(loop_dirs, desc="loops"):
        samples = scan_loop(Path(loop))
        print(f"  ✅ {Path(loop).name}: {len(samples):,} samples")
        all_samples.extend(samples)

    total = len(all_samples)
    print(f"Total samples: {total:,}")

    random.seed(seed)
    random.shuffle(all_samples)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = all_samples[:train_end]
    val_data = all_samples[train_end:val_end]
    test_data = all_samples[val_end:]

    splits = {
        'combined_train.json': train_data,
        'combined_val.json': val_data,
        'combined_test.json': test_data,
    }

    print("\n" + "="*60)
    print("💾 Writing splits...")
    print("="*60)
    for name, data in splits.items():
        out_path = output_dir / name
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        pct = (len(data) / total * 100) if total else 0
        print(f"  {name}: {len(data):,} ({pct:.1f}%) -> {out_path}")

    print("\nDone.")
    print(f"Output dir: {output_dir}")
    print(f"Train/Val/Test: {len(train_data):,} / {len(val_data):,} / {len(test_data):,}")
    print(f"Ratios: {train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f}")
    print(f"Seed: {seed}")

    return train_data, val_data, test_data

def main():
    parser = argparse.ArgumentParser(description='Create combined indoor splits from loop_* folders')
    parser.add_argument('--root', '-r', required=True, help='Root folder containing loop_* directories')
    parser.add_argument('--output', '-o', required=True, help='Output directory for splits')
    parser.add_argument('--ratio', nargs=3, type=float, default=[0.8, 0.1, 0.1], help='Train/val/test ratios')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    root = Path(args.root)
    loop_dirs = sorted([p for p in root.glob('loop_*_640x384_newest') if p.is_dir()])
    if not loop_dirs:
        raise SystemExit(f"No loop_*_640x384_newest found under {root}")

    create_combined_splits(
        loop_dirs=loop_dirs,
        output_dir=args.output,
        train_ratio=args.ratio[0],
        val_ratio=args.ratio[1],
        test_ratio=args.ratio[2],
        seed=args.seed,
    )

if __name__ == '__main__':
    main()
