#!/usr/bin/env python3
"""
Resize images in a directory to a specified size (default 640x384).

This script:
  - Walks a directory (recursively) and finds image files.
  - Reports current sizes (counts per size) before any change.
  - Optionally confirms with the user before proceeding.
  - Resizes images to the target size and saves them to an output directory
    preserving relative paths by default.
  - Optionally can overwrite images in-place with --inplace.
  - Verifies sizes after processing and prints a summary.

Usage examples:
  # Default, create a new directory with resized images
  python scripts/data_processing/resize_images.py --input-dir /path/to/images

  # Overwrite originals (in-place) - BE CAREFUL
  python scripts/data_processing/resize_images.py --input-dir /path/to/images --inplace --yes

  # Resize and place images in a specific output directory
  python scripts/data_processing/resize_images.py --input-dir /path/to/images --output-dir /path/to/resized --width 640 --height 384

Comments:
  - This script intentionally uses direct resize to the target width/height (no aspect-preserving crop)
    to match the existing camera scaling used elsewhere in this project (different x/y scale).
  - For transparent PNGs, alpha channel will be preserved.
  - For JPEGs, the script uses quality parameter (default 95).

"""

import argparse
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps
import os
from typing import List, Dict, Tuple

# Try to import tqdm for progress bars; fallback to identity iterator
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x

# Supported image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def is_image_file(path: Path) -> bool:
    """Return True if the file path looks like an image by extension."""
    return path.suffix.lower() in IMAGE_EXTS


def collect_image_files(root: Path, recursive: bool = True, exclude_root: Path = None) -> List[Path]:
    """Collect images under 'root' recursively (or non-recursively)."""
    files = []
    if recursive:
        for p in root.rglob('*'):
            if p.is_file() and is_image_file(p):
                # Optionally exclude a specific subtree (e.g., output dir under input)
                if exclude_root and exclude_root in p.parents:
                    continue
                files.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and is_image_file(p):
                if exclude_root and exclude_root in p.parents:
                    continue
                files.append(p)
    return sorted(files)


def analyze_sizes(files: List[Path], show_progress: bool = True) -> Dict[Tuple[int, int], int]:
    """Analyze sizes of a list of image files and return a counter of (w,h)->count.

    This function tries to avoid full image decoding during analysis by only
    opening images and reading their `.size`. For EXIF orientation-aware
    size detection, it checks EXIF orientation and swaps dimensions if needed
    for common rotated orientations (5,6,7,8).
    """
    counter = Counter()
    iter_files = tqdm(files, desc="Analyzing sizes", total=len(files)) if show_progress else files
    for f in iter_files:
        try:
            with Image.open(f) as im:
                # For analysis, don't call exif_transpose (which forces image load).
                # Instead, just check EXIF orientation and swap dimensions if needed.
                size = im.size
                try:
                    exif = im.getexif()
                    orientation = exif.get(274) if exif else None
                except Exception:
                    orientation = None
                if orientation in {5, 6, 7, 8}:
                    # These orientations imply a 90/270 degree rotation
                    size = (size[1], size[0])
                counter[size] += 1
        except Exception:
            counter[(0, 0)] += 1  # mark unreadable images
    return dict(counter)


def print_size_summary(counter: Dict[Tuple[int, int], int]) -> None:
    """Print a human-readable summary of sizes from the counter."""
    if not counter:
        print("No images found.")
        return
    total = sum(counter.values())
    print(f"Found {total:,} image(s), sizes:")
    for (w, h), count in sorted(counter.items(), key=lambda kv: kv[1], reverse=True):
        if (w, h) == (0, 0):
            print(f"  - unreadable/unsupported: {count}")
        else:
            print(f"  - {w}x{h}: {count}")


def create_output_path_for_file(src_file: Path, src_root: Path, output_root: Path, inplace: bool) -> Path:
    """Build the destination path for saving a processed file.

    If inplace=True, return the original path (overwrite). Otherwise, return
    a path under output_root with the same relative structure as under src_root.
    """
    if inplace:
        return src_file
    rel = src_file.relative_to(src_root)
    dest = output_root / rel
    return dest


def process_images(files: List[Path], src_root: Path, output_root: Path,
                   width: int, height: int, inplace: bool, quality: int = 95,
                   show_progress: bool = True) -> Tuple[int, int, List[Path]]:
    """Resize and save images to destination.

    Returns: (processed_count, skipped_count, failed_list)
    """
    processed = 0
    skipped = 0
    failed = []

    iter_files = tqdm(files, desc="Resizing images", total=len(files)) if show_progress else files
    for f in iter_files:
        dest = create_output_path_for_file(f, src_root, output_root, inplace)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(f) as im:
                # Apply EXIF orientation, if any (do this only for actual processing)
                im = ImageOps.exif_transpose(im)

                if im.size == (width, height):
                    # Already correct size
                    if inplace:
                        skipped += 1
                    else:
                        # Copy as-is to output root
                        im.save(dest, quality=quality)
                        skipped += 1
                    continue

                # Resize to target size (do not preserve aspect ratio by default)
                im_resized = im.resize((width, height), Image.LANCZOS)

                # Choose format and save parameters based on file suffix
                if dest.suffix.lower() in {'.jpg', '.jpeg'}:
                    im_resized = im_resized.convert('RGB')
                    im_resized.save(dest, 'JPEG', quality=quality)
                elif dest.suffix.lower() in {'.png'}:
                    im_resized.save(dest, 'PNG')
                else:
                    # For other formats, attempt to save in original format
                    fmt = im.format if im.format else 'PNG'
                    im_resized.save(dest, fmt)

            processed += 1
        except Exception as e:
            failed.append(f)
            print(f"⚠️ Failed to process {f}: {e}")

    return processed, skipped, failed


def verify_output_files(output_root: Path, width: int, height: int, show_progress: bool = True) -> Dict[str, int]:
    """Verify that all *_resized files in output_root have the expected size.

    Returns a summary: {"ok": count, "wrong_size": count, "unreadable": count}
    """
    summary = {"ok": 0, "wrong_size": 0, "unreadable": 0, "total": 0}
    files = collect_image_files(output_root, recursive=True)
    iter_files = tqdm(files, desc="Verifying files", total=len(files)) if show_progress else files
    for f in iter_files:
        summary['total'] += 1
        try:
            with Image.open(f) as im:
                im = ImageOps.exif_transpose(im)
                if im.size == (width, height):
                    summary['ok'] += 1
                else:
                    summary['wrong_size'] += 1
        except Exception:
            summary['unreadable'] += 1
    return summary


def parse_args():
    p = argparse.ArgumentParser(description='Resize a folder of images to a target size (default 640x384).')
    p.add_argument('--input-dir', '-i', required=True, type=Path,
                   help='Input directory (recursively searched)')
    p.add_argument('--output-dir', '-o', type=Path, default=None,
                   help='Output directory; if omitted a new folder named <input_dir>_resized will be created')
    p.add_argument('--width', type=int, default=640, help='Target width (default 640)')
    p.add_argument('--height', type=int, default=384, help='Target height (default 384)')
    p.add_argument('--inplace', action='store_true', help='Overwrite originals (in-place). CAUTION: This will destroy originals')
    p.add_argument('--yes', action='store_true', help='Skip confirmation prompts')
    p.add_argument('--quality', type=int, default=95, help='JPEG quality when saving (default 95)')
    p.add_argument('--recursive', action='store_true', default=True, help='Search directories recursively (default True)')
    p.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars (useful for non-interactive logs)')
    return p.parse_args()


def main():
    args = parse_args()

    input_root = args.input_dir
    if not input_root.exists() or not input_root.is_dir():
        print(f"❌ Input directory does not exist: {input_root}")
        return 1

    # Decide on output directory default
    if args.output_dir:
        output_root = args.output_dir
    else:
        # Default: <inputdir>_resized (same parent)
        output_root = input_root.parent / f"{input_root.name}_{args.width}x{args.height}"

    if args.inplace:
        # When in-place, output root is input root
        output_root = input_root

    # Collect images
    print(f"Scanning for images in: {input_root}")
    # Avoid scanning output directory if it's inside input_root (and not in-place)
    exclude_root = None
    if not args.inplace:
        try:
            # Only exclude if output_root is a sub-path of input_root
            if output_root.resolve().relative_to(input_root.resolve()):
                exclude_root = output_root
        except Exception:
            exclude_root = None
    files = collect_image_files(input_root, recursive=args.recursive, exclude_root=exclude_root)
    show_progress = not getattr(args, 'no_progress', False)
    print_size_summary(analyze_sizes(files, show_progress=show_progress))

    if not files:
        print("No image files found. Nothing to do.")
        return 0

    # Prompt for confirmation (unless --yes or inplace==False and output exists?)
    if not args.yes:
        print(f"Will resize {len(files)} image(s) to {args.width}x{args.height} and save to: {output_root}")
        if args.inplace:
            print("WARNING: You are about to overwrite your original files (in-place). This is destructive.")
        proceed = input('Proceed? (y/N): ').strip().lower()
        if proceed not in ('y', 'yes'):
            print('Aborting.')
            return 0

    # If not inplace and output exists, ensure not accidentally overwriting user
    if not args.inplace and output_root.exists():
        print(f"Output directory {output_root} already exists. Files may be overwritten there.")

    # If output dir doesn't exist, create it
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    # Process images
    print("\nProcessing images...")
    processed, skipped, failed = process_images(files, input_root, output_root, args.width, args.height, args.inplace, quality=args.quality, show_progress=show_progress)

    print(f"\nDone. Processed: {processed}, Skipped (already correct size): {skipped}, Failed: {len(failed)}")

    # Verify outputs
    print("\nVerifying output images...")
    summary = verify_output_files(output_root, args.width, args.height, show_progress=show_progress)
    print(f"Verification: total={summary['total']}, ok={summary['ok']}, wrong_size={summary['wrong_size']}, unreadable={summary['unreadable']}")

    if failed:
        print("Some files failed to process:")
        for f in failed[:20]:
            print(f"  - {f}")

    print("\nAll done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
