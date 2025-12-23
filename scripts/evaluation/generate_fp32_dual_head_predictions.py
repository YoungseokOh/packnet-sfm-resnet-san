#!/usr/bin/env python3
"""
Generate FP32 dual-head predictions and save as npy files
for evaluation using evaluate_dual_head.py

Usage:
    # From image directory
    python generate_fp32_dual_head_predictions.py \
        --checkpoint path/to/epoch=49.ckpt \
        --image_dir path/to/images \
        --output_dir outputs/fp32_val

    # From JSON split file (NCDB format)
    python generate_fp32_dual_head_predictions.py \
        --checkpoint path/to/epoch=49.ckpt \
        --split_file path/to/combined_val.json \
        --output_dir outputs/fp32_val
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug


def load_split_file(split_path):
    """
    Load JSON split file and return list of (image_path, filename) tuples
    
    Args:
        split_path: Path to JSON split file
    
    Returns:
        list of tuples: [(image_path, filename), ...]
    """
    with open(split_path, 'r') as f:
        data = json.load(f)
    
    entries = []
    for entry in data:
        dataset_root = entry.get('dataset_root')
        new_filename = entry.get('new_filename')
        
        if dataset_root and new_filename:
            # Find image in synced_data/image_* or images/ (try multiple extensions)
            image_path = None
            extensions = ['.png', '.jpg', '.jpeg']
            subdirs = ['image_a6', 'synced_data/image_a6', 'synced_data/images', 'images']
            
            for subdir in subdirs:
                for ext in extensions:
                    candidate = Path(dataset_root) / subdir / f"{new_filename}{ext}"
                    if candidate.exists():
                        image_path = str(candidate)
                        break
                if image_path:
                    break
            
            if image_path is None:
                # Try searching for the image
                for ext in extensions:
                    for img_dir in Path(dataset_root).rglob(f"{new_filename}{ext}"):
                        if 'depth' not in str(img_dir).lower():
                            image_path = str(img_dir)
                            break
                    if image_path:
                        break
            
            if image_path:
                entries.append((image_path, new_filename))
    
    return entries


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Generate FP32 dual-head predictions')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint (.ckpt)')
    parser.add_argument('--image_dir', type=str, default=None, help='RGB images directory')
    parser.add_argument('--split_file', type=str, default=None, help='JSON split file (NCDB format)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--image_shape', type=int, nargs='+', default=[384, 640],
                        help='Input image shape (H, W)')
    parser.add_argument('--num_samples', type=int, default=None, help='Limit samples')
    args = parser.parse_args()
    
    # Validate input
    if not args.image_dir and not args.split_file:
        parser.error("Either --image_dir or --split_file must be provided")
    
    # Initialize horovod
    hvd_init()
    
    # Parse checkpoint
    print0(f"Loading checkpoint: {args.checkpoint}")
    config, state_dict = parse_test_file(args.checkpoint)
    set_debug(config.debug)
    
    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, load_datasets=False)
    model_wrapper.load_state_dict(state_dict)
    
    # Send to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print0(f"Using device: {device}")
    model_wrapper = model_wrapper.to(device)
    model_wrapper.eval()
    
    # Get max_depth
    max_depth = 10.0
    if hasattr(model_wrapper.model, 'max_depth'):
        max_depth = model_wrapper.model.max_depth
    print0(f"Using max_depth: {max_depth}m")
    
    # Create output directories
    int_dir = Path(args.output_dir) / 'integer_sigmoid'
    frac_dir = Path(args.output_dir) / 'fractional_sigmoid'
    int_dir.mkdir(parents=True, exist_ok=True)
    frac_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files - either from split file or directory
    if args.split_file:
        print0(f"Loading split file: {args.split_file}")
        image_entries = load_split_file(args.split_file)
        print0(f"Found {len(image_entries)} images in split file")
    else:
        image_files = sorted([f for f in os.listdir(args.image_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_entries = [(os.path.join(args.image_dir, f), os.path.splitext(f)[0]) 
                         for f in image_files]
    
    if args.num_samples:
        image_entries = image_entries[:args.num_samples]
    
    print0(f'Processing {len(image_entries)} images')
    print0(f'Output: {args.output_dir}')
    
    image_shape = tuple(args.image_shape)
    
    for img_path, base_name in tqdm(image_entries, desc="Generating FP32 predictions"):
        # Load and preprocess
        image = load_image(img_path)
        image_resized = resize_image(image, image_shape)
        image_tensor = to_tensor(image_resized).unsqueeze(0).to(device)
        
        # Inference
        output = model_wrapper.depth(image_tensor)
        
        # Get dual-head outputs
        if ('integer', 0) in output:
            integer_sigmoid = output[('integer', 0)].cpu().numpy()  # [1, 1, H, W]
            fractional_sigmoid = output[('fractional', 0)].cpu().numpy()
        else:
            print0(f"Warning: {base_name} - not dual-head output")
            continue
        
        # Save (keeping shape [1, 1, H, W] to match NPU format)
        np.save(int_dir / f"{base_name}.npy", integer_sigmoid.astype(np.float32))
        np.save(frac_dir / f"{base_name}.npy", fractional_sigmoid.astype(np.float32))
    
    print0(f"\n✅ Generated {len(image_entries)} FP32 predictions")
    print0(f"   integer_sigmoid: {int_dir}")
    print0(f"   fractional_sigmoid: {frac_dir}")


if __name__ == '__main__':
    main()