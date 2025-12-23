# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch
from PIL import Image
from glob import glob
import cv2
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.networks.layers.resnet.layers import dual_head_to_inv_depth

def is_image(file, ext=('.png', '.jpg', '.jpeg')):
    """Check if a file is an image with certain extensions"""
    return file.lower().endswith(ext)

def parse_args():
    parser = argparse.ArgumentParser(description='NCDB Dual-Head Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, required=True, help='Input file or folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=[384, 640],
                        help='Input image shape (H, W)')
    parser.add_argument('--save_depth', action='store_true', help='Save raw depth as .png')
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    
    # Initialize horovod
    hvd_init()

    # Parse checkpoint
    config, state_dict = parse_test_file(args.checkpoint)
    
    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, load_datasets=False)
    model_wrapper.load_state_dict(state_dict)
    
    # Send to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_wrapper = model_wrapper.to(device)
    model_wrapper.eval()

    # Get max_depth from model or config
    max_depth = 15.0
    if hasattr(model_wrapper.model, 'max_depth'):
        max_depth = model_wrapper.model.max_depth
    elif hasattr(config.model, 'params') and hasattr(config.model.params, 'max_depth'):
        max_depth = config.model.params.max_depth
    
    print0(f"Using max_depth: {max_depth}m")

    # Prepare input files
    if os.path.isdir(args.input):
        files = []
        for ext in ['png', 'jpg', 'jpeg']:
            files.extend(glob(os.path.join(args.input, f'*.{ext}')))
        files.sort()
    else:
        files = [args.input]
    
    print0(f'Found {len(files)} files')
    os.makedirs(args.output, exist_ok=True)

    image_shape = tuple(args.image_shape)

    for fn in files[rank()::world_size()]:
        # Load and resize image
        image = load_image(fn)
        image_resized = resize_image(image, image_shape)
        image_tensor = to_tensor(image_resized).unsqueeze(0).to(device)

        # Inference
        output = model_wrapper.depth(image_tensor)
        
        # Handle Dual-Head vs Single-Head
        if ('integer', 0) in output:
            # Dual-Head
            integer_sigmoid = output[('integer', 0)]
            fractional_sigmoid = output[('fractional', 0)]
            pred_inv_depth = dual_head_to_inv_depth(integer_sigmoid, fractional_sigmoid, max_depth)
        else:
            # Single-Head
            pred_inv_depth = output['inv_depths'][0]

        # Visualization
        viz = (viz_inv_depth(pred_inv_depth[0]) * 255).astype(np.uint8)
        
        # RGB for concat
        rgb = (np.array(image_resized)).astype(np.uint8)
        
        # Concat vertically
        concat = np.concatenate([rgb, viz], axis=0)
        
        # Save
        basename = os.path.basename(fn)
        output_fn = os.path.join(args.output, f'viz_{basename}')
        cv2.imwrite(output_fn, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
        
        if args.save_depth:
            depth_fn = os.path.join(args.output, f'depth_{os.path.splitext(basename)[0]}.png')
            write_depth(depth_fn, inv2depth(pred_inv_depth))
            
        print(f'Saved {output_fn}')

if __name__ == '__main__':
    main()
