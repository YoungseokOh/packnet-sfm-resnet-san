# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch

from glob import glob
from cv2 import imwrite

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape ' 
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    parser.add_argument('--mask_file', type=str, default=None,
                        help='Path to the binary mask file (e.g., /workspace/packnet-sfm/ncdb-cls/synced_data/binary_mask.png).')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) and is_image(args.output)) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args


import numpy as np
import os
import torch
from PIL import Image # Add PIL for mask loading

from glob import glob
from cv2 import imwrite

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape ' 
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    parser.add_argument('--mask_file', type=str, default=None,
                        help='Path to the binary mask file (e.g., /workspace/packnet-sfm/ncdb-cls/synced_data/binary_mask.png).')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) and is_image(args.output)) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args


@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, half, save):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image_tensor = to_tensor(image).unsqueeze(0) # Renamed to avoid conflict with numpy image

    # Load and apply mask if mask_file is provided in config
    mask = None
    if hasattr(model_wrapper.config.datasets, 'mask_file') and model_wrapper.config.datasets.mask_file:
        mask_path = model_wrapper.config.datasets.mask_file
        if os.path.exists(mask_path):
            mask = (np.array(Image.open(mask_path).convert('L')) > 0).astype(np.uint8)
            # Resize mask to image_shape if dimensions do not match
            if mask.shape[:2] != image_shape:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                mask_img = mask_img.resize((image_shape[1], image_shape[0]), Image.NEAREST)
                mask = np.array(mask_img)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() # Add channel and batch dim

            # Apply mask to image_tensor (element-wise multiplication)
            image_tensor = image_tensor * mask_tensor

    # Send image to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.to('cuda:{}'.format(rank()), dtype=dtype)
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)
        if mask is not None:
            mask_tensor = mask_tensor.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image_tensor)['inv_depths'][0]

    # Apply mask to predicted inverse depth if mask exists
    if mask is not None:
        pred_inv_depth = pred_inv_depth * mask_tensor[0, 0] # Apply mask to the inverse depth map

    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        write_depth(filename, depth=inv2depth(pred_inv_depth))
    else:
        # Prepare RGB image (original image, not masked for visualization)
        rgb = (image_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        # Prepare inverse depth
        viz_pred_inv_depth = (viz_inv_depth(pred_inv_depth[0]) * 255).astype(np.uint8)
        # Concatenate both vertically
        image_concat = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        imwrite(output_file, image_concat[:, :, ::-1])


def main(args):


    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Add mask_file to config if provided
    if args.mask_file:
        config.datasets.update({'mask_file': args.mask_file})

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    if os.path.isdir(args.input):
        # If input file is a folder, search for image files
        files = []
        for ext in ['png', 'jpg']:
            files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
        files.sort()
        print0('Found {} files'.format(len(files)))
    else:
        # Otherwise, use it as is
        files = [args.input]

    # Process each file
    for fn in files[rank()::world_size()]:
        if os.path.isdir(args.output):
            output_path = os.path.join(args.output, os.path.basename(fn))
            os.makedirs(args.output, exist_ok=True)
        else:
            output_path = args.output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        infer_and_save_depth(
            fn, output_path, model_wrapper, image_shape, args.half, args.save)


if __name__ == '__main__':
    args = parse_args()
    main(args)
