"""
PyTorch Direct Depth Inference using checkpoint

checkpoints/resnetsan_direct_depth_05_15.ckpt 사용
기존 scripts/infer_ncdb.py 기반으로 작성
"""

import argparse
import numpy as np
import os
import torch
from pathlib import Path
import json
from tqdm import tqdm

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.depth import inv2depth

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Direct Depth Inference')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/resnetsan_direct_depth_05_15.ckpt',
                       help='Checkpoint (.ckpt)')
    parser.add_argument('--test_json', type=str,
                       default='/workspace/data/ncdb-cls-640x384/splits/combined_test.json',
                       help='Test split JSON')
    parser.add_argument('--output', type=str,
                       default='outputs/pytorch_fp32_direct_depth_inference',
                       help='Output folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=[384, 640],
                       help='Input image shape (H W)')
    parser.add_argument('--half', action='store_true', help='Use FP16')
    return parser.parse_args()


@torch.no_grad()
def infer_depth(image_path, model_wrapper, image_shape, device, dtype=None):
    """
    Single image inference
    
    Parameters
    ----------
    image_path : str
        Path to input image
    model_wrapper : ModelWrapper
        Model wrapper for inference
    image_shape : tuple
        (H, W) for resizing
    device : torch.device
        Device to run inference on
    dtype : torch.dtype
        Data type (None for FP32, torch.float16 for FP16)
    
    Returns
    -------
    depth : np.ndarray
        Predicted depth map (H, W)
    """
    # Load and preprocess image
    image = load_image(image_path)
    image = resize_image(image, image_shape)
    image_tensor = to_tensor(image).unsqueeze(0)  # Add batch dimension
    
    # Send to device
    image_tensor = image_tensor.to(device, dtype=dtype)
    
    # Run inference
    output = model_wrapper.depth(image_tensor)
    
    # Extract depth prediction
    # For direct depth mode, output should already be in depth space
    if 'depth' in output:
        # Direct depth output
        pred_depth = output['depth'][0]  # First scale
    elif 'inv_depths' in output:
        # Check if this is actually direct depth masquerading as inv_depth
        # (legacy naming from model_wrapper)
        pred_inv_depth = output['inv_depths'][0]  # First scale
        
        # If depth_output_mode is 'direct', this is already depth, not inverse
        if hasattr(model_wrapper.model.depth_net, 'depth_output_mode') and \
           model_wrapper.model.depth_net.depth_output_mode == 'direct':
            # Already in depth space, no conversion needed!
            pred_depth = pred_inv_depth
        else:
            # Traditional inverse depth, convert to depth
            pred_depth = inv2depth(pred_inv_depth)
    else:
        raise ValueError(f"Unknown output format. Keys: {output.keys()}")
    
    # Convert to numpy
    if torch.is_tensor(pred_depth):
        pred_depth = pred_depth.squeeze().detach().cpu().numpy()
    else:
        pred_depth = pred_depth.squeeze()
    
    return pred_depth


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🚀 PyTorch Direct Depth Inference (from checkpoint)")
    print("=" * 80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if args.half else None
    print(f"\n🔧 Device: {device}")
    print(f"   Precision: {'FP16' if args.half else 'FP32'}")
    
    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    config, state_dict = parse_test_file(args.checkpoint)
    
    # Initialize model wrapper
    print("🔧 Creating model wrapper...")
    model_wrapper = ModelWrapper(config, load_datasets=False)
    model_wrapper.load_state_dict(state_dict)
    model_wrapper.to(device, dtype=dtype)
    model_wrapper.eval()
    
    print(f"   Model: {type(model_wrapper.model).__name__}")
    if hasattr(model_wrapper.model, 'depth_net'):
        depth_net = model_wrapper.model.depth_net
        print(f"   Depth net: {type(depth_net).__name__}")
        if hasattr(depth_net, 'depth_output_mode'):
            print(f"   Depth output mode: {depth_net.depth_output_mode}")
    
    # Image shape
    image_shape = tuple(args.image_shape)
    print(f"   Image shape (H, W): {image_shape}")
    
    # Load test JSON
    print(f"\n📂 Loading test split: {args.test_json}")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)
    print(f"   Total samples: {len(test_data)}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Output directory: {output_dir}")
    
    # Run inference
    print(f"\n🔄 Running inference on {len(test_data)} images...")
    
    for entry in tqdm(test_data, desc="Inference"):
        new_filename = entry['new_filename']
        image_path = entry['image_path']
        
        # Infer depth
        depth_pred = infer_depth(image_path, model_wrapper, image_shape, device, dtype)
        
        # Save as .npy (matching NPU format)
        output_file = output_dir / f'{new_filename}.npy'
        np.save(output_file, depth_pred)
    
    print(f"\n✅ Inference complete!")
    print(f"   Total files: {len(list(output_dir.glob('*.npy')))}")
    
    # Sample statistics
    sample_files = list(output_dir.glob('*.npy'))
    if sample_files:
        sample = np.load(sample_files[0])
        print(f"\n📊 Sample output stats:")
        print(f"   File: {sample_files[0].name}")
        print(f"   Shape: {sample.shape}")
        print(f"   Min depth: {sample.min():.3f}m")
        print(f"   Max depth: {sample.max():.3f}m")
        print(f"   Mean depth: {sample.mean():.3f}m")
        print(f"   Median depth: {np.median(sample):.3f}m")
    
    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
