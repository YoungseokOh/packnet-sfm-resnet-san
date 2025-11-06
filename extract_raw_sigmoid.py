#!/usr/bin/env python3
"""
Inference script to extract raw sigmoid outputs from ResNetSAN model.

This script:
1. Loads checkpoint: checkpoints/resnetsan_linear_05_15.ckpt
2. Processes 5 test images from test_set_rgb/
3. Outputs raw sigmoid [0, 1] values (NO post-processing)
4. Saves as .npy files for analysis
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, '/workspace/packnet-sfm')

from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01


def load_model(checkpoint_path, device='cuda'):
    """
    Load ResNetSAN01 model from checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded model in eval mode
        config: Model configuration from checkpoint
    """
    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\n✅ Config found in checkpoint")
        # Extract depth params
        if hasattr(config, 'model') and hasattr(config.model, 'params'):
            min_depth = float(config.model.params.min_depth)
            max_depth = float(config.model.params.max_depth)
        else:
            min_depth = 0.5
            max_depth = 15.0
    else:
        print(f"\n⚠️  No config in checkpoint, using defaults")
        min_depth = 0.5
        max_depth = 15.0
        config = None
    
    print(f"   Depth range: [{min_depth}, {max_depth}]m")
    
    # Initialize model
    model = ResNetSAN01(
        version='18',
        use_film=False,
        use_enhanced_lidar=False,
        min_depth=min_depth,
        max_depth=max_depth
    )
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Filter keys (remove 'model.depth_net.' prefix if present)
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove various prefixes
        new_key = key
        for prefix in ['model.depth_net.', 'depth_net.', 'module.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_state_dict[new_key] = value
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"\n✅ Model loaded successfully")
    if missing_keys:
        # Filter out expected missing keys (training-only params)
        expected_missing = ['weight', 'bias', 'mconvs']
        actual_missing = [k for k in missing_keys if not any(em in k for em in expected_missing)]
        if actual_missing:
            print(f"   ⚠️  Missing keys: {actual_missing[:5]}")
    if unexpected_keys:
        print(f"   ⚠️  Unexpected keys: {unexpected_keys[:5]}")
    
    # Set to eval mode
    model = model.to(device)
    model.eval()
    
    return model, config


def load_image(image_path, target_size=(384, 640)):
    """
    Load and preprocess image for inference.
    
    Args:
        image_path: Path to image file
        target_size: (H, W) tuple
    
    Returns:
        tensor: Preprocessed image tensor [1, 3, H, W]
        original: Original PIL image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original = img.copy()
    
    # Resize to target size
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Convert to tensor and normalize (ImageNet stats)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    
    return tensor, original


def inference_single(model, image_tensor, device='cuda'):
    """
    Run inference on a single image.
    
    Args:
        model: ResNetSAN01 model
        image_tensor: [1, 3, H, W] tensor
        device: 'cuda' or 'cpu'
    
    Returns:
        sigmoid_output: [1, 1, H, W] tensor with values in [0, 1]
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        sigmoid_output = output['inv_depths'][0]  # First scale [1, 1, H, W]
    
    return sigmoid_output


def main():
    """Main inference pipeline."""
    
    # Configuration
    checkpoint_path = '/workspace/packnet-sfm/checkpoints/resnetsan_linear_05_15.ckpt'
    test_images_dir = '/workspace/data/ncdb-cls-640x384/test_set_rgb'
    output_dir = '/workspace/packnet-sfm/outputs/raw_sigmoid_outputs'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = 5
    
    print(f"\n{'='*80}")
    print(f"RAW SIGMOID OUTPUT EXTRACTION")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test images: {test_images_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Samples: {num_samples}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    
    # Get test images (first 5)
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.png')])[:num_samples]
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(image_files)} IMAGES")
    print(f"{'='*80}")
    
    # Process each image
    results = {}
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(test_images_dir, image_file)
        image_name = os.path.splitext(image_file)[0]
        
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_file}")
        
        # Load image
        image_tensor, original_img = load_image(image_path)
        print(f"   Image size: {original_img.size} → Tensor: {image_tensor.shape}")
        
        # Run inference
        sigmoid_output = inference_single(model, image_tensor, device)
        
        # Convert to numpy
        sigmoid_np = sigmoid_output.cpu().numpy()  # [1, 1, H, W]
        sigmoid_np = sigmoid_np.squeeze()  # [H, W]
        
        # Statistics
        print(f"   Sigmoid output shape: {sigmoid_np.shape}")
        print(f"   Value range: [{sigmoid_np.min():.6f}, {sigmoid_np.max():.6f}]")
        print(f"   Mean: {sigmoid_np.mean():.6f}, Std: {sigmoid_np.std():.6f}")
        print(f"   Median: {np.median(sigmoid_np):.6f}")
        
        # Save as .npy
        output_path = os.path.join(output_dir, f'{image_name}_sigmoid.npy')
        np.save(output_path, sigmoid_np)
        print(f"   ✅ Saved: {output_path}")
        
        # Store results
        results[image_name] = {
            'shape': sigmoid_np.shape,
            'min': float(sigmoid_np.min()),
            'max': float(sigmoid_np.max()),
            'mean': float(sigmoid_np.mean()),
            'std': float(sigmoid_np.std()),
            'median': float(np.median(sigmoid_np)),
            'file': output_path
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RAW SIGMOID OUTPUT SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Num samples: {num_samples}\n\n")
        
        for image_name, stats in results.items():
            f.write(f"\n{image_name}:\n")
            f.write(f"  Shape: {stats['shape']}\n")
            f.write(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
            f.write(f"  Mean: {stats['mean']:.6f}\n")
            f.write(f"  Std: {stats['std']:.6f}\n")
            f.write(f"  Median: {stats['median']:.6f}\n")
            f.write(f"  File: {stats['file']}\n")
    
    print(f"\n{'='*80}")
    print(f"✅ INFERENCE COMPLETE!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"\nFiles saved:")
    for image_name in results.keys():
        print(f"  - {image_name}_sigmoid.npy")
    print(f"\n💡 These are RAW sigmoid outputs [0, 1]")
    print(f"   NO post-processing applied (no Linear/Log transform)")
    print(f"   Use sigmoid_to_inv_depth() or sigmoid_to_depth_*() for depth conversion")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
