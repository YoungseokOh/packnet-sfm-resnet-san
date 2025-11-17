#!/usr/bin/env python3
"""
Test ONNX models with real KITTI images
Compare PyTorch checkpoint vs ONNX outputs
"""

import argparse
import torch
import numpy as np
import onnxruntime as ort
from collections import OrderedDict
from PIL import Image
import os


def load_pytorch_model(checkpoint_path):
    """Load PyTorch Dual-Head model"""
    print("Loading PyTorch model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    
    depth_net = ResNetSAN01(
        dropout=config.model.depth_net.get('dropout', 0.5),
        version=config.model.depth_net.get('version', '18A'),
        use_dual_head=True,
        use_enhanced_lidar=False,
        use_film=False
    )
    
    # Load weights
    depth_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('model.depth_net.'):
            new_key = key.replace('model.depth_net.', '')
            if new_key not in ['weight', 'bias']:
                depth_state[new_key] = value
    
    depth_net.load_state_dict(depth_state, strict=False)
    depth_net.eval()
    
    max_depth = config.model.params.get('max_depth', 15.0)
    
    return depth_net, max_depth


def preprocess_image(image_path, target_size=(640, 384)):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy and normalize
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Convert HWC to CHW
    img_np = img_np.transpose(2, 0, 1)
    
    # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0)
    
    return img_np


def main():
    parser = argparse.ArgumentParser(description='Test ONNX with real images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--onnx_separate', type=str, required=True,
                        help='Path to separate outputs ONNX model')
    parser.add_argument('--onnx_composed', type=str, required=True,
                        help='Path to composed output ONNX model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--size', type=int, nargs=2, default=[640, 384],
                        help='Input size [width, height]')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Real Image ONNX Validation Test")
    print("="*80)
    print()
    
    # Load models
    pytorch_model, max_depth = load_pytorch_model(args.checkpoint)
    
    print(f"Loading ONNX models...")
    ort_separate = ort.InferenceSession(args.onnx_separate)
    ort_composed = ort.InferenceSession(args.onnx_composed)
    
    print(f"Max depth: {max_depth}m")
    print()
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    img_np = preprocess_image(args.image, tuple(args.size))
    img_torch = torch.from_numpy(img_np)
    
    print(f"Image shape: {img_np.shape}")
    print(f"Image range: [{img_np.min():.3f}, {img_np.max():.3f}]")
    print()
    
    # PyTorch inference
    print("Running PyTorch inference...")
    with torch.no_grad():
        pytorch_outputs = pytorch_model(img_torch)
        integer_sig_pt = pytorch_outputs[('integer', 0)].numpy()
        fractional_sig_pt = pytorch_outputs[('fractional', 0)].numpy()
        depth_composed_pt = integer_sig_pt * max_depth + fractional_sig_pt
    
    print(f"PyTorch results:")
    print(f"  Integer sigmoid:    [{integer_sig_pt.min():.6f}, {integer_sig_pt.max():.6f}]")
    print(f"  Fractional sigmoid: [{fractional_sig_pt.min():.6f}, {fractional_sig_pt.max():.6f}]")
    print(f"  Composed depth:     [{depth_composed_pt.min():.6f}m, {depth_composed_pt.max():.6f}m]")
    print()
    
    # ONNX separate inference
    print("Running ONNX (separate outputs) inference...")
    onnx_sep_outputs = ort_separate.run(None, {'rgb': img_np})
    integer_sig_onnx_sep = onnx_sep_outputs[0]
    fractional_sig_onnx_sep = onnx_sep_outputs[1]
    depth_composed_onnx_sep = onnx_sep_outputs[2]
    
    print(f"ONNX (separate) results:")
    print(f"  Integer sigmoid:    [{integer_sig_onnx_sep.min():.6f}, {integer_sig_onnx_sep.max():.6f}]")
    print(f"  Fractional sigmoid: [{fractional_sig_onnx_sep.min():.6f}, {fractional_sig_onnx_sep.max():.6f}]")
    print(f"  Composed depth:     [{depth_composed_onnx_sep.min():.6f}m, {depth_composed_onnx_sep.max():.6f}m]")
    print()
    
    # ONNX composed inference
    print("Running ONNX (composed only) inference...")
    onnx_comp_outputs = ort_composed.run(None, {'rgb': img_np})
    depth_composed_onnx_comp = onnx_comp_outputs[0]
    
    print(f"ONNX (composed) results:")
    print(f"  Composed depth:     [{depth_composed_onnx_comp.min():.6f}m, {depth_composed_onnx_comp.max():.6f}m]")
    print()
    
    # Compare results
    print("="*80)
    print("Comparison Results:")
    print("="*80)
    
    # PyTorch vs ONNX separate
    int_error = np.abs(integer_sig_pt - integer_sig_onnx_sep).max()
    frac_error = np.abs(fractional_sig_pt - fractional_sig_onnx_sep).max()
    depth_sep_error = np.abs(depth_composed_pt - depth_composed_onnx_sep).max()
    
    print(f"PyTorch vs ONNX (separate):")
    print(f"  Integer error:      {int_error:.9f}")
    print(f"  Fractional error:   {frac_error:.9f}")
    print(f"  Composed depth err: {depth_sep_error:.9f}m")
    print()
    
    # PyTorch vs ONNX composed
    depth_comp_error = np.abs(depth_composed_pt - depth_composed_onnx_comp).max()
    
    print(f"PyTorch vs ONNX (composed):")
    print(f"  Composed depth err: {depth_comp_error:.9f}m")
    print()
    
    # ONNX separate vs ONNX composed
    onnx_consistency_error = np.abs(depth_composed_onnx_sep - depth_composed_onnx_comp).max()
    
    print(f"ONNX (separate) vs ONNX (composed):")
    print(f"  Depth consistency:  {onnx_consistency_error:.9f}m")
    print()
    
    # Verify composition formula
    depth_manual = integer_sig_onnx_sep * max_depth + fractional_sig_onnx_sep
    composition_error = np.abs(depth_composed_onnx_sep - depth_manual).max()
    
    print(f"Composition formula verification:")
    print(f"  int*{max_depth}+frac == composed: {composition_error:.9f}m")
    print()
    
    # Final verdict
    print("="*80)
    print("Final Verdict:")
    print("="*80)
    
    all_errors = [int_error, frac_error, depth_sep_error, depth_comp_error, onnx_consistency_error]
    max_error = max(all_errors)
    
    if max_error < 1e-5:
        print("✅ EXCELLENT: All models match exactly (error < 1e-5)")
    elif max_error < 1e-3:
        print("✅ GOOD: All models match closely (error < 1e-3)")
    elif max_error < 1e-2:
        print("⚠️ ACCEPTABLE: Models have small differences (error < 1e-2)")
    else:
        print("❌ WARNING: Models have significant differences (error >= 1e-2)")
    
    print()
    print(f"Maximum error across all comparisons: {max_error:.9f}")
    print()


if __name__ == '__main__':
    main()
