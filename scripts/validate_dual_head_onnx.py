#!/usr/bin/env python3
"""
Quick validation script for Dual-Head ONNX models
Compares PyTorch and ONNX outputs to ensure they match
"""

import argparse
import torch
import numpy as np
import onnxruntime as ort
from collections import OrderedDict

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


def validate_onnx_model(onnx_path, checkpoint_path, num_samples=5):
    """Validate ONNX model against PyTorch model"""
    print(f"\n{'='*80}")
    print(f"Validating ONNX model: {onnx_path}")
    print(f"{'='*80}\n")
    
    # Load PyTorch model
    pytorch_model, max_depth = load_pytorch_model(checkpoint_path)
    
    # Load ONNX model
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input/output info
    print("\nONNX Model Info:")
    print(f"  Inputs: {[inp.name for inp in ort_session.get_inputs()]}")
    print(f"  Outputs: {[out.name for out in ort_session.get_outputs()]}")
    
    separate_outputs = len(ort_session.get_outputs()) > 1
    
    # Run comparison tests
    print(f"\nRunning {num_samples} comparison tests...")
    print(f"Max depth: {max_depth}m")
    print()
    
    max_int_error = 0.0
    max_frac_error = 0.0
    max_depth_error = 0.0
    
    int_errors = []
    frac_errors = []
    depth_errors = []
    
    for i in range(num_samples):
        # Create random input
        rgb_np = np.random.randn(1, 3, 384, 640).astype(np.float32)
        rgb_torch = torch.from_numpy(rgb_np)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_outputs = pytorch_model(rgb_torch)
            integer_sig_pt = pytorch_outputs[('integer', 0)].numpy()
            fractional_sig_pt = pytorch_outputs[('fractional', 0)].numpy()
            depth_composed_pt = integer_sig_pt * max_depth + fractional_sig_pt
        
        # ONNX inference
        onnx_outputs = ort_session.run(None, {'rgb': rgb_np})
        
        if separate_outputs:
            # Separate outputs: integer, fractional, composed
            integer_sig_onnx = onnx_outputs[0]
            fractional_sig_onnx = onnx_outputs[1]
            depth_composed_onnx = onnx_outputs[2]
            
            # Compare all three
            int_error = np.abs(integer_sig_pt - integer_sig_onnx).max()
            frac_error = np.abs(fractional_sig_pt - fractional_sig_onnx).max()
            depth_error = np.abs(depth_composed_pt - depth_composed_onnx).max()
            
            # Also compute composed depth from ONNX integer/fractional
            depth_manual_onnx = integer_sig_onnx * max_depth + fractional_sig_onnx
            composition_error = np.abs(depth_composed_onnx - depth_manual_onnx).max()
            
            max_int_error = max(max_int_error, int_error)
            max_frac_error = max(max_frac_error, frac_error)
            max_depth_error = max(max_depth_error, depth_error)
            
            int_errors.append(int_error)
            frac_errors.append(frac_error)
            depth_errors.append(depth_error)
            
            print(f"Sample {i+1}:")
            print(f"  PyTorch Integer:    [{integer_sig_pt.min():.6f}, {integer_sig_pt.max():.6f}]")
            print(f"  ONNX Integer:       [{integer_sig_onnx.min():.6f}, {integer_sig_onnx.max():.6f}]")
            print(f"  Integer Error:      {int_error:.9f}")
            print()
            print(f"  PyTorch Fractional: [{fractional_sig_pt.min():.6f}, {fractional_sig_pt.max():.6f}]")
            print(f"  ONNX Fractional:    [{fractional_sig_onnx.min():.6f}, {fractional_sig_onnx.max():.6f}]")
            print(f"  Fractional Error:   {frac_error:.9f}")
            print()
            print(f"  PyTorch Depth:      [{depth_composed_pt.min():.6f}m, {depth_composed_pt.max():.6f}m]")
            print(f"  ONNX Depth:         [{depth_composed_onnx.min():.6f}m, {depth_composed_onnx.max():.6f}m]")
            print(f"  Depth Error:        {depth_error:.9f}m")
            print(f"  Composition Check:  {composition_error:.9f}m (ONNX int*{max_depth}+frac == composed)")
            print()
            
        else:
            # Single output: composed depth only
            depth_composed_onnx = onnx_outputs[0]
            depth_error = np.abs(depth_composed_pt - depth_composed_onnx).max()
            
            max_depth_error = max(max_depth_error, depth_error)
            depth_errors.append(depth_error)
            
            print(f"Sample {i+1}:")
            print(f"  PyTorch Depth: [{depth_composed_pt.min():.6f}m, {depth_composed_pt.max():.6f}m]")
            print(f"  ONNX Depth:    [{depth_composed_onnx.min():.6f}m, {depth_composed_onnx.max():.6f}m]")
            print(f"  Depth Error:   {depth_error:.9f}m")
            print()
    
    # Summary
    print()
    print(f"{'='*80}")
    print("Validation Summary:")
    print(f"{'='*80}")
    
    if separate_outputs:
        print(f"  Integer sigmoid:")
        print(f"    Max error:  {max_int_error:.9f}")
        print(f"    Mean error: {np.mean(int_errors):.9f}")
        print(f"    Std error:  {np.std(int_errors):.9f}")
        print()
        print(f"  Fractional sigmoid:")
        print(f"    Max error:  {max_frac_error:.9f}")
        print(f"    Mean error: {np.mean(frac_errors):.9f}")
        print(f"    Std error:  {np.std(frac_errors):.9f}")
        print()
        print(f"  Composed depth:")
        print(f"    Max error:  {max_depth_error:.9f}m")
        print(f"    Mean error: {np.mean(depth_errors):.9f}m")
        print(f"    Std error:  {np.std(depth_errors):.9f}m")
        
        overall_max_error = max(max_int_error, max_frac_error, max_depth_error)
    else:
        print(f"  Composed depth only:")
        print(f"    Max error:  {max_depth_error:.9f}m")
        print(f"    Mean error: {np.mean(depth_errors):.9f}m")
        print(f"    Std error:  {np.std(depth_errors):.9f}m")
        
        overall_max_error = max_depth_error
    
    if overall_max_error < 1e-5:
        print(f"\n✅ EXCELLENT: ONNX model matches PyTorch exactly (error < 1e-5)")
    elif overall_max_error < 1e-3:
        print(f"\n✅ GOOD: ONNX model matches PyTorch closely (error < 1e-3)")
    elif overall_max_error < 1e-2:
        print(f"\n⚠️ ACCEPTABLE: ONNX model has small differences (error < 1e-2)")
    else:
        print(f"\n❌ WARNING: ONNX model has significant differences (error >= 1e-2)")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Validate Dual-Head ONNX models')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples to test')
    
    args = parser.parse_args()
    
    try:
        validate_onnx_model(args.onnx, args.checkpoint, args.num_samples)
        print("✅ Validation complete!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
