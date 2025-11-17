#!/usr/bin/env python3
# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Modified for Dual-Head ResNetSAN01 ONNX export

import argparse
import torch
import torch.onnx
import os
import numpy as np
import sys
from pathlib import Path
from collections import OrderedDict

# Import required modules but handle horovod dependency
try:
    import horovod.torch as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

from packnet_sfm.utils.load import set_debug


class DualHeadDepthNet(torch.nn.Module):
    """
    Wrapper for Dual-Head depth network
    Outputs integer and fractional sigmoid values separately
    """
    def __init__(self, resnet_san_model, max_depth=15.0):
        super().__init__()
        self.depth_net = resnet_san_model
        self.max_depth = max_depth
        
    def forward(self, rgb):
        """
        Forward pass for Dual-Head model
        Returns: integer_sigmoid, fractional_sigmoid, depth_composed
        """
        # Set eval mode to ensure single-scale output
        self.depth_net.eval()
        
        # Direct inference
        outputs = self.depth_net(rgb)
        
        # Extract Dual-Head outputs
        # outputs is dict with keys: ('integer', 0), ('fractional', 0)
        integer_sigmoid = outputs[('integer', 0)]  # [B, 1, H, W]
        fractional_sigmoid = outputs[('fractional', 0)]  # [B, 1, H, W]
        
        # Compose final depth
        depth_composed = integer_sigmoid * self.max_depth + fractional_sigmoid
        
        # Return all three outputs for flexibility
        return integer_sigmoid, fractional_sigmoid, depth_composed


class DualHeadDepthNetTwoOutputs(torch.nn.Module):
    """
    Wrapper that returns only integer and fractional sigmoid (for NPU)
    Composition is done outside the model
    """
    def __init__(self, resnet_san_model, max_depth=15.0):
        super().__init__()
        self.depth_net = resnet_san_model
        self.max_depth = max_depth
        
    def forward(self, rgb):
        """
        Forward pass returning only integer and fractional sigmoid
        """
        self.depth_net.eval()
        outputs = self.depth_net(rgb)
        
        integer_sigmoid = outputs[('integer', 0)]
        fractional_sigmoid = outputs[('fractional', 0)]
        
        # Return only two outputs (no composition)
        return integer_sigmoid, fractional_sigmoid


class DualHeadDepthNetSingleOutput(torch.nn.Module):
    """
    Wrapper that returns only composed depth (simpler ONNX graph)
    """
    def __init__(self, resnet_san_model, max_depth=15.0):
        super().__init__()
        self.depth_net = resnet_san_model
        self.max_depth = max_depth
        
    def forward(self, rgb):
        """
        Forward pass returning only composed depth
        """
        self.depth_net.eval()
        outputs = self.depth_net(rgb)
        
        integer_sigmoid = outputs[('integer', 0)]
        fractional_sigmoid = outputs[('fractional', 0)]
        
        # Compose final depth
        depth_composed = integer_sigmoid * self.max_depth + fractional_sigmoid
        
        return depth_composed


def parse_args():
    """Parse arguments for Dual-Head ONNX conversion"""
    parser = argparse.ArgumentParser(description='Dual-Head ResNetSAN01 to ONNX conversion')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Dual-Head model checkpoint (.ckpt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[384, 640],
                        help='Input image shape [height, width]')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--separate_outputs', action='store_true',
                        help='Export integer/fractional/composed as separate outputs (default: composed only)')
    parser.add_argument('--keep_reflection_pad', action='store_true',
                        help='Keep ReflectionPad2d (better quality, may not work with NNEF)')
    parser.add_argument('--max_depth', type=float, default=15.0,
                        help='Maximum depth value for composition (default: 15.0)')
    parser.add_argument('--dynamic_axes', action='store_true',
                        help='Enable dynamic batch size (default: fixed batch size for NPU)')
    
    args = parser.parse_args()
    
    # Validate checkpoint file
    assert args.checkpoint.endswith('.ckpt'), 'Checkpoint must be .ckpt'
    assert os.path.exists(args.checkpoint), f'Checkpoint not found: {args.checkpoint}'
    
    # Set default output path
    if args.output is None:
        base_name = os.path.basename(args.checkpoint).replace('.ckpt', '')
        pad_type = "reflect" if args.keep_reflection_pad else "zero"
        output_type = "separate" if args.separate_outputs else "composed"
        batch_type = "dynamic" if args.dynamic_axes else "static"
        args.output = f"onnx/dual_head_{base_name}_{args.input_shape[1]}x{args.input_shape[0]}_{output_type}_{pad_type}_{batch_type}.onnx"
    
    return args


def monkey_patch_horovod():
    """Simple horovod patching"""
    if not HAS_HOROVOD:
        from types import ModuleType
        dummy_hvd = ModuleType('horovod')
        dummy_hvd.rank = lambda: 0
        dummy_hvd.size = lambda: 1
        sys.modules['horovod'] = dummy_hvd
        sys.modules['horovod.torch'] = dummy_hvd


def load_dual_head_model(checkpoint_path):
    """Load Dual-Head model from checkpoint"""
    print(f"Loading Dual-Head model from: {checkpoint_path}")
    
    # Load checkpoint (remove weights_only for older PyTorch compatibility)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    
    # Verify this is a Dual-Head model
    use_dual_head = config.model.depth_net.get('use_dual_head', False)
    if not use_dual_head:
        print("⚠️ WARNING: Checkpoint does not have use_dual_head=true!")
        print("⚠️ This script is designed for Dual-Head models.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Set debug
    set_debug(config.debug)
    
    # Import and create the model
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    
    # Create Dual-Head depth network
    depth_net = ResNetSAN01(
        dropout=config.model.depth_net.get('dropout', 0.5),
        version=config.model.depth_net.get('version', '18A'),
        use_dual_head=True,  # Force Dual-Head
        use_enhanced_lidar=False,  # ONNX: disable MinkowskiEngine
        use_film=False  # ONNX: disable FiLM
    )
    
    # Extract max_depth from config
    max_depth = config.model.params.get('max_depth', 15.0)
    print(f"Max depth from config: {max_depth}m")
    
    # Extract and load state dict
    depth_state = OrderedDict()
    excluded_keys = []
    for key, value in state_dict.items():
        if key.startswith('model.depth_net.'):
            new_key = key.replace('model.depth_net.', '')
            
            # Exclude training-only parameters
            if new_key in ['weight', 'bias']:
                excluded_keys.append(new_key)
                continue
            
            depth_state[new_key] = value
    
    if excluded_keys:
        print(f"🗑️ Excluded training-only parameters: {excluded_keys}")
    
    # Load state dict
    missing_keys, unexpected_keys = depth_net.load_state_dict(depth_state, strict=False)
    
    if missing_keys:
        expected_missing = {'weight', 'bias'}
        actual_missing = set(missing_keys)
        unexpected_missing = actual_missing - expected_missing
        if unexpected_missing:
            print(f"⚠️ Unexpected missing keys: {unexpected_missing}")
        else:
            print(f"✅ Expected missing keys (training-only): {expected_missing & actual_missing}")
    if unexpected_keys:
        print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
    
    print("✅ Dual-Head model loaded successfully")
    return depth_net, config, max_depth


def create_dummy_input(height, width, device='cpu'):
    """Create dummy input"""
    print(f"Creating dummy input: [1, 3, {height}, {width}]")
    dummy_input = torch.randn(1, 3, height, width).to(device)
    return dummy_input


def patch_model_for_nnef_compatibility(model):
    """Replace ReflectionPad2d with ZeroPad2d for NNEF compatibility"""
    reflection_count = 0
    
    def replace_reflection_pad(module):
        nonlocal reflection_count
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ReflectionPad2d):
                print(f"🔄 Replacing ReflectionPad2d -> ZeroPad2d: {name}")
                setattr(module, name, torch.nn.ZeroPad2d(child.padding))
                reflection_count += 1
            else:
                replace_reflection_pad(child)
    
    print("🔧 Patching model for NNEF compatibility...")
    replace_reflection_pad(model)
    print(f"✅ Replaced {reflection_count} ReflectionPad2d layers")
    
    return model


def export_dual_head_onnx(model, dummy_input, output_path, separate_outputs=False, opset_version=11, use_dynamic_axes=False):
    """Export Dual-Head model to ONNX"""
    print(f"Exporting to ONNX: {output_path}")
    print(f"Separate outputs: {separate_outputs}")
    print(f"Opset version: {opset_version}")
    print(f"Dynamic axes: {use_dynamic_axes}")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if separate_outputs:
        # Export with integer and fractional outputs only (no composed depth)
        if use_dynamic_axes:
            # With dynamic batch size
            torch.onnx.export(
                model, 
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=['rgb'],
                output_names=['integer_sigmoid', 'fractional_sigmoid'],
                verbose=False,
                dynamic_axes={
                    'rgb': {0: 'batch_size'},
                    'integer_sigmoid': {0: 'batch_size'},
                    'fractional_sigmoid': {0: 'batch_size'}
                }
            )
        else:
            # Fixed batch size (better for NPU conversion)
            torch.onnx.export(
                model, 
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=['rgb'],
                output_names=['integer_sigmoid', 'fractional_sigmoid'],
                verbose=False
            )
        print("✅ Exported with separate outputs: integer_sigmoid, fractional_sigmoid (no composed depth)")
    else:
        # Export with composed depth only
        if use_dynamic_axes:
            torch.onnx.export(
                model, 
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=['rgb'],
                output_names=['depth'],
                verbose=False,
                dynamic_axes={
                    'rgb': {0: 'batch_size'},
                    'depth': {0: 'batch_size'}
                }
            )
        else:
            torch.onnx.export(
                model, 
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=['rgb'],
                output_names=['depth'],
                verbose=False
            )
        print("✅ Exported with composed depth output only")
    
    print("✅ PyTorch to ONNX conversion completed!")


def main():
    """Main function for Dual-Head ONNX conversion"""
    try:
        # Setup
        monkey_patch_horovod()
        args = parse_args()
        
        print("🚀 Starting Dual-Head ResNetSAN01 to ONNX conversion")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Output: {args.output}")
        print(f"Input shape: {args.input_shape}")
        print(f"Opset version: {args.opset_version}")
        print(f"Separate outputs: {args.separate_outputs}")
        print(f"Keep ReflectionPad2d: {args.keep_reflection_pad}")
        print(f"Max depth: {args.max_depth}m")
        print()
        
        # Load Dual-Head model
        depth_net, config, config_max_depth = load_dual_head_model(args.checkpoint)
        
        # Use config max_depth if not specified
        max_depth = args.max_depth if args.max_depth != 15.0 else config_max_depth
        print(f"Using max_depth: {max_depth}m")
        
        # Create wrapper
        if args.separate_outputs:
            # Use TwoOutputs wrapper for NPU (integer + fractional only)
            wrapper_model = DualHeadDepthNetTwoOutputs(depth_net, max_depth)
        else:
            # Use SingleOutput wrapper for simple deployment (composed depth only)
            wrapper_model = DualHeadDepthNetSingleOutput(depth_net, max_depth)
        
        # Apply NNEF compatibility patch if needed
        if not args.keep_reflection_pad:
            print("🔧 Converting ReflectionPad2d → ZeroPad2d for NNEF compatibility...")
            wrapper_model = patch_model_for_nnef_compatibility(wrapper_model)
        else:
            print("✅ Keeping ReflectionPad2d for better quality")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wrapper_model = wrapper_model.to(device)
        print(f"Using device: {device}")
        print()
        
        # Create dummy input
        height, width = args.input_shape
        dummy_input = create_dummy_input(height, width, device)
        
        # Test forward pass
        print("Testing forward pass...")
        wrapper_model.eval()
        with torch.no_grad():
            if args.separate_outputs:
                # TwoOutputs wrapper returns only integer and fractional
                integer_sig, fractional_sig = wrapper_model(dummy_input)
                print(f"Integer sigmoid shape: {integer_sig.shape}, range: [{integer_sig.min():.3f}, {integer_sig.max():.3f}]")
                print(f"Fractional sigmoid shape: {fractional_sig.shape}, range: [{fractional_sig.min():.3f}, {fractional_sig.max():.3f}]")
                # Compute depth for display (not part of ONNX output)
                depth = integer_sig * max_depth + fractional_sig
                print(f"Composed depth (computed): {depth.shape}, range: [{depth.min():.3f}m, {depth.max():.3f}m]")
            else:
                depth = wrapper_model(dummy_input)
                print(f"Composed depth shape: {depth.shape}, range: [{depth.min():.3f}m, {depth.max():.3f}m]")
        print()
        
        # Export to ONNX
        export_dual_head_onnx(
            wrapper_model, 
            dummy_input, 
            args.output, 
            args.separate_outputs, 
            args.opset_version,
            args.dynamic_axes
        )
        
        # Print results
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)
            print()
            print(f"📊 ONNX model size: {file_size:.1f} MB")
            print(f"📁 Output file: {args.output}")
        
        print()
        print("🎉 Dual-Head ONNX conversion completed!")
        print(f"✅ The deploy model (ONNX) is saved: {args.output}")
        
        # NNEF conversion info
        nnef_compatible = "✅" if not args.keep_reflection_pad else "❌"
        print()
        print(f"💡 NNEF compatibility: {nnef_compatible}")
        if not args.keep_reflection_pad:
            print("For NNEF conversion, try:")
            print(f"python -m nnef_tools.convert \\")
            print(f"    --input-format=onnx \\")
            print(f"    --output-format=nnef \\")
            print(f"    --input-model={args.output} \\")
            print(f"    --output-model={args.output.replace('.onnx', '.nnef')} \\")
            print(f"    --input-shapes='rgb:[1,3,{height},{width}]'")
        
        print()
        print("✅ The work is done.")
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
