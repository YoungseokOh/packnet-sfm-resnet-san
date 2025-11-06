#!/usr/bin/env python3
# Copyright 2020 Toyota Research Institute.  All rights reserved.

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


class SimpleDepthNet(torch.nn.Module):
    """
    Very simple wrapper for depth network - mimics your LwDepthResNet approach
    """
    def __init__(self, resnet_san_model):
        super().__init__()
        self.depth_net = resnet_san_model
        
    def forward(self, rgb):
        """Simple RGB-only inference - returns single tensor"""
        # Set eval mode to ensure single-scale output
        self.depth_net.eval()
        
        # Direct inference without complex wrapper logic
        inv_depths, _ = self.depth_net.run_network(rgb, input_depth=None)
        
        # Return single tensor (not list) to avoid ONNX graph complexity
        # inv_depths[0] is [B, 1, H, W]
        return inv_depths[0]


def parse_args():
    """Parse arguments for simple ONNX conversion"""
    parser = argparse.ArgumentParser(description='Simple PackNet-SFM to ONNX conversion')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[352, 1216],  # 🆕 올바른 기본값
                        help='Input image shape [height, width]')
    parser.add_argument('--opset_version', type=int, default=10,
                        help='ONNX opset version (default: 10 for compatibility)')
    parser.add_argument('--keep_reflection_pad', action='store_true',  # 🆕 새 옵션
                        help='Keep ReflectionPad2d (better quality, may not work with NNEF)')
    
    args = parser.parse_args()
    
    # Validate checkpoint file
    assert args.checkpoint.endswith('.ckpt'), 'Checkpoint must be .ckpt'
    assert os.path.exists(args.checkpoint), f'Checkpoint not found: {args.checkpoint}'
    
    # Set default output path
    if args.output is None:
        base_name = os.path.basename(args.checkpoint).replace('.ckpt', '')
        pad_type = "reflect" if args.keep_reflection_pad else "zero"
        args.output = f"onnx/{base_name}_{args.input_shape[1]}x{args.input_shape[0]}_{pad_type}.onnx"
    
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


def load_model_simple(checkpoint_path):
    """Load model using simple approach - similar to your method"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    
    # Set debug
    set_debug(config.debug)
    
    # Import and create the model
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    
    # 🆕 Get depth output mode from config (if available)
    depth_output_mode = config.model.depth_net.get('depth_output_mode', 'sigmoid')
    min_depth = config.model.params.get('min_depth', 0.5)
    max_depth = config.model.params.get('max_depth', 80.0)
    
    print(f"📊 Model Configuration:")
    print(f"   depth_output_mode: {depth_output_mode}")
    print(f"   min_depth: {min_depth}m")
    print(f"   max_depth: {max_depth}m")
    
    # Create depth network
    depth_net = ResNetSAN01(
        dropout=config.model.depth_net.get('dropout', 0.5),
        version=config.model.depth_net.get('version', '1A'),
        use_enhanced_lidar=False,  # 🔧 ONNX 변환 시 MinkowskiEngine 비활성화
        use_film=False,  # 🔧 ONNX 변환 시 FiLM 비활성화 (MinkowskiEncoder 불필요)
        depth_output_mode=depth_output_mode,  # 🆕 Direct depth mode support
        min_depth=min_depth,  # 🆕 Depth range
        max_depth=max_depth   # 🆕 Depth range
    )
    
    # Extract and load state dict - similar to your OrderedDict approach
    depth_state = OrderedDict()
    excluded_keys = []
    for key, value in state_dict.items():
        if key.startswith('model.depth_net.'):
            new_key = key.replace('model.depth_net.', '')
            
            # 🔧 ONNX 변환 시 학습 전용 파라미터 제외
            if new_key in ['weight', 'bias']:
                excluded_keys.append(new_key)
                continue
            
            depth_state[new_key] = value
    
    if excluded_keys:
        print(f"🗑️ Excluded training-only parameters: {excluded_keys}")
    
    # Load state dict (strict=False: weight, bias 없어도 됨)
    missing_keys, unexpected_keys = depth_net.load_state_dict(depth_state, strict=False)
    
    if missing_keys:
        # weight, bias는 정상적으로 누락되어야 함
        expected_missing = {'weight', 'bias'}
        actual_missing = set(missing_keys)
        unexpected_missing = actual_missing - expected_missing
        if unexpected_missing:
            print(f"⚠️ Unexpected missing keys: {unexpected_missing}")
        else:
            print(f"✅ Expected missing keys (training-only): {expected_missing & actual_missing}")
    if unexpected_keys:
        print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
    
    print("✅ Model loaded successfully")
    return depth_net, config


def create_dummy_input(height, width, device='cpu'):
    """Create dummy input similar to your PIL image approach"""
    print("Creating dummy input...")
    
    # Create dummy RGB input [1, 3, H, W]
    dummy_input = torch.randn(1, 3, height, width).to(device)
    
    print(f"Dummy input shape: {dummy_input.shape}")
    return dummy_input


def simple_onnx_export(model, dummy_input, output_path, opset_version=10):
    """Very simple ONNX export - following your exact pattern"""
    print(f"Exporting to ONNX: {output_path}")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Simple ONNX export - no complex settings
    torch.onnx.export(
        model, 
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['rgb'],
        output_names=['depth'],
        verbose=False
    )
    
    print("✅ PyTorch to ONNX conversion completed!")


def patch_model_for_nnef_compatibility(model):
    """모델을 NNEF 호환되도록 패치"""
    
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


def main():
    """Main function - following your simple pattern"""
    try:
        # Setup
        monkey_patch_horovod()
        args = parse_args()
        
        print("🚀 Starting simple PackNet-SFM to ONNX conversion")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Output: {args.output}")
        print(f"Input shape: {args.input_shape}")
        print(f"Opset version: {args.opset_version}")
        print(f"Keep ReflectionPad2d: {args.keep_reflection_pad}")
        
        # Load model
        depth_net, config = load_model_simple(args.checkpoint)
        
        # Create simple wrapper
        simple_model = SimpleDepthNet(depth_net)
        
        # 🆕 조건부 패치 적용
        if not args.keep_reflection_pad:
            print("🔧 Converting ReflectionPad2d → ZeroPad2d for NNEF compatibility...")
            simple_model = patch_model_for_nnef_compatibility(simple_model)
        else:
            print("✅ Keeping ReflectionPad2d for better quality")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        simple_model = simple_model.to(device)
        
        print(f"Using device: {device}")
        
        # Create dummy input
        height, width = args.input_shape
        dummy_input = create_dummy_input(height, width, device)
        
        # Test forward pass
        print("Testing forward pass...")
        simple_model.eval()
        with torch.no_grad():
            test_output = simple_model(dummy_input)
            print(f"Output shape: {test_output.shape}")
            print(f"Depth range: {test_output.min().item():.3f} - {test_output.max().item():.3f}")
        
        # Simple ONNX export
        simple_onnx_export(simple_model, dummy_input, args.output, args.opset_version)
        
        # Print results
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)
            print(f"📊 ONNX model size: {file_size:.1f} MB")
            print(f"📁 Output file: {args.output}")
        
        print("\n🎉 Simple conversion completed!")
        print(f"✅ The deploy model (ONNX) is saved: {args.output}")
        print("✅ The work is done.")
        
        # Create simple NNEF conversion command
        nnef_compatible = "✅" if not args.keep_reflection_pad else "❌"
        print(f"\n💡 NNEF compatibility: {nnef_compatible}")
        if not args.keep_reflection_pad:
            print("For NNEF conversion, try:")
            print(f"python -m nnef_tools.convert \\")
            print(f"    --input-format=onnx \\")
            print(f"    --output-format=nnef \\")
            print(f"    --input-model={args.output} \\")
            print(f"    --output-model={args.output.replace('.onnx', '.nnef')} \\")
            print(f"    --input-shapes='rgb:[1,3,{height},{width}]'")
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()