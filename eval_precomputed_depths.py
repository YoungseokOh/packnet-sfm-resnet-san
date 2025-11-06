#!/usr/bin/env python3
"""
Evaluate pre-computed depth predictions (NPU INT8, ONNX FP32, etc.)
using the official evaluation pipeline

This creates a custom ModelWrapper that loads .npy depth files
instead of running inference, allowing us to use the same metrics
computation as training/validation.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.horovod import hvd_init


class PrecomputedDepthWrapper(ModelWrapper):
    """
    ModelWrapper that loads pre-computed depth predictions
    instead of running inference
    """
    
    def __init__(self, config, depth_dir):
        super().__init__(config)
        self.depth_dir = Path(depth_dir)
        self.current_filename = None
        print(f"📂 Loading pre-computed depths from: {self.depth_dir}")
        
        # Count available depth files
        npy_files = list(self.depth_dir.glob('*.npy'))
        print(f"   Found {len(npy_files)} .npy files")
        
        # Replace model's depth_net with our custom loader
        self._replace_depth_net()
    
    def _replace_depth_net(self):
        """Replace model's depth_net with custom loader"""
        original_depth_net = self.model.depth_net
        parent = self
        
        class CustomDepthNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parent_wrapper = parent
            
            def forward(self, x):
                # Load pre-computed depth based on current filename
                return self.parent_wrapper._load_precomputed_depth(x)
        
        self.model.depth_net = CustomDepthNet()
        print(f"   ✅ Replaced depth_net with custom loader")
    
    def _load_precomputed_depth(self, image):
        """Load pre-computed depth from file"""
        if self.current_filename is None:
            # Return zeros
            B, _, H, W = image.shape
            dummy_depth = torch.zeros((B, 1, H, W), device=image.device, dtype=image.dtype)
            if not hasattr(self, '_warned_none'):
                print(f"   ⚠️  current_filename is None! Returning zeros")
                self._warned_none = True
            return dummy_depth
        
        # Load pre-computed depth
        depth_path = self.depth_dir / f"{self.current_filename}.npy"
        
        if not depth_path.exists():
            if not hasattr(self, '_warned_missing'):
                print(f"   ⚠️  Missing: {depth_path.name}")
                self._warned_missing = True
            # Return zeros
            B, _, H, W = image.shape
            return torch.zeros((B, 1, H, W), device=image.device, dtype=image.dtype)
        
        # Load depth prediction
        pred_depth = np.load(depth_path)
        
        # Handle different shapes (squeeze if needed)
        if pred_depth.ndim == 3:
            pred_depth = pred_depth.squeeze(0)  # Remove batch dimension if present
        
        # Debug: Print for first few samples
        if not hasattr(self, '_load_count'):
            self._load_count = 0
        if self._load_count < 3:
            print(f"   ✅ Loaded {depth_path.name}: shape={pred_depth.shape}, range=[{pred_depth.min():.2f}, {pred_depth.max():.2f}]m")
            self._load_count += 1
        
        pred_depth_tensor = torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0).to(image.device)
        
        return pred_depth_tensor
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Set current filename before calling parent test_step
        """
        # Debug: Print batch keys on first call
        if not hasattr(self, '_debug_batch_printed'):
            print(f"\n🔍 Batch keys: {list(batch.keys())}")
            for key in ['filename', 'fname', 'idx', 'index', 'new_filename']:
                if key in batch:
                    print(f"   Found key '{key}': {batch[key]}")
            self._debug_batch_printed = True
        
        # Try to extract filename from batch
        self.current_filename = None
        
        # Try different possible keys
        for key in ['filename', 'new_filename', 'fname']:
            if key in batch:
                fname = batch[key]
                if isinstance(fname, (list, tuple)):
                    self.current_filename = fname[0]
                elif torch.is_tensor(fname):
                    continue  # Skip tensors
                else:
                    self.current_filename = fname
                break
        
        # Call parent test_step
        return super().test_step(batch, batch_idx, dataloader_idx)


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate pre-computed depth predictions using official pipeline'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Original checkpoint (.ckpt) for loading config')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration (.yaml)')
    parser.add_argument('--depth_dir', type=str, required=True,
                       help='Directory containing pre-computed .npy depth files')
    parser.add_argument('--model_name', type=str, default='PrecomputedDepth',
                       help='Model name for display (e.g., "NPU INT8", "ONNX FP32")')
    args = parser.parse_args()
    return args


def evaluate_precomputed(ckpt_file, cfg_file, depth_dir, model_name):
    """
    Evaluate pre-computed depth predictions
    
    Parameters
    ----------
    ckpt_file : str
        Checkpoint path (used only for loading config)
    cfg_file : str
        Configuration file
    depth_dir : str
        Directory with pre-computed .npy depth files
    model_name : str
        Model name for display
    """
    print("\n" + "="*80)
    print(f"📊 {model_name} EVALUATION (Official Pipeline)")
    print("="*80)
    
    # Initialize horovod
    hvd_init()
    
    # Parse configuration
    config, state_dict = parse_test_file(ckpt_file, cfg_file)
    
    # Set debug if requested
    set_debug(config.debug)
    
    # Create custom wrapper that loads pre-computed depths
    model_wrapper = PrecomputedDepthWrapper(config, depth_dir)
    
    # Don't load state dict - we're not using the model weights
    print("ℹ️  Skipping model weight loading (using pre-computed depths)")
    
    # Use FP32
    config.arch["dtype"] = None
    
    # Create trainer
    trainer = HorovodTrainer(**config.arch)
    
    # Run evaluation on test set
    print(f"\n📊 Running {model_name} evaluation on test set...")
    trainer.test(model_wrapper)
    
    print(f"\n✅ {model_name} evaluation complete!\n")


if __name__ == '__main__':
    args = parse_args()
    evaluate_precomputed(
        args.checkpoint,
        args.config,
        args.depth_dir,
        args.model_name
    )
