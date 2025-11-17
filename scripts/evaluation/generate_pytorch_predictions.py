#!/usr/bin/env python3
"""
Generate PyTorch FP32 predictions using the same pipeline as official eval.
This ensures predictions match exactly what the official evaluation uses.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Horovod mocking (for single GPU)
import sys
import types
mock_hvd = types.ModuleType('horovod')
mock_hvd.torch = types.ModuleType('torch')
mock_hvd.torch.init = lambda: print("🔧 Mock Horovod init (single GPU mode)")
mock_hvd.torch.size = lambda: 1
mock_hvd.torch.rank = lambda: 0
mock_hvd.torch.local_rank = lambda: 0
mock_hvd.torch.allreduce = lambda x, *args, **kwargs: x
mock_hvd.torch.allgather = lambda x, *args, **kwargs: [x]
mock_hvd.torch.broadcast = lambda x, *args, **kwargs: x
sys.modules['horovod'] = mock_hvd
sys.modules['horovod.torch'] = mock_hvd.torch

from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.models.model_wrapper import ModelWrapper


def hvd_init():
    """Initialize Horovod (mocked for single GPU)"""
    import horovod.torch as hvd
    hvd.init()


def generate_predictions(checkpoint_path, config_path, output_dir):
    """Generate and save predictions using the official model pipeline"""
    
    print("="*80)
    print("📊 Generating PyTorch FP32 Predictions (Official Pipeline)")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config:     {config_path}")
    print(f"Output dir: {output_dir}")
    print("="*80)
    print()
    
    # Initialize horovod
    hvd_init()
    
    # Load config and checkpoint
    config, state_dict = parse_test_file(checkpoint_path, config_path)
    
    # Create model wrapper
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_state_dict(state_dict, strict=False)
    model_wrapper = model_wrapper.to('cuda')
    model_wrapper.eval()
    
    # Get test dataloader
    test_dataloaders = model_wrapper.test_dataloader()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Saving predictions to: {output_dir}\n")
    
    # Process each batch
    total_samples = 0
    with torch.no_grad():
        for n, dataloader in enumerate(test_dataloaders):
            print(f"Processing dataloader {n}...")
            for batch in tqdm(dataloader, desc="Generating predictions"):
                # Move batch to GPU
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to('cuda')
                
                # Run model inference (same as evaluate_depth)
                model_outputs = model_wrapper.model(batch)['inv_depths']
                output0 = model_outputs[0]  # (B,1,H,W)
                
                # Check if direct depth mode
                depth_net = getattr(model_wrapper.model, 'depth_net', None)
                depth_output_mode = getattr(depth_net, 'depth_output_mode', 'sigmoid')
                
                if depth_output_mode == 'direct':
                    # Direct depth mode - output is already depth
                    depth_pred = output0
                else:
                    # Sigmoid mode - need to convert
                    from packnet_sfm.utils.post_process_depth import sigmoid_to_inv_depth
                    from packnet_sfm.utils.depth import inv2depth
                    min_depth = float(config.model.params.min_depth)
                    max_depth = float(config.model.params.max_depth)
                    use_log_space = getattr(model_wrapper.model, 'use_log_space', False)
                    
                    sigmoid0 = output0
                    inv_depth = sigmoid_to_inv_depth(sigmoid0, min_depth, max_depth, use_log_space=use_log_space)
                    depth_pred = inv2depth(inv_depth)
                
                # Save predictions as .npy files
                batch_size = depth_pred.shape[0]
                for i in range(batch_size):
                    # Get sample info
                    idx = batch['idx'][i].item()
                    
                    # Get filename from batch (if available)
                    # Otherwise use idx
                    if 'filename' in batch:
                        filename = batch['filename'][i]
                        if isinstance(filename, list):
                            filename = filename[0]
                    else:
                        # Try to get from dataset
                        dataset = dataloader.dataset
                        sample = dataset.samples[idx]
                        filename = sample.get('new_filename', f'{idx:010d}')
                    
                    # Extract depth map (H, W)
                    depth_map = depth_pred[i, 0].cpu().numpy()
                    
                    # Save as .npy
                    save_path = output_dir / f"{filename}.npy"
                    np.save(save_path, depth_map)
                    
                    total_samples += 1
    
    print(f"\n✅ Generated {total_samples} predictions")
    print(f"📁 Saved to: {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PyTorch FP32 predictions using official pipeline'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint (.ckpt) file')
    parser.add_argument('--config', type=str, default=None,
                       help='Config (.yaml) file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for .npy files')
    
    args = parser.parse_args()
    
    generate_predictions(
        args.checkpoint,
        args.config,
        args.output_dir
    )


if __name__ == '__main__':
    main()
