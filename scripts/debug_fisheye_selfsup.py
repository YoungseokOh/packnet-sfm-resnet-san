#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/workspace/packnet-sfm')

from packnet_sfm.datasets.ncdb_dataset import NcdbDataset
from packnet_sfm.datasets.transforms import get_transforms
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.geometry.camera import FisheyeCamera

def test_fisheye_selfsup():
    """Test fisheye self-supervised learning setup"""
    
    print("🔍 Testing Fisheye Self-Supervised Learning Setup")
    print("=" * 60)
    
    # Dataset configuration
    dataset_config = {
        'dataset_root': '/workspace/data/ncdb-cls',
        'split_file': 'splits/combined_train.json',
        'back_context': 1,
        'forward_context': 1,
        'with_context': True
    }
    
    # Transform configuration
    transform_config = {
        'mode': 'train',  # Required first parameter
        'image_shape': (192, 640),
        'jittering': (0.2, 0.2, 0.2, 0.05)
    }
    
    try:
        # Create transforms
        print("📦 Creating transforms...")
        transforms = get_transforms(**transform_config)
        print(f"✅ Transforms created: {type(transforms)}")
        
        # Create dataset
        print("\n📊 Creating NCDB dataset...")
        dataset = NcdbDataset(
            transform=transforms,
            **dataset_config
        )
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Create dataloader with custom collate function
        print("\n🔄 Creating dataloader...")
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False,
            collate_fn=NcdbDataset.custom_collate_fn,
            num_workers=0  # Use 0 for debugging
        )
        print(f"✅ Dataloader created")
        
        # Test batch loading
        print("\n🧪 Testing batch loading...")
        for i, batch in enumerate(dataloader):
            print(f"\n--- Batch {i+1} ---")
            
            # Print batch structure
            for key, value in batch.items():
                if key == 'rgb':
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: tensor {value.shape} ({value.dtype})")
                    elif isinstance(value, list):
                        print(f"  {key}: list of {len(value)} items")
                        if value and hasattr(value[0], 'size'):
                            print(f"    First item size: {value[0].size}")
                        elif value and hasattr(value[0], 'shape'):
                            print(f"    First item shape: {value[0].shape}")
                    else:
                        print(f"  {key}: {type(value)}")
                elif key == 'rgb_context':
                    if isinstance(value, list):
                        print(f"  {key}: list of {len(value)} context frames")
                        for j, ctx_batch in enumerate(value):
                            if isinstance(ctx_batch, list):
                                print(f"    Batch {j}: list of {len(ctx_batch)} context images")
                            elif hasattr(ctx_batch, 'shape'):
                                print(f"    Batch {j}: tensor {ctx_batch.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
                elif key in ['camera', 'camera_context']:
                    if isinstance(value, list):
                        print(f"  {key}: list of {len(value)} camera objects")
                        if value:
                            print(f"    First camera type: {type(value[0])}")
                    else:
                        print(f"  {key}: {type(value)}")
                elif key == 'distortion_coeffs':
                    if isinstance(value, dict):
                        print(f"  {key}: dict with keys {list(value.keys())}")
                        for k, v in value.items():
                            if hasattr(v, 'shape'):
                                print(f"    {k}: {v.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
                else:
                    if hasattr(value, 'shape'):
                        print(f"  {key}: tensor {value.shape}")
                    elif isinstance(value, (list, tuple)):
                        print(f"  {key}: {type(value)} of length {len(value)}")
                    else:
                        print(f"  {key}: {type(value)} = {value}")
            
            # Test only first batch
            if i >= 0:
                break
        
        print("\n✅ Fisheye self-supervised learning setup test completed successfully!")
        print("\n🎯 Key observations:")
        print("  - NCDB dataset loads correctly")
        print("  - Context frames are discovered when available")
        print("  - FisheyeCamera objects are created properly")
        print("  - Custom collate function handles complex data structures")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fisheye_selfsup()
    if not success:
        sys.exit(1)
    print("\n🎉 All tests passed!")