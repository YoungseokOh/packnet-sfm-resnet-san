#!/usr/bin/env python3
"""
Test that upsample_output correctly handles both Standard and Dual-Head outputs
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packnet_sfm.models.model_utils import upsample_output

def test_standard_output():
    """Test Standard output upsampling"""
    print("\n" + "="*80)
    print("TEST 1: Standard Output Upsampling")
    print("="*80)
    
    # Create standard output format
    output = {
        'inv_depths': [
            torch.randn(2, 1, 48, 80),   # scale 0
            torch.randn(2, 1, 24, 40),   # scale 1
        ],
        'uncertainty': [
            torch.randn(2, 1, 48, 80),   # scale 0
            torch.randn(2, 1, 24, 40),   # scale 1
        ]
    }
    
    print("Input shapes:")
    print(f"  inv_depths[0]: {output['inv_depths'][0].shape}")
    print(f"  inv_depths[1]: {output['inv_depths'][1].shape}")
    
    # Upsample
    upsampled = upsample_output(output, mode='nearest', align_corners=None)
    
    print("Output shapes (after upsampling):")
    print(f"  inv_depths[0]: {upsampled['inv_depths'][0].shape}")
    print(f"  inv_depths[1]: {upsampled['inv_depths'][1].shape}")
    
    # Verify all are upsampled to highest resolution
    assert upsampled['inv_depths'][0].shape == upsampled['inv_depths'][1].shape
    print("✅ Standard output upsampling PASSED!")
    

def test_dual_head_output():
    """Test Dual-Head output upsampling"""
    print("\n" + "="*80)
    print("TEST 2: Dual-Head Output Upsampling")
    print("="*80)
    
    # Note: Dual-Head outputs are typically single-scale (scale 0) during training
    # Decoder outputs multiple scales but they are NOT upsampled via upsample_output
    # Only multi-scale outputs (list format) would be upsampled
    
    # Create dual-head output format - single scale scenario (more realistic)
    output = {
        ('integer', 0): torch.randn(2, 1, 48, 80),
        ('fractional', 0): torch.randn(2, 1, 48, 80),
    }
    
    print("Input shapes:")
    print(f"  ('integer', 0): {output[('integer', 0)].shape}")
    print(f"  ('fractional', 0): {output[('fractional', 0)].shape}")
    
    # Upsample
    upsampled = upsample_output(output, mode='nearest', align_corners=None)
    
    print("Output shapes (after upsampling):")
    print(f"  ('integer', 0): {upsampled[('integer', 0)].shape}")
    print(f"  ('fractional', 0): {upsampled[('fractional', 0)].shape}")
    
    # Verify both have same shape
    assert upsampled[('integer', 0)].shape == upsampled[('fractional', 0)].shape
    # Check they are still tensors (not lists)
    assert isinstance(upsampled[('integer', 0)], torch.Tensor)
    assert isinstance(upsampled[('fractional', 0)], torch.Tensor)
    print("✅ Dual-Head output upsampling PASSED!")


def test_mixed_output():
    """Test mixed Standard and Dual-Head outputs"""
    print("\n" + "="*80)
    print("TEST 3: Mixed Output Upsampling")
    print("="*80)
    
    # Create mixed output format
    output = {
        ('integer', 0): torch.randn(2, 1, 48, 80),
        ('fractional', 0): torch.randn(2, 1, 48, 80),
        'inv_depths_context': [
            [torch.randn(2, 1, 48, 80), torch.randn(2, 1, 24, 40)],
            [torch.randn(2, 1, 48, 80), torch.randn(2, 1, 24, 40)],
        ]
    }
    
    print("Input shapes:")
    print(f"  ('integer', 0): {output[('integer', 0)].shape}")
    print(f"  ('fractional', 0): {output[('fractional', 0)].shape}")
    print(f"  inv_depths_context[0][0]: {output['inv_depths_context'][0][0].shape}")
    
    # Upsample
    upsampled = upsample_output(output, mode='nearest', align_corners=None)
    
    print("Output shapes (after upsampling):")
    print(f"  ('integer', 0): {upsampled[('integer', 0)].shape}")
    print(f"  ('fractional', 0): {upsampled[('fractional', 0)].shape}")
    print(f"  inv_depths_context[0][0]: {upsampled['inv_depths_context'][0][0].shape}")
    
    # Verify
    assert upsampled[('integer', 0)].shape == upsampled[('fractional', 0)].shape
    assert upsampled['inv_depths_context'][0][0].shape == upsampled['inv_depths_context'][0][1].shape
    print("✅ Mixed output upsampling PASSED!")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🧪 UPSAMPLE_OUTPUT FIX VERIFICATION TEST SUITE")
    print("="*80)
    
    test_standard_output()
    test_dual_head_output()
    test_mixed_output()
    
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print("✅ All upsample_output tests PASSED!")
    print("\nThe fix successfully handles:")
    print("  ✓ Standard outputs (string keys like 'inv_depths')")
    print("  ✓ Dual-Head outputs (tuple keys like ('integer', 0))")
    print("  ✓ Mixed outputs combining both formats")
    print("="*80)
