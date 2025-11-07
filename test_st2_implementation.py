#!/usr/bin/env python3
"""
Quick test script for ST2 Dual-Head implementation

This script tests all 5 phases of the Dual-Head implementation:
1. DualHeadDepthDecoder
2. Helper functions (decompose_depth, dual_head_to_depth)
3. DualHeadDepthLoss
4. ResNetSAN01 integration
5. Model wrapper auto-detection
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_phase_1_decoder():
    """Test Phase 1: DualHeadDepthDecoder"""
    print("\n" + "="*80)
    print("TEST PHASE 1: DualHeadDepthDecoder")
    print("="*80)
    
    from packnet_sfm.networks.layers.resnet.dual_head_depth_decoder import DualHeadDepthDecoder
    
    # Create decoder
    num_ch_enc = [64, 64, 128, 256, 512]
    decoder = DualHeadDepthDecoder(
        num_ch_enc=num_ch_enc,
        scales=[0],
        max_depth=15.0
    )
    
    # Dummy encoder features
    batch_size = 2
    features = [
        torch.randn(batch_size, 64, 96, 160),   # scale 0
        torch.randn(batch_size, 64, 48, 80),    # scale 1
        torch.randn(batch_size, 128, 24, 40),   # scale 2
        torch.randn(batch_size, 256, 12, 20),   # scale 3
        torch.randn(batch_size, 512, 6, 10),    # scale 4
    ]
    
    # Forward pass
    outputs = decoder(features)
    
    # Check outputs
    assert ("integer", 0) in outputs, "Missing integer output"
    assert ("fractional", 0) in outputs, "Missing fractional output"
    
    integer_out = outputs[("integer", 0)]
    fractional_out = outputs[("fractional", 0)]
    
    # Decoder upsamples by 2x at each level (5 levels total: 6->12->24->48->96->192)
    expected_h = 192  # 6 * (2^5) = 192
    expected_w = 320  # 10 * (2^5) = 320
    assert integer_out.shape == (batch_size, 1, expected_h, expected_w), f"Wrong integer shape: {integer_out.shape}"
    assert fractional_out.shape == (batch_size, 1, expected_h, expected_w), f"Wrong fractional shape: {fractional_out.shape}"
    
    # Check value range
    assert integer_out.min() >= 0.0 and integer_out.max() <= 1.0, "Integer out of range"
    assert fractional_out.min() >= 0.0 and fractional_out.max() <= 1.0, "Fractional out of range"
    
    print("✅ Phase 1 PASSED: DualHeadDepthDecoder works correctly!")
    return True


def test_phase_2_helpers():
    """Test Phase 2: Helper Functions"""
    print("\n" + "="*80)
    print("TEST PHASE 2: Helper Functions")
    print("="*80)
    
    from packnet_sfm.networks.layers.resnet.layers import decompose_depth, dual_head_to_depth
    
    # Test decompose_depth
    depth = torch.tensor([[[[5.7]]]])  # 5.7m
    integer_gt, frac_gt = decompose_depth(depth, 15.0)
    
    expected_integer = 5.0 / 15.0  # 0.333
    expected_frac = 0.7
    
    assert abs(integer_gt.item() - expected_integer) < 1e-5, f"Integer decomposition failed: {integer_gt.item()} vs {expected_integer}"
    assert abs(frac_gt.item() - expected_frac) < 1e-5, f"Fractional decomposition failed: {frac_gt.item()} vs {expected_frac}"
    
    # Test dual_head_to_depth (round-trip)
    depth_reconstructed = dual_head_to_depth(integer_gt, frac_gt, 15.0)
    assert abs(depth_reconstructed.item() - 5.7) < 1e-5, f"Reconstruction failed: {depth_reconstructed.item()} vs 5.7"
    
    print("✅ Phase 2 PASSED: Helper functions work correctly!")
    return True


def test_phase_3_loss():
    """Test Phase 3: DualHeadDepthLoss"""
    print("\n" + "="*80)
    print("TEST PHASE 3: DualHeadDepthLoss")
    print("="*80)
    
    from packnet_sfm.losses.dual_head_depth_loss import DualHeadDepthLoss
    
    # Create loss
    loss_fn = DualHeadDepthLoss(
        max_depth=15.0,
        min_depth=0.5,
        integer_weight=1.0,
        fractional_weight=10.0,
        consistency_weight=0.5
    )
    
    # Dummy predictions (requires_grad=True for gradient)
    batch_size = 2
    outputs = {
        ("integer", 0): torch.rand(batch_size, 1, 96, 160, requires_grad=True),
        ("fractional", 0): torch.rand(batch_size, 1, 96, 160, requires_grad=True)
    }
    
    # Dummy GT depth
    depth_gt = torch.rand(batch_size, 1, 96, 160) * 10 + 2  # [2, 12]m
    
    # Forward pass
    loss_dict = loss_fn(outputs, depth_gt, return_logs=True)
    
    # Check outputs
    assert 'loss' in loss_dict, "Missing total loss"
    assert 'integer_loss' in loss_dict, "Missing integer_loss"
    assert 'fractional_loss' in loss_dict, "Missing fractional_loss"
    assert 'consistency_loss' in loss_dict, "Missing consistency_loss"
    
    total_loss = loss_dict['loss']
    assert total_loss.requires_grad, "Loss should have gradient"
    assert total_loss.item() > 0, "Loss should be positive"
    
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Integer loss: {loss_dict['integer_loss'].item():.4f}")
    print(f"   Fractional loss: {loss_dict['fractional_loss'].item():.4f}")
    print(f"   Consistency loss: {loss_dict['consistency_loss'].item():.4f}")
    
    print("✅ Phase 3 PASSED: DualHeadDepthLoss works correctly!")
    return True


def test_phase_4_resnet():
    """Test Phase 4: ResNetSAN01 Integration"""
    print("\n" + "="*80)
    print("TEST PHASE 4: ResNetSAN01 Integration")
    print("="*80)
    
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    
    # Create Dual-Head model
    model = ResNetSAN01(
        version='18A',
        use_dual_head=True,
        max_depth=15.0
    )
    
    # Check attributes
    assert hasattr(model, 'is_dual_head'), "Missing is_dual_head attribute"
    assert model.is_dual_head == True, "is_dual_head should be True"
    
    # Check decoder type
    from packnet_sfm.networks.layers.resnet.dual_head_depth_decoder import DualHeadDepthDecoder
    assert isinstance(model.decoder, DualHeadDepthDecoder), "Decoder should be DualHeadDepthDecoder"
    
    # Forward pass
    batch_size = 1
    rgb = torch.randn(batch_size, 3, 384, 640)
    
    model.eval()
    with torch.no_grad():
        output, _ = model.run_network(rgb)
    
    # For Dual-Head, output is a dict, not a list
    if isinstance(output, dict):
        # Check outputs
        assert ("integer", 0) in output, "Missing integer output"
        assert ("fractional", 0) in output, "Missing fractional output"
    else:
        raise AssertionError(f"Expected dict output for Dual-Head, got {type(output)}")
    
    print("✅ Phase 4 PASSED: ResNetSAN01 integration works correctly!")
    return True


def test_phase_5_model_wrapper():
    """Test Phase 5: Model Wrapper Auto-Detection"""
    print("\n" + "="*80)
    print("TEST PHASE 5: Model Wrapper Auto-Detection")
    print("="*80)
    
    from packnet_sfm.models.SemiSupCompletionModel import SemiSupCompletionModel
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    
    # Create Dual-Head depth network
    depth_net = ResNetSAN01(
        version='18A',
        use_dual_head=True,
        max_depth=15.0
    )
    
    # Create model wrapper
    model = SemiSupCompletionModel(
        depth_net=depth_net,
        supervised_loss_weight=0.9,
        min_depth=0.5,
        max_depth=15.0
    )
    
    # Check auto-detection
    assert hasattr(model.depth_net, 'is_dual_head'), "depth_net missing is_dual_head"
    assert model.depth_net.is_dual_head == True, "is_dual_head should be True"
    
    print("✅ Phase 5 PASSED: Model wrapper auto-detection works correctly!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 ST2 DUAL-HEAD IMPLEMENTATION TEST SUITE")
    print("="*80)
    
    tests = [
        ("Phase 1: DualHeadDepthDecoder", test_phase_1_decoder),
        ("Phase 2: Helper Functions", test_phase_2_helpers),
        ("Phase 3: DualHeadDepthLoss", test_phase_3_loss),
        ("Phase 4: ResNetSAN01 Integration", test_phase_4_resnet),
        ("Phase 5: Model Wrapper", test_phase_5_model_wrapper),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"❌ {name} FAILED:")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print(f"📊 TEST SUMMARY")
    print("="*80)
    print(f"   Total:  {passed + failed}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! ST2 implementation is ready!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
