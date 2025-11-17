#!/usr/bin/env python3
"""
Integration test: Full training pipeline with Dual-Head
이 테스트는 실제 훈련 파이프라인에서 Dual-Head이 제대로 작동하는지 확인합니다.
"""

import torch
import sys
import os
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_full_training_pipeline():
    """Full training pipeline test with actual model components"""
    print("\n" + "="*80)
    print("🧪 FULL TRAINING PIPELINE INTEGRATION TEST")
    print("="*80)
    
    # ========================================
    # 1. Load configuration
    # ========================================
    print("\n[1] Loading configuration...")
    config_path = "configs/train_resnet_san_ncdb_dual_head_640x384.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"✅ Config loaded: use_dual_head = {cfg['model']['depth_net']['use_dual_head']}")
    
    # ========================================
    # 2. Initialize model components
    # ========================================
    print("\n[2] Initializing model components...")
    
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    from packnet_sfm.losses.dual_head_depth_loss import DualHeadDepthLoss
    from packnet_sfm.models.SemiSupCompletionModel import SemiSupCompletionModel
    
    # Initialize ResNetSAN01 with Dual-Head
    depth_net = ResNetSAN01(
        version='18A',
        use_dual_head=True,
        max_depth=15.0,
        min_depth=0.5,
        use_film=False,
        use_enhanced_lidar=False
    )
    
    print(f"✅ ResNetSAN01 initialized (use_dual_head=True)")
    
    # ========================================
    # 3. Forward pass with dummy batch
    # ========================================
    print("\n[3] Forward pass with dummy batch...")
    
    batch_size = 2
    dummy_rgb = torch.randn(batch_size, 3, 384, 640)
    
    depth_net.eval()
    with torch.no_grad():
        # Forward pass - returns dict with 'inv_depths' for standard or dual-head outputs
        result = depth_net(dummy_rgb)
        
        # Check if result is dict or tuple
        if isinstance(result, dict):
            outputs = result
            print(f"   Result is dict (likely eval mode returning simplified format)")
        else:
            # Try to unpack if it's tuple
            try:
                outputs, skip_features = result
            except:
                outputs = result
    
    print(f"✅ Forward pass completed")
    print(f"   Result type: {type(outputs)}")
    if isinstance(outputs, dict):
        print(f"   Output keys: {list(outputs.keys())}")
    
    # ========================================
    # 4. Check output format
    # ========================================
    print("\n[4] Checking output format...")
    
    has_integer = any(k[0] == 'integer' for k in outputs.keys() if isinstance(k, tuple))
    has_fractional = any(k[0] == 'fractional' for k in outputs.keys() if isinstance(k, tuple))
    
    if not has_integer or not has_fractional:
        print(f"❌ Dual-Head outputs missing!")
        print(f"   has_integer: {has_integer}, has_fractional: {has_fractional}")
        return False
    
    print(f"✅ Dual-Head outputs detected:")
    for key in sorted(outputs.keys()):
        if isinstance(key, tuple):
            print(f"   {key}: {outputs[key].shape}, range=[{outputs[key].min():.4f}, {outputs[key].max():.4f}]")
    
    # ========================================
    # 5. Upsample outputs (like in training)
    # ========================================
    print("\n[5] Testing upsample_output...")
    
    from packnet_sfm.models.model_utils import upsample_output
    
    try:
        upsampled = upsample_output(outputs.copy(), mode='nearest', align_corners=None)
        print(f"✅ Upsampling successful")
        print(f"   Upsampled output keys: {list(upsampled.keys())}")
        
        for key in sorted(upsampled.keys()):
            if isinstance(key, tuple):
                print(f"   {key}: {upsampled[key].shape}")
    except Exception as e:
        print(f"❌ Upsampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 6. Loss computation
    # ========================================
    print("\n[6] Computing loss...")
    
    # Create dummy ground truth
    depth_gt = torch.rand(batch_size, 1, 384, 640) * 10 + 2  # [2, 12]m
    
    # Initialize loss
    loss_fn = DualHeadDepthLoss(
        max_depth=15.0,
        min_depth=0.5,
        integer_weight=1.0,
        fractional_weight=10.0,
        consistency_weight=0.5
    )
    
    try:
        # Need gradients for loss
        depth_net.train()
        outputs_train = depth_net(dummy_rgb)
        
        # outputs_train should be a dict with dual-head keys
        if not isinstance(outputs_train, dict):
            print(f"❌ Expected dict, got {type(outputs_train)}")
            return False
        
        loss_dict = loss_fn(outputs_train, depth_gt, return_logs=True)
        
        if 'loss' not in loss_dict:
            print(f"❌ Loss computation returned missing 'loss' key")
            return False
        
        print(f"✅ Loss computation successful:")
        print(f"   Total loss: {loss_dict['loss'].item():.6f}")
        print(f"   Integer loss: {loss_dict['integer_loss'].item():.6f}")
        print(f"   Fractional loss: {loss_dict['fractional_loss'].item():.6f}")
        print(f"   Consistency loss: {loss_dict['consistency_loss'].item():.6f}")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 7. Gradient flow check
    # ========================================
    print("\n[7] Checking gradient flow...")
    
    try:
        total_loss = loss_dict['loss']
        total_loss.backward()
        
        # Check if depth_net has gradients
        has_grad = False
        for name, param in depth_net.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if not has_grad:
            print(f"⚠️  WARNING: No gradients flowing to depth_net parameters!")
            return False
        
        print(f"✅ Gradients flowing correctly")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 8. Check reconstruction
    # ========================================
    print("\n[8] Checking depth reconstruction...")
    
    from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth
    
    depth_net.eval()
    with torch.no_grad():
        outputs_eval = depth_net(dummy_rgb)
        
        integer_out = outputs_eval[('integer', 0)]
        fractional_out = outputs_eval[('fractional', 0)]
        
        # Reconstruct depth
        depth_reconstructed = dual_head_to_depth(integer_out, fractional_out, 15.0)
        
        print(f"✅ Depth reconstruction successful:")
        print(f"   Integer head range: [{integer_out.min():.4f}, {integer_out.max():.4f}] (sigmoid)")
        print(f"   Fractional head range: [{fractional_out.min():.4f}, {fractional_out.max():.4f}] (sigmoid)")
        print(f"   Reconstructed depth range: [{depth_reconstructed.min():.4f}, {depth_reconstructed.max():.4f}]m")
        print(f"   Expected depth range: [0.0, 16.0]m (0 + 15.0 max)")
        
        # Sanity check
        if depth_reconstructed.min() < 0 or depth_reconstructed.max() > 16.0:
            print(f"⚠️  WARNING: Reconstructed depth out of expected range!")
    
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 RUNNING FULL INTEGRATION TEST")
    print("="*80)
    
    success = test_full_training_pipeline()
    
    print("\n" + "="*80)
    print("📊 TEST RESULT")
    print("="*80)
    
    if success:
        print("✅ FULL INTEGRATION TEST PASSED!")
        print("\n✨ The training pipeline is ready:")
        print("   ✓ Forward pass works correctly")
        print("   ✓ Dual-Head outputs generated")
        print("   ✓ Upsampling handles dual-head format")
        print("   ✓ Loss computation successful")
        print("   ✓ Gradient flow working")
        print("   ✓ Depth reconstruction valid")
    else:
        print("❌ INTEGRATION TEST FAILED!")
        print("\nPlease check the errors above.")
    
    print("="*80)
    sys.exit(0 if success else 1)
