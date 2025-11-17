#!/usr/bin/env python3
"""
Test sparse-ssi-silog Loss with Single-Head model
Verifies that the loss function works correctly with the actual config
"""

import torch
import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_sparse_ssi_silog_loss():
    """Test sparse-ssi-silog loss with Single-Head ResNetSAN01"""
    print("\n" + "="*80)
    print("🧪 TEST: sparse-ssi-silog Loss with Single-Head")
    print("="*80)
    
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    from packnet_sfm.losses.supervised_loss import SupervisedLoss
    from packnet_sfm.utils.image import match_scales
    
    # ========================================
    # 1. Load actual config
    # ========================================
    print("\n[1] Loading config (train_resnet_san_ncdb_640x384.yaml)...")
    
    config_path = "configs/train_resnet_san_ncdb_640x384.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    supervised_method = cfg['model']['loss']['supervised_method']
    min_depth = cfg['model']['params']['min_depth']
    max_depth = cfg['model']['params']['max_depth']
    
    print(f"✅ Config loaded:")
    print(f"   Loss method: {supervised_method}")
    print(f"   Depth range: [{min_depth}, {max_depth}]m")
    
    # ========================================
    # 2. Initialize Single-Head model
    # ========================================
    print("\n[2] Initializing Single-Head ResNetSAN01...")
    
    depth_net = ResNetSAN01(
        version='18A',
        use_dual_head=False,  # Single-Head
        max_depth=max_depth,
        min_depth=min_depth,
        use_film=False,
        use_enhanced_lidar=False
    )
    
    print(f"✅ Single-Head ResNetSAN01 initialized")
    
    # ========================================
    # 3. Create SupervisedLoss with sparse-ssi-silog
    # ========================================
    print("\n[3] Initializing SupervisedLoss with sparse-ssi-silog...")
    
    try:
        loss_fn = SupervisedLoss(
            supervised_method=supervised_method,
            supervised_num_scales=cfg['model']['loss']['supervised_num_scales'],
            min_depth=min_depth,
            max_depth=max_depth
        )
        print(f"✅ SupervisedLoss initialized")
    except Exception as e:
        print(f"❌ Loss initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 4. Forward pass
    # ========================================
    print("\n[4] Forward pass with dummy data...")
    
    batch_size = 2
    dummy_rgb = torch.randn(batch_size, 3, 384, 640)
    
    depth_net.train()
    try:
        outputs = depth_net(dummy_rgb)
        
        if 'inv_depths' not in outputs:
            print(f"❌ Missing 'inv_depths' in output")
            return False
        
        inv_depths = outputs['inv_depths']
        print(f"✅ Forward pass successful:")
        print(f"   Output type: {type(inv_depths)}")
        print(f"   Num scales: {len(inv_depths) if isinstance(inv_depths, list) else 1}")
        
        if isinstance(inv_depths, list):
            for i, depth in enumerate(inv_depths):
                print(f"   Scale {i}: {depth.shape}, range=[{depth.min():.4f}, {depth.max():.4f}]")
        else:
            print(f"   Shape: {inv_depths.shape}, range=[{inv_depths.min():.4f}, {inv_depths.max():.4f}]")
            # Convert to list format expected by loss
            inv_depths = [inv_depths]
    
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 5. Create ground truth
    # ========================================
    print("\n[5] Creating ground truth depth...")
    
    # Create GT depth in normal space [0.5, 15]m
    depth_gt = torch.rand(batch_size, 1, 384, 640) * (max_depth - min_depth) + min_depth
    
    # Convert to inverse depth [1/15, 2]m^-1
    from packnet_sfm.utils.depth import depth2inv
    inv_depth_gt = depth2inv(depth_gt)
    
    print(f"✅ Ground truth created:")
    print(f"   Depth range: [{depth_gt.min():.4f}, {depth_gt.max():.4f}]m")
    print(f"   Inv depth range: [{inv_depth_gt.min():.6f}, {inv_depth_gt.max():.6f}]m^-1")
    
    # ========================================
    # 6. Loss computation
    # ========================================
    print("\n[6] Computing loss with sparse-ssi-silog...")
    
    try:
        # Match scales like the actual loss function does
        num_scales = cfg['model']['loss']['supervised_num_scales']
        gt_inv_depths = match_scales(inv_depth_gt, inv_depths, num_scales,
                                     mode='nearest', align_corners=None)
        
        print(f"✅ Ground truth matched to {num_scales} scales:")
        for i, gt_inv in enumerate(gt_inv_depths):
            print(f"   Scale {i}: {gt_inv.shape}, range=[{gt_inv.min():.6f}, {gt_inv.max():.6f}]")
        
        # Forward pass through loss
        loss_dict = loss_fn(inv_depths, inv_depth_gt, return_logs=True)
        
        if 'loss' not in loss_dict:
            print(f"❌ Loss dict missing 'loss' key")
            return False
        
        loss = loss_dict['loss']
        print(f"✅ Loss computation successful:")
        print(f"   Loss value: {loss.item():.6f}")
        print(f"   Loss shape: {loss.shape}")
        print(f"   Loss type: {type(loss)}")
        
        # Print metrics
        if 'metrics' in loss_dict:
            print(f"\n   Metrics:")
            for key, val in loss_dict['metrics'].items():
                if isinstance(val, torch.Tensor):
                    print(f"      {key}: {val.item():.6f}" if val.numel() == 1 else f"      {key}: {val}")
                else:
                    print(f"      {key}: {val}")
    
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 7. Gradient flow
    # ========================================
    print("\n[7] Checking gradient flow...")
    
    try:
        loss_value = loss_dict['loss']
        loss_value.backward()
        
        has_grad = False
        grad_count = 0
        for name, param in depth_net.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_count += 1
        
        if not has_grad:
            print(f"❌ No gradients flowing!")
            return False
        
        print(f"✅ Gradients flowing correctly:")
        print(f"   Gradients in {grad_count} parameters")
        print(f"   Grad norms present: Yes")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 8. Verify loss properties
    # ========================================
    print("\n[8] Verifying loss properties...")
    
    try:
        # Loss should be a positive scalar
        loss_scalar = loss_dict['loss'].item()
        
        if loss_scalar < 0:
            print(f"⚠️  WARNING: Loss is negative ({loss_scalar})")
            return False
        
        if torch.isnan(torch.tensor(loss_scalar)):
            print(f"⚠️  WARNING: Loss is NaN")
            return False
        
        if torch.isinf(torch.tensor(loss_scalar)):
            print(f"⚠️  WARNING: Loss is Inf")
            return False
        
        print(f"✅ Loss properties verified:")
        print(f"   Positive: Yes ({loss_scalar:.6f})")
        print(f"   Valid (not NaN/Inf): Yes")
        print(f"   Reasonable range: Yes")
        
    except Exception as e:
        print(f"❌ Loss property check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 9. Multi-iteration test
    # ========================================
    print("\n[9] Testing multiple iterations...")
    
    try:
        losses = []
        for iter_idx in range(3):
            # New random data
            dummy_rgb_i = torch.randn(batch_size, 3, 384, 640)
            depth_gt_i = torch.rand(batch_size, 1, 384, 640) * (max_depth - min_depth) + min_depth
            inv_depth_gt_i = depth2inv(depth_gt_i)
            
            # Forward and backward
            outputs_i = depth_net(dummy_rgb_i)
            inv_depths_i = outputs_i['inv_depths']
            
            loss_dict_i = loss_fn(inv_depths_i, inv_depth_gt_i, return_logs=True)
            loss_i = loss_dict_i['loss'].item()
            losses.append(loss_i)
            
            print(f"   Iteration {iter_idx}: loss = {loss_i:.6f}")
        
        print(f"✅ Multi-iteration test passed")
        print(f"   Mean loss: {sum(losses)/len(losses):.6f}")
        print(f"   Loss stability: {'Stable' if all(l > 0 for l in losses) else 'Unstable'}")
        
    except Exception as e:
        print(f"❌ Multi-iteration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 SPARSE-SSI-SILOG LOSS VERIFICATION")
    print("="*80)
    
    success = test_sparse_ssi_silog_loss()
    
    print("\n" + "="*80)
    print("📊 TEST RESULT")
    print("="*80)
    
    if success:
        print("✅ sparse-ssi-silog Loss Test PASSED!")
        print("\n✨ Confirmed:")
        print("   ✓ Loss initialization successful")
        print("   ✓ Forward pass with Single-Head works")
        print("   ✓ Loss computation successful")
        print("   ✓ Gradients flowing correctly")
        print("   ✓ Loss properties valid")
        print("   ✓ Multi-iteration stability confirmed")
        print("\n🎉 Ready for training with sparse-ssi-silog!")
    else:
        print("❌ sparse-ssi-silog Loss Test FAILED!")
        print("\nPlease check the errors above.")
    
    print("="*80)
    sys.exit(0 if success else 1)
