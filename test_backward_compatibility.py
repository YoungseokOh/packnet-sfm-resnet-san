#!/usr/bin/env python3
"""
Comprehensive Backward Compatibility Test
Single-Head와 Dual-Head 모두 정상 작동하는지 확인
"""

import torch
import sys
import os
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_single_head_training():
    """Test Single-Head (기존 방식) training pipeline"""
    print("\n" + "="*80)
    print("🧪 TEST 1: SINGLE-HEAD (LEGACY) TRAINING PIPELINE")
    print("="*80)
    
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    from packnet_sfm.losses.supervised_loss import SupervisedLoss
    
    # ========================================
    # 1. Initialize Single-Head model
    # ========================================
    print("\n[1] Initializing Single-Head ResNetSAN01...")
    
    depth_net_single = ResNetSAN01(
        version='18A',
        use_dual_head=False,  # 🔴 Single-Head mode
        max_depth=15.0,
        min_depth=0.5,
        use_film=False,
        use_enhanced_lidar=False
    )
    
    print(f"✅ Single-Head ResNetSAN01 initialized (use_dual_head=False)")
    
    # ========================================
    # 2. Forward pass
    # ========================================
    print("\n[2] Forward pass with Single-Head model...")
    
    batch_size = 2
    dummy_rgb = torch.randn(batch_size, 3, 384, 640)
    depth_net_single.eval()
    
    with torch.no_grad():
        outputs = depth_net_single(dummy_rgb)
    
    print(f"✅ Forward pass completed")
    print(f"   Output type: {type(outputs)}")
    print(f"   Output keys: {list(outputs.keys())}")
    
    # ========================================
    # 3. Check output format
    # ========================================
    print("\n[3] Checking Single-Head output format...")
    
    if 'inv_depths' not in outputs:
        print(f"❌ Missing 'inv_depths' key in Single-Head output!")
        print(f"   Available keys: {list(outputs.keys())}")
        return False
    
    inv_depths = outputs['inv_depths']
    print(f"✅ Single-Head output format correct:")
    print(f"   inv_depths type: {type(inv_depths)}")
    
    if isinstance(inv_depths, list):
        print(f"   inv_depths is list with {len(inv_depths)} scales:")
        for i, depth in enumerate(inv_depths):
            print(f"      Scale {i}: {depth.shape}, range=[{depth.min():.4f}, {depth.max():.4f}]")
    else:
        print(f"   inv_depths shape: {inv_depths.shape}")
        print(f"   inv_depths range: [{inv_depths.min():.4f}, {inv_depths.max():.4f}]")
    
    # ========================================
    # 4. Loss computation
    # ========================================
    print("\n[4] Computing loss (Single-Head)...")
    
    # Create dummy ground truth
    depth_gt = torch.rand(batch_size, 1, 384, 640) * 10 + 2  # [2, 12]m
    
    # Initialize loss - should work with inv_depths format
    loss_fn = SupervisedLoss(method='sparse-l1', min_depth=0.5, max_depth=15.0)
    
    try:
        depth_net_single.train()
        outputs_train = depth_net_single(dummy_rgb)
        
        # For Single-Head, outputs should have 'inv_depths'
        if 'inv_depths' not in outputs_train:
            print(f"❌ Loss input format wrong: missing 'inv_depths'")
            return False
        
        loss_result = loss_fn(outputs_train['inv_depths'], depth_gt)
        
        # SupervisedLoss returns a dict with 'loss' key
        if isinstance(loss_result, dict):
            loss = loss_result['loss']
            print(f"✅ Loss computation successful:")
            print(f"   Total loss: {loss.item():.6f}")
            print(f"   Loss dict keys: {list(loss_result.keys())}")
        else:
            loss = loss_result
            print(f"✅ Loss computation successful:")
            print(f"   Loss value: {loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 5. Gradient flow
    # ========================================
    print("\n[5] Checking gradient flow (Single-Head)...")
    
    try:
        # Get the actual loss tensor
        if isinstance(loss_result, dict):
            loss_tensor = loss_result['loss']
        else:
            loss_tensor = loss_result
        
        loss_tensor.backward()
        
        has_grad = False
        for name, param in depth_net_single.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if not has_grad:
            print(f"❌ No gradients flowing to model!")
            return False
        
        print(f"✅ Gradients flowing correctly")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_dual_head_training():
    """Test Dual-Head (새로운 방식) training pipeline"""
    print("\n" + "="*80)
    print("🧪 TEST 2: DUAL-HEAD (NEW) TRAINING PIPELINE")
    print("="*80)
    
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    from packnet_sfm.losses.dual_head_depth_loss import DualHeadDepthLoss
    
    # ========================================
    # 1. Initialize Dual-Head model
    # ========================================
    print("\n[1] Initializing Dual-Head ResNetSAN01...")
    
    depth_net_dual = ResNetSAN01(
        version='18A',
        use_dual_head=True,  # 🟢 Dual-Head mode
        max_depth=15.0,
        min_depth=0.5,
        use_film=False,
        use_enhanced_lidar=False
    )
    
    print(f"✅ Dual-Head ResNetSAN01 initialized (use_dual_head=True)")
    
    # ========================================
    # 2. Forward pass
    # ========================================
    print("\n[2] Forward pass with Dual-Head model...")
    
    batch_size = 2
    dummy_rgb = torch.randn(batch_size, 3, 384, 640)
    depth_net_dual.eval()
    
    with torch.no_grad():
        outputs = depth_net_dual(dummy_rgb)
    
    print(f"✅ Forward pass completed")
    print(f"   Output type: {type(outputs)}")
    print(f"   Output keys: {list(outputs.keys())}")
    
    # ========================================
    # 3. Check output format
    # ========================================
    print("\n[3] Checking Dual-Head output format...")
    
    has_integer = any(k[0] == 'integer' for k in outputs.keys() if isinstance(k, tuple))
    has_fractional = any(k[0] == 'fractional' for k in outputs.keys() if isinstance(k, tuple))
    
    if not has_integer or not has_fractional:
        print(f"❌ Missing dual-head keys!")
        print(f"   has_integer: {has_integer}, has_fractional: {has_fractional}")
        return False
    
    print(f"✅ Dual-Head output format correct:")
    print(f"   Integer heads: {[k for k in outputs.keys() if isinstance(k, tuple) and k[0] == 'integer']}")
    print(f"   Fractional heads: {[k for k in outputs.keys() if isinstance(k, tuple) and k[0] == 'fractional']}")
    
    # ========================================
    # 4. Loss computation
    # ========================================
    print("\n[4] Computing loss (Dual-Head)...")
    
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
        depth_net_dual.train()
        outputs_train = depth_net_dual(dummy_rgb)
        
        loss_dict = loss_fn(outputs_train, depth_gt, return_logs=True)
        
        if 'loss' not in loss_dict:
            print(f"❌ Loss dict missing 'loss' key")
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
    # 5. Gradient flow
    # ========================================
    print("\n[5] Checking gradient flow (Dual-Head)...")
    
    try:
        loss_dict['loss'].backward()
        
        has_grad = False
        for name, param in depth_net_dual.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if not has_grad:
            print(f"❌ No gradients flowing to model!")
            return False
        
        print(f"✅ Gradients flowing correctly")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_config_loading():
    """Test loading configs and model initialization"""
    print("\n" + "="*80)
    print("🧪 TEST 3: CONFIG LOADING AND MODEL INITIALIZATION")
    print("="*80)
    
    from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
    
    # ========================================
    # 1. Load Single-Head config
    # ========================================
    print("\n[1] Loading Single-Head config (train_resnet_san_ncdb_640x384.yaml)...")
    
    config_path_single = "configs/train_resnet_san_ncdb_640x384.yaml"
    
    if not os.path.exists(config_path_single):
        print(f"❌ Config not found: {config_path_single}")
        return False
    
    with open(config_path_single, 'r') as f:
        cfg_single = yaml.safe_load(f)
    
    use_dual_head_single = cfg_single['model']['depth_net'].get('use_dual_head', False)
    print(f"✅ Config loaded: use_dual_head = {use_dual_head_single}")
    
    # ========================================
    # 2. Load Dual-Head config
    # ========================================
    print("\n[2] Loading Dual-Head config (train_resnet_san_ncdb_dual_head_640x384.yaml)...")
    
    config_path_dual = "configs/train_resnet_san_ncdb_dual_head_640x384.yaml"
    
    if not os.path.exists(config_path_dual):
        print(f"❌ Config not found: {config_path_dual}")
        return False
    
    with open(config_path_dual, 'r') as f:
        cfg_dual = yaml.safe_load(f)
    
    use_dual_head_dual = cfg_dual['model']['depth_net'].get('use_dual_head', False)
    print(f"✅ Config loaded: use_dual_head = {use_dual_head_dual}")
    
    # ========================================
    # 3. Initialize models from configs
    # ========================================
    print("\n[3] Initializing models from configs...")
    
    try:
        # Single-Head from config
        depth_cfg_single = cfg_single['model']['depth_net']
        model_single = ResNetSAN01(
            version=depth_cfg_single.get('version', '18A'),
            use_dual_head=depth_cfg_single.get('use_dual_head', False),
            max_depth=cfg_single['model']['params']['max_depth'],
            min_depth=cfg_single['model']['params']['min_depth'],
            use_film=depth_cfg_single.get('use_film', False),
            use_enhanced_lidar=depth_cfg_single.get('use_enhanced_lidar', False)
        )
        print(f"✅ Single-Head model initialized from config")
        
        # Dual-Head from config
        depth_cfg_dual = cfg_dual['model']['depth_net']
        model_dual = ResNetSAN01(
            version=depth_cfg_dual.get('version', '18A'),
            use_dual_head=depth_cfg_dual.get('use_dual_head', False),
            max_depth=cfg_dual['model']['params']['max_depth'],
            min_depth=cfg_dual['model']['params']['min_depth'],
            use_film=depth_cfg_dual.get('use_film', False),
            use_enhanced_lidar=depth_cfg_dual.get('use_enhanced_lidar', False)
        )
        print(f"✅ Dual-Head model initialized from config")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 4. Compare configs
    # ========================================
    print("\n[4] Comparing Single-Head vs Dual-Head configs...")
    
    differences = {
        'use_dual_head': (use_dual_head_single, use_dual_head_dual),
        'supervised_method': (
            cfg_single['model']['loss'].get('supervised_method'),
            cfg_dual['model']['loss'].get('supervised_method')
        ),
        'depth_lr': (
            cfg_single['model']['optimizer']['depth'].get('lr'),
            cfg_dual['model']['optimizer']['depth'].get('lr')
        ),
    }
    
    print("✅ Config differences:")
    for key, (single_val, dual_val) in differences.items():
        marker = "✓ Different" if single_val != dual_val else "- Same"
        print(f"   {marker}: {key}")
        print(f"      Single: {single_val}")
        print(f"      Dual:   {dual_val}")
    
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 BACKWARD COMPATIBILITY TEST SUITE")
    print("Testing Single-Head (Legacy) and Dual-Head (New) Coexistence")
    print("="*80)
    
    results = {}
    
    # Test 1: Single-Head
    print("\n" + "="*80)
    results['single_head'] = test_single_head_training()
    
    # Test 2: Dual-Head
    print("\n" + "="*80)
    results['dual_head'] = test_dual_head_training()
    
    # Test 3: Config Loading
    print("\n" + "="*80)
    results['config_loading'] = test_config_loading()
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL TEST SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print("\n" + "="*80)
    
    if all_passed:
        print("🎉 ALL BACKWARD COMPATIBILITY TESTS PASSED!")
        print("\n✨ Summary:")
        print("   ✓ Single-Head (legacy) mode works correctly")
        print("   ✓ Dual-Head (new) mode works correctly")
        print("   ✓ Both modes can coexist with different configs")
        print("   ✓ Output formats are preserved correctly")
        print("\n✅ Ready for training with either Single-Head or Dual-Head!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
    
    print("="*80)
    sys.exit(0 if all_passed else 1)
