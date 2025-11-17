#!/usr/bin/env python3
"""
Test SemiSupCompletionModel with Dual-Head output handling
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_semisup_model_with_dualhead():
    """Test SemiSupCompletionModel forward with Dual-Head"""
    print("\n" + "="*80)
    print("🧪 TEST: SemiSupCompletionModel with Dual-Head")
    print("="*80)
    
    from packnet_sfm.models.SemiSupCompletionModel import SemiSupCompletionModel
    import yaml
    
    # ========================================
    # 1. Load config
    # ========================================
    print("\n[1] Loading Dual-Head config...")
    
    config_path = "configs/train_resnet_san_ncdb_dual_head_640x384.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"✅ Config loaded")
    
    # ========================================
    # 2. Create model
    # ========================================
    print("\n[2] Creating SemiSupCompletionModel...")
    
    try:
        model = SemiSupCompletionModel(
            pretrained=False,
            depth_net_cfg=cfg['model']['depth_net'],
            loss_cfg=cfg['model']['loss'],
            min_depth=cfg['model']['params']['min_depth'],
            max_depth=cfg['model']['params']['max_depth'],
            use_log_space=cfg['model']['params'].get('use_log_space', False)
        )
        print(f"✅ SemiSupCompletionModel created")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 3. Create dummy batch
    # ========================================
    print("\n[3] Creating dummy batch...")
    
    batch = {
        'rgb': torch.randn(2, 3, 384, 640),
        'depth': torch.rand(2, 1, 384, 640) * 10 + 2,  # [2, 12]m
    }
    
    print(f"✅ Batch created:")
    print(f"   RGB: {batch['rgb'].shape}")
    print(f"   Depth: {batch['depth'].shape}")
    
    # ========================================
    # 4. Forward pass (training mode)
    # ========================================
    print("\n[4] Forward pass (training mode)...")
    
    model.train()
    try:
        output = model(batch, return_logs=False, progress=0.0)
        
        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(output.keys())}")
        
        if 'loss' not in output:
            print(f"❌ Missing 'loss' key in output")
            return False
        
        loss = output['loss']
        print(f"   Loss: {loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 5. Backward pass
    # ========================================
    print("\n[5] Backward pass...")
    
    try:
        loss.backward()
        
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if not has_grad:
            print(f"❌ No gradients flowing!")
            return False
        
        print(f"✅ Backward pass successful")
        print(f"   Gradients flowing: Yes")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_semisup_model_with_singlehead():
    """Test SemiSupCompletionModel forward with Single-Head"""
    print("\n" + "="*80)
    print("🧪 TEST: SemiSupCompletionModel with Single-Head")
    print("="*80)
    
    from packnet_sfm.models.SemiSupCompletionModel import SemiSupCompletionModel
    import yaml
    
    # ========================================
    # 1. Load config
    # ========================================
    print("\n[1] Loading Single-Head config...")
    
    config_path = "configs/train_resnet_san_ncdb_640x384.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"✅ Config loaded")
    
    # ========================================
    # 2. Create model
    # ========================================
    print("\n[2] Creating SemiSupCompletionModel...")
    
    try:
        model = SemiSupCompletionModel(
            pretrained=False,
            depth_net_cfg=cfg['model']['depth_net'],
            loss_cfg=cfg['model']['loss'],
            min_depth=cfg['model']['params']['min_depth'],
            max_depth=cfg['model']['params']['max_depth'],
            use_log_space=cfg['model']['params'].get('use_log_space', False)
        )
        print(f"✅ SemiSupCompletionModel created")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 3. Create dummy batch
    # ========================================
    print("\n[3] Creating dummy batch...")
    
    batch = {
        'rgb': torch.randn(2, 3, 384, 640),
        'depth': torch.rand(2, 1, 384, 640) * 10 + 2,  # [2, 12]m
    }
    
    print(f"✅ Batch created")
    
    # ========================================
    # 4. Forward pass (training mode)
    # ========================================
    print("\n[4] Forward pass (training mode)...")
    
    model.train()
    try:
        output = model(batch, return_logs=False, progress=0.0)
        
        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(output.keys())}")
        
        if 'loss' not in output:
            print(f"❌ Missing 'loss' key in output")
            return False
        
        loss = output['loss']
        print(f"   Loss: {loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================
    # 5. Backward pass
    # ========================================
    print("\n[5] Backward pass...")
    
    try:
        loss.backward()
        
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if not has_grad:
            print(f"❌ No gradients flowing!")
            return False
        
        print(f"✅ Backward pass successful")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 SEMISUPCOMPLETIONMODEL DUAL-HEAD FIX TEST")
    print("="*80)
    
    results = {}
    
    # Test 1: Dual-Head
    print("\n" + "="*80)
    results['dual_head'] = test_semisup_model_with_dualhead()
    
    # Test 2: Single-Head
    print("\n" + "="*80)
    results['single_head'] = test_semisup_model_with_singlehead()
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    if all_passed:
        print("\n🎉 All tests PASSED!")
        print("\n✨ SemiSupCompletionModel now supports both Single-Head and Dual-Head!")
    else:
        print("\n❌ Some tests FAILED!")
    
    print("="*80)
    sys.exit(0 if all_passed else 1)
