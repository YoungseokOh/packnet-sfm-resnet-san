#!/usr/bin/env python3
"""
Loss Scale Analysis for Direct Depth Training

Compares loss magnitudes between Bounded Inverse and Direct Linear modes.
"""

import sys
sys.path.append('/workspace/packnet-sfm')

import torch
from packnet_sfm.losses.ssi_silog_loss import SSISilogLoss

def analyze_loss_scale():
    """Analyze loss scale for both modes"""
    
    min_depth, max_depth = 0.5, 15.0
    batch_size = 4
    h, w = 384, 640
    
    # Create realistic synthetic data
    # Simulate typical depth distribution (more pixels near 5-10m range)
    gt_depth = torch.randn(batch_size, 1, h, w) * 3.0 + 7.0  # Mean ~7m, std ~3m
    gt_depth = torch.clamp(gt_depth, min_depth, max_depth)
    
    # Add realistic noise to prediction (±10% error)
    pred_depth = gt_depth * (1.0 + torch.randn_like(gt_depth) * 0.1)
    pred_depth = torch.clamp(pred_depth, min_depth, max_depth)
    
    mask = torch.ones_like(gt_depth, dtype=torch.bool)
    
    print("=" * 80)
    print("📊 Loss Scale Analysis: Bounded Inverse vs Direct Linear")
    print("=" * 80)
    
    print(f"\n📐 Input Statistics:")
    print(f"   GT depth range: [{gt_depth.min():.2f}, {gt_depth.max():.2f}]m")
    print(f"   GT depth mean: {gt_depth.mean():.2f}m ± {gt_depth.std():.2f}m")
    print(f"   Pred depth range: [{pred_depth.min():.2f}, {pred_depth.max():.2f}]m")
    print(f"   Pred depth mean: {pred_depth.mean():.2f}m ± {pred_depth.std():.2f}m")
    print(f"   Absolute error: {(pred_depth - gt_depth).abs().mean():.3f}m")
    print(f"   Relative error: {((pred_depth - gt_depth).abs() / gt_depth).mean()*100:.2f}%")
    
    # ============================================================
    # Test 1: inv_depth Mode (Legacy - Bounded Inverse)
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 Test 1: inv_depth Mode (Bounded Inverse - Legacy)")
    print("=" * 80)
    
    loss_inv = SSISilogLoss(
        min_depth=min_depth,
        max_depth=max_depth,
        input_mode='inv_depth',
        ssi_weight=0.7,
        silog_weight=0.3
    )
    
    # Convert to inv_depth
    pred_inv = 1.0 / (pred_depth + 1e-8)
    gt_inv = 1.0 / (gt_depth + 1e-8)
    
    print(f"\n📐 Inverse Depth Statistics:")
    print(f"   GT inv_depth range: [{gt_inv.min():.4f}, {gt_inv.max():.4f}]")
    print(f"   GT inv_depth mean: {gt_inv.mean():.4f}")
    print(f"   Pred inv_depth range: [{pred_inv.min():.4f}, {pred_inv.max():.4f}]")
    print(f"   Pred inv_depth mean: {pred_inv.mean():.4f}")
    
    loss_value_inv = loss_inv(pred_inv, gt_inv, mask)
    
    print(f"\n✅ inv_depth Mode Loss:")
    print(f"   Total Loss: {loss_value_inv.item():.6f}")
    print(f"   SSI component: {loss_inv.metrics.get('ssi_component', 0):.6f}")
    print(f"   Silog component: {loss_inv.metrics.get('silog_component', 0):.6f}")
    print(f"   Weighted: 0.7×{loss_inv.metrics.get('ssi_component', 0):.6f} + 0.3×{loss_inv.metrics.get('silog_component', 0):.6f}")
    
    # ============================================================
    # Test 2: depth Mode (Direct Linear - NEW)
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 Test 2: depth Mode (Direct Linear - NEW)")
    print("=" * 80)
    
    loss_depth = SSISilogLoss(
        min_depth=min_depth,
        max_depth=max_depth,
        input_mode='depth',
        ssi_weight=0.7,
        silog_weight=0.3
    )
    
    loss_value_depth = loss_depth(pred_depth, gt_depth, mask)
    
    print(f"\n✅ depth Mode Loss:")
    print(f"   Total Loss: {loss_value_depth.item():.6f}")
    print(f"   SSI component: {loss_depth.metrics.get('ssi_component', 0):.6f}")
    print(f"   Silog component: {loss_depth.metrics.get('silog_component', 0):.6f}")
    print(f"   Weighted: 0.7×{loss_depth.metrics.get('ssi_component', 0):.6f} + 0.3×{loss_depth.metrics.get('silog_component', 0):.6f}")
    
    # ============================================================
    # Comparison
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 Loss Comparison")
    print("=" * 80)
    
    ratio = loss_value_inv.item() / max(loss_value_depth.item(), 1e-8)
    
    print(f"\n✅ Loss Ratio Analysis:")
    print(f"   inv_depth mode: {loss_value_inv.item():.6f}")
    print(f"   depth mode: {loss_value_depth.item():.6f}")
    print(f"   Ratio (inv/depth): {ratio:.2f}x")
    
    if ratio > 2.0:
        print(f"\n⚠️  inv_depth mode loss is {ratio:.1f}x HIGHER!")
        print(f"   This is EXPECTED due to inv_depth space characteristics:")
        print(f"   - inv_depth has larger variance (range [{gt_inv.min():.4f}, {gt_inv.max():.4f}])")
        print(f"   - depth has smaller variance (range [{gt_depth.min():.2f}, {gt_depth.max():.2f}])")
        print(f"   - SSI loss magnitude depends on input scale")
    elif ratio < 0.5:
        print(f"\n⚠️  depth mode loss is {1/ratio:.1f}x HIGHER!")
    else:
        print(f"\n✅ Loss scales are comparable")
    
    # ============================================================
    # Expected Training Loss Range
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 Expected Training Loss Range")
    print("=" * 80)
    
    print(f"\n✅ Direct Depth Mode (depth input):")
    print(f"   Initial random loss: ~{loss_value_depth.item():.2f} (untrained model)")
    print(f"   Expected converged loss: ~0.01-0.02")
    print(f"   Target abs_rel: <0.035")
    
    print(f"\n✅ Bounded Inverse Mode (inv_depth input):")
    print(f"   Initial random loss: ~{loss_value_inv.item():.2f} (untrained model)")
    print(f"   Expected converged loss: ~0.01-0.02")
    print(f"   Target abs_rel: <0.030")
    
    print(f"\n🎯 Current Training Loss: ~8.27")
    if loss_value_depth.item() < 1.0:
        print(f"   → This is {8.27/loss_value_depth.item():.1f}x HIGHER than synthetic test!")
        print(f"   → Possible causes:")
        print(f"      1. Model not converged yet (early training)")
        print(f"      2. Real data has larger errors than synthetic")
        print(f"      3. Batch normalization not stabilized")
        print(f"   → Solution: Continue training, loss should decrease")
    
    # ============================================================
    # Gradient Analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 Gradient Scale Analysis")
    print("=" * 80)
    
    # Compute gradients
    pred_depth_grad = pred_depth.clone().detach().requires_grad_(True)
    gt_depth_grad = gt_depth.clone().detach()
    
    loss_for_grad = loss_depth(pred_depth_grad, gt_depth_grad, mask)
    loss_for_grad.backward()
    
    grad_magnitude = pred_depth_grad.grad.abs().mean()
    grad_max = pred_depth_grad.grad.abs().max()
    
    print(f"\n✅ Gradient Statistics (Direct Depth Mode):")
    print(f"   Mean gradient magnitude: {grad_magnitude:.6f}")
    print(f"   Max gradient magnitude: {grad_max:.6f}")
    print(f"   Gradient-to-loss ratio: {grad_magnitude / loss_value_depth.item():.4f}")
    
    if grad_magnitude > 1.0:
        print(f"\n⚠️  Gradients are large (>{grad_magnitude:.2f})")
        print(f"   Consider using gradient clipping (already set to 10.0 in config)")
    else:
        print(f"\n✅ Gradients are stable (<{grad_magnitude:.2f})")


if __name__ == '__main__':
    print("\n🔬 Loss Scale Analysis for Direct Depth Training\n")
    analyze_loss_scale()
    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)
