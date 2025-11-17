# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Dual-Head Depth Loss for Integer-Fractional prediction

이 Loss는 기존 SupervisedLoss와 동일한 인터페이스를 유지하면서,
Integer/Fractional 헤드를 별도로 학습합니다.

Key Features:
- Integer Loss: 정수부 예측 (L1 loss, coarse prediction)
- Fractional Loss: 소수부 예측 (L1 loss, fine prediction) - 높은 가중치!
- Consistency Loss: 복원된 전체 깊이의 일관성 (L1 loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from packnet_sfm.losses.loss_base import LossBase
from packnet_sfm.networks.layers.resnet.layers import decompose_depth, dual_head_to_depth


class DualHeadDepthLoss(LossBase):
    """
    Integer-Fractional Dual-Head Depth Loss
    
    이 Loss는 세 가지 컴포넌트로 구성됩니다:
    1. Integer Loss: 정수부 예측 (L1 loss)
    2. Fractional Loss: 소수부 예측 (L1 loss, 높은 가중치)
    3. Consistency Loss: 복원된 깊이의 일관성 (L1 loss)
    
    Parameters
    ----------
    max_depth : float
        Maximum depth for integer normalization (default: 15.0)
    integer_weight : float
        Weight for integer loss (default: 1.0)
    fractional_weight : float
        Weight for fractional loss (default: 10.0) - 정밀도 핵심!
    consistency_weight : float
        Weight for consistency loss (default: 0.5)
    min_depth : float
        Minimum valid depth (default: 0.5)
    """
    
    def __init__(self, max_depth=15.0, 
                 integer_weight=1.0, 
                 fractional_weight=10.0,
                 consistency_weight=0.5,
                 min_depth=0.5,
                 **kwargs):
        super().__init__()
        
        # 🆕 파라미터 검증 (Critical!)
        assert max_depth > min_depth, \
            f"max_depth ({max_depth}) must be > min_depth ({min_depth})"
        assert max_depth > 0, \
            f"max_depth must be positive, got {max_depth}"
        assert min_depth >= 0, \
            f"min_depth must be non-negative, got {min_depth}"
        assert integer_weight >= 0, \
            f"integer_weight must be non-negative, got {integer_weight}"
        assert fractional_weight > 0, \
            f"fractional_weight must be positive (핵심!), got {fractional_weight}"
        assert consistency_weight >= 0, \
            f"consistency_weight must be non-negative, got {consistency_weight}"
        
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.integer_weight = integer_weight
        self.fractional_weight = fractional_weight
        self.consistency_weight = consistency_weight
        
        print(f"🎯 DualHeadDepthLoss initialized:")
        print(f"   Max depth: {max_depth}m")
        print(f"   Min depth: {min_depth}m")
        print(f"   Integer weight: {integer_weight}")
        print(f"   Fractional weight: {fractional_weight} (high precision!)")
        print(f"   Consistency weight: {consistency_weight}")
        print(f"   ✅ All parameters validated")
    
    def forward(self, outputs, depth_gt, return_logs=False, progress=0.0):
        """
        Compute dual-head depth loss
        
        Parameters
        ----------
        outputs : dict
            Model outputs containing:
            - ("integer", 0): [B, 1, H, W] sigmoid [0, 1]
            - ("fractional", 0): [B, 1, H, W] sigmoid [0, 1]
        depth_gt : torch.Tensor [B, 1, H, W]
            Ground truth depth
        return_logs : bool
            Whether to return detailed logs
        progress : float
            Training progress [0, 1] for dynamic weighting
        
        Returns
        -------
        loss_dict : dict
            {
                'loss': total_loss,
                'integer_loss': ...,
                'fractional_loss': ...,
                'consistency_loss': ...
            }
        """
        # Resize GT to match prediction size
        if depth_gt.shape[-2:] != outputs[("integer", 0)].shape[-2:]:
            depth_gt = F.interpolate(
                depth_gt, 
                size=outputs[("integer", 0)].shape[-2:],
                mode='nearest'
            )
        
        # Create valid mask
        mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
        
        if mask.sum() == 0:
            # No valid pixels
            return {
                'loss': torch.tensor(0.0, device=depth_gt.device, requires_grad=True),
                'integer_loss': torch.tensor(0.0, device=depth_gt.device),
                'fractional_loss': torch.tensor(0.0, device=depth_gt.device),
                'consistency_loss': torch.tensor(0.0, device=depth_gt.device)
            }
        
        # ========================================
        # 1. Decompose GT depth
        # ========================================
        # 🆕 PTQ: Use 256-level quantization for integer part
        integer_gt, fractional_gt = decompose_depth(depth_gt, self.max_depth, n_integer_levels=256)
        
        # ========================================
        # 2. Integer Loss (coarse prediction)
        # ========================================
        integer_pred = outputs[("integer", 0)]
        integer_loss = F.l1_loss(
            integer_pred[mask],
            integer_gt[mask],
            reduction='mean'
        )
        
        # ========================================
        # 3. Fractional Loss (fine prediction) - 핵심!
        # ========================================
        fractional_pred = outputs[("fractional", 0)]
        fractional_loss = F.l1_loss(
            fractional_pred[mask],
            fractional_gt[mask],
            reduction='mean'
        )
        
        # ========================================
        # 4. Consistency Loss (전체 깊이 일관성)
        # ========================================
        # 🆕 PTQ: Use 256-level quantization
        depth_pred = dual_head_to_depth(integer_pred, fractional_pred, self.max_depth, n_integer_levels=256)
        consistency_loss = F.l1_loss(
            depth_pred[mask],
            depth_gt[mask],
            reduction='mean'
        )
        
        # ========================================
        # 5. Total Loss (가중치 적용)
        # ========================================
        total_loss = (
            self.integer_weight * integer_loss +
            self.fractional_weight * fractional_loss +
            self.consistency_weight * consistency_loss
        )
        
        # Metrics for logging
        if return_logs:
            self.add_metric('integer_loss', integer_loss)
            self.add_metric('fractional_loss', fractional_loss)
            self.add_metric('consistency_loss', consistency_loss)
            self.add_metric('total_loss', total_loss)
            
            # Additional metrics
            with torch.no_grad():
                # Depth error
                depth_error = torch.abs(depth_pred[mask] - depth_gt[mask])
                self.add_metric('mean_depth_error', depth_error.mean())
                self.add_metric('median_depth_error', depth_error.median())
                
                # Integer accuracy (within 1 meter)
                integer_error = torch.abs(integer_pred[mask] * self.max_depth - integer_gt[mask] * self.max_depth)
                integer_acc = (integer_error < 1.0).float().mean()
                self.add_metric('integer_accuracy', integer_acc)
                
                # Fractional precision
                frac_error = torch.abs(fractional_pred[mask] - fractional_gt[mask])
                self.add_metric('fractional_rmse', torch.sqrt((frac_error ** 2).mean()))
        
        return {
            'loss': total_loss,
            'integer_loss': integer_loss.detach(),
            'fractional_loss': fractional_loss.detach(),
            'consistency_loss': consistency_loss.detach()
        }
