# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling


class EnhancedSSILoss(LossBase):
    """
    🆕 Enhanced Scale-Shift-Invariant depth loss with L1 regularization
    
    Combines SSI loss (for relative accuracy) with L1 loss (for absolute accuracy)
    to maintain the benefits of Enhanced LiDAR while improving RMSE performance.
    
    Parameters
    ----------
    alpha : float
        SSI loss parameter (default: 0.85)
    l1_weight : float
        Weight for L1 component (default: 0.2)
    ssi_weight : float
        Weight for SSI component (default: 0.8)
    adaptive_weighting : bool
        Whether to use adaptive weighting based on training progress
    """
    def __init__(self, alpha=0.85, l1_weight=0.2, ssi_weight=0.8, adaptive_weighting=True):
        super().__init__()
        self.alpha = alpha
        self.l1_weight = l1_weight
        self.ssi_weight = ssi_weight
        self.adaptive_weighting = adaptive_weighting
        
        # L1 loss for absolute accuracy
        self.l1_loss = nn.L1Loss(reduction='none')
        
        print(f"🎯 Enhanced SSI Loss initialized:")
        print(f"   SSI weight: {ssi_weight}")
        print(f"   L1 weight: {l1_weight}")
        print(f"   Adaptive weighting: {adaptive_weighting}")

    def compute_ssi_loss(self, pred_inv_depth, gt_inv_depth, mask):
        """Compute original SSI loss"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_inv_depth.device, requires_grad=True)
        
        diff = (pred_inv_depth[mask] - gt_inv_depth[mask])
        diff2 = diff ** 2
        mean = diff.mean()
        var = diff2.mean() - mean ** 2
        ssi_loss = var + self.alpha * mean ** 2
        return ssi_loss

    def compute_l1_loss(self, pred_inv_depth, gt_inv_depth, mask):
        """Compute L1 loss for absolute accuracy"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_inv_depth.device, requires_grad=True)
        
        # Convert to depth for L1 loss (more intuitive for absolute accuracy)
        pred_depth = 1.0 / (pred_inv_depth + 1e-6)
        gt_depth = 1.0 / (gt_inv_depth + 1e-6)
        
        l1_loss = self.l1_loss(pred_depth, gt_depth)
        l1_loss = l1_loss[mask].mean()
        return l1_loss

    def get_adaptive_weights(self, progress=None):
        """
        🆕 Get adaptive weights based on training progress
        
        Early training: Focus more on relative accuracy (SSI)
        Later training: Balance relative and absolute accuracy
        """
        if not self.adaptive_weighting or progress is None:
            return self.ssi_weight, self.l1_weight
        
        # Progress: 0.0 (start) -> 1.0 (end)
        # Early training: More SSI weight
        # Later training: More balanced
        progress = max(0.0, min(1.0, progress))
        
        # Dynamic weighting: Start with more SSI, gradually increase L1
        ssi_weight = self.ssi_weight + (1.0 - progress) * 0.1  # 최대 0.9
        l1_weight = self.l1_weight + progress * 0.1             # 최대 0.3
        
        # Normalize to maintain total weight = 1.0
        total_weight = ssi_weight + l1_weight
        ssi_weight = ssi_weight / total_weight
        l1_weight = l1_weight / total_weight
        
        return ssi_weight, l1_weight

    def forward(self, pred_inv_depth, gt_inv_depth, mask=None, progress=None):
        """
        🆕 Enhanced forward pass with multi-objective loss
        
        Parameters
        ----------
        pred_inv_depth : torch.Tensor [B,1,H,W]
            Predicted inverse depth
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground truth inverse depth
        mask : torch.Tensor [B,1,H,W], optional
            Valid pixel mask
        progress : float, optional
            Training progress (0.0 to 1.0) for adaptive weighting
            
        Returns
        -------
        loss : torch.Tensor
            Combined SSI + L1 loss
        """
        if mask is None:
            mask = (gt_inv_depth > 0)
        
        # Compute individual losses
        ssi_loss = self.compute_ssi_loss(pred_inv_depth, gt_inv_depth, mask)
        l1_loss = self.compute_l1_loss(pred_inv_depth, gt_inv_depth, mask)
        
        # Get weights (adaptive or fixed)
        ssi_weight, l1_weight = self.get_adaptive_weights(progress)
        
        # Combined loss
        total_loss = ssi_weight * ssi_loss + l1_weight * l1_loss
        
        # Store individual losses for monitoring
        self.add_metric('ssi_component', ssi_loss)
        self.add_metric('l1_component', l1_loss)
        self.add_metric('ssi_weight_used', ssi_weight)
        self.add_metric('l1_weight_used', l1_weight)
        
        return total_loss


class ProgressiveEnhancedSSILoss(LossBase):
    """
    🆕 Progressive version that starts with pure SSI and gradually adds L1
    
    This allows the Enhanced LiDAR features to learn relative relationships first,
    then gradually improve absolute accuracy.
    """
    def __init__(self, alpha=0.85, max_l1_weight=0.3, transition_epochs=15):
        super().__init__()
        self.alpha = alpha
        self.max_l1_weight = max_l1_weight
        self.transition_epochs = transition_epochs
        
        self.l1_loss = nn.L1Loss(reduction='none')
        
        print(f"🚀 Progressive Enhanced SSI Loss initialized:")
        print(f"   Max L1 weight: {max_l1_weight}")
        print(f"   Transition epochs: {transition_epochs}")

    def get_progressive_weights(self, epoch=0):
        """Get weights based on current epoch"""
        if epoch >= self.transition_epochs:
            l1_weight = self.max_l1_weight
        else:
            # Linear increase from 0 to max_l1_weight
            l1_weight = (epoch / self.transition_epochs) * self.max_l1_weight
        
        ssi_weight = 1.0 - l1_weight
        return ssi_weight, l1_weight

    def forward(self, pred_inv_depth, gt_inv_depth, mask=None, epoch=0):
        """Progressive forward pass"""
        if mask is None:
            mask = (gt_inv_depth > 0)
        
        # SSI loss
        if mask.sum() == 0:
            ssi_loss = torch.tensor(0.0, device=pred_inv_depth.device, requires_grad=True)
        else:
            diff = (pred_inv_depth[mask] - gt_inv_depth[mask])
            diff2 = diff ** 2
            mean = diff.mean()
            var = diff2.mean() - mean ** 2
            ssi_loss = var + self.alpha * mean ** 2
        
        # L1 loss
        if mask.sum() == 0:
            l1_loss = torch.tensor(0.0, device=pred_inv_depth.device, requires_grad=True)
        else:
            pred_depth = 1.0 / (pred_inv_depth + 1e-6)
            gt_depth = 1.0 / (gt_inv_depth + 1e-6)
            l1_loss = self.l1_loss(pred_depth, gt_depth)[mask].mean()
        
        # Progressive weighting
        ssi_weight, l1_weight = self.get_progressive_weights(epoch)
        total_loss = ssi_weight * ssi_loss + l1_weight * l1_loss
        
        # Metrics
        self.add_metric('ssi_component', ssi_loss)
        self.add_metric('l1_component', l1_loss)
        self.add_metric('ssi_weight', ssi_weight)
        self.add_metric('l1_weight', l1_weight)
        
        return total_loss