# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from packnet_sfm.utils.depth import inv2depth
from packnet_sfm.losses.loss_base import LossBase


class SSISilogLoss(LossBase):
    """
    🆕 Scale-Shift-Invariant + Silog combined loss
    
    Combines SSI loss (for relative accuracy) with Silog loss (for log-scale accuracy)
    to maintain scale-shift invariance while improving absolute depth accuracy.
    
    Parameters
    ----------
    alpha : float
        SSI loss parameter (default: 0.85)
    silog_ratio : float
        Silog ratio parameter (default: 10)
    silog_ratio2 : float
        Silog second ratio parameter (default: 0.85)
    ssi_weight : float
        Weight for SSI component (default: 0.7)
    silog_weight : float
        Weight for Silog component (default: 0.3)
    """
    def __init__(self, alpha=0.85, silog_ratio=10, silog_ratio2=0.85, 
                 ssi_weight=0.7, silog_weight=0.3):
        super().__init__()
        self.alpha = alpha
        self.silog_ratio = silog_ratio
        self.silog_ratio2 = silog_ratio2
        self.ssi_weight = ssi_weight
        self.silog_weight = silog_weight
        
        print(f"🎯 SSI-Silog Loss initialized:")
        print(f"   SSI weight: {ssi_weight}")
        print(f"   Silog weight: {silog_weight}")
        print(f"   Alpha: {alpha}")
        print(f"   Silog ratio: {silog_ratio}")

    def compute_ssi_loss(self, pred_inv_depth, gt_inv_depth, mask):
        """Compute SSI loss on inverse depth"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_inv_depth.device, requires_grad=True)
        
        diff = (pred_inv_depth[mask] - gt_inv_depth[mask])
        diff2 = diff ** 2
        mean = diff.mean()
        var = diff2.mean() - mean ** 2
        ssi_loss = var + self.alpha * mean ** 2
        return ssi_loss

    def compute_silog_loss(self, pred_inv_depth, gt_inv_depth, mask):
        """Compute Silog loss on depth"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_inv_depth.device, requires_grad=True)
        
        # 안전한 inverse depth to depth 변환
        pred_depth = inv2depth(pred_inv_depth)
        gt_depth = inv2depth(gt_inv_depth)
        
        # 안전한 범위로 클램프
        pred_depth = torch.clamp(pred_depth, min=1e-3, max=100.0)
        gt_depth = torch.clamp(gt_depth, min=1e-3, max=100.0)
        
        # 마스크된 픽셀에서만 계산
        pred_depth_masked = pred_depth[mask]
        gt_depth_masked = gt_depth[mask]
        
        # Silog 계산
        log_pred = torch.log(pred_depth_masked * self.silog_ratio)
        log_gt = torch.log(gt_depth_masked * self.silog_ratio)
        
        log_diff = log_pred - log_gt
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.silog_ratio2 * (log_diff.mean() ** 2)
        
        # 분산이 음수가 되지 않도록 보호
        silog_var = silog1 - silog2
        if silog_var < 0:
            silog_var = torch.abs(silog_var)
        
        silog_loss = torch.sqrt(silog_var + 1e-8) * self.silog_ratio
        return silog_loss

    def forward(self, pred_inv_depth, gt_inv_depth, mask=None):
        """
        Forward pass
        
        Parameters
        ----------
        pred_inv_depth : torch.Tensor [B,1,H,W]
            Predicted inverse depth
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground truth inverse depth
        mask : torch.Tensor [B,1,H,W], optional
            Valid pixel mask
            
        Returns
        -------
        loss : torch.Tensor
            Combined SSI + Silog loss
        """
        if mask is None:
            mask = (gt_inv_depth > 0)
        
        # 유효 픽셀 수 확인
        valid_pixels = mask.sum()
        if valid_pixels < 100:
            return torch.tensor(0.0, device=pred_inv_depth.device, requires_grad=True)
        
        # 입력 유효성 검사
        if torch.isnan(pred_inv_depth).any() or torch.isnan(gt_inv_depth).any():
            return torch.tensor(1.0, device=pred_inv_depth.device, requires_grad=True)
        
        # 개별 손실 계산
        ssi_loss = self.compute_ssi_loss(pred_inv_depth, gt_inv_depth, mask)
        silog_loss = self.compute_silog_loss(pred_inv_depth, gt_inv_depth, mask)
        
        # NaN 체크
        if torch.isnan(ssi_loss) or torch.isnan(silog_loss):
            return torch.tensor(1.0, device=pred_inv_depth.device, requires_grad=True)
        
        # 결합된 손실
        total_loss = self.ssi_weight * ssi_loss + self.silog_weight * silog_loss
        
        # 메트릭 저장
        self.add_metric('ssi_component', ssi_loss)
        self.add_metric('silog_component', silog_loss)
        self.add_metric('ssi_weight_used', self.ssi_weight)
        self.add_metric('silog_weight_used', self.silog_weight)
        
        return total_loss