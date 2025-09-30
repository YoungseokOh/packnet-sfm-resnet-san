# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import torch.nn as nn
from typing import Optional
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
                 ssi_weight=0.7, silog_weight=0.3,
                #  ssi_weight=1.0, silog_weight=0.0,
                 min_depth: Optional[float] = None, max_depth: Optional[float] = None):
        super().__init__()
        self.alpha = alpha
        self.silog_ratio = silog_ratio
        self.silog_ratio2 = silog_ratio2
        self.ssi_weight = ssi_weight
        self.silog_weight = silog_weight
        # Optional clamp range sourced from YAML; if None, fall back to safe defaults
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        print(f"🎯 SSI-Silog Loss initialized:")
        print(f"   SSI weight: {ssi_weight}")
        print(f"   Silog weight: {silog_weight}")
        print(f"   Alpha: {alpha}")
        print(f"   Silog ratio: {silog_ratio}")
        if (self.min_depth is not None) or (self.max_depth is not None):
            print(f"   Depth clamp (YAML): min={self.min_depth}, max={self.max_depth}")

    def set_depth_range(self, min_depth: float, max_depth: float):
        """Optionally set depth clamp range after construction."""
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def compute_ssi_loss(self, pred_inv_depth, gt_inv_depth, mask):
        """Compute SSI loss on inverse depth"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_inv_depth.device, requires_grad=True)
        
        diff = (pred_inv_depth[mask] - gt_inv_depth[mask])
        diff2 = diff ** 2
        mean = diff.mean()
        var = diff2.mean() - mean ** 2
        ssi_loss = var + self.alpha * mean ** 2
        # Expose internals for debugging/metrics
        try:
            self.add_metric('ssi_mean', mean)
            self.add_metric('ssi_var', var)
        except Exception:
            pass
        return ssi_loss

    def compute_silog_loss(self, pred_inv_depth, gt_inv_depth, mask):
        """Compute Silog loss on depth"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_inv_depth.device, requires_grad=True)

        # 1) inv -> depth 변환
        pred_depth = inv2depth(pred_inv_depth)
        gt_depth = inv2depth(gt_inv_depth)

        # 2) YAML의 min/max로 클램프 (없으면 기본값)
        clamp_min = 1e-3 if self.min_depth is None else float(self.min_depth)
        clamp_max = 100.0 if self.max_depth is None else float(self.max_depth)
        if clamp_max <= clamp_min:
            clamp_max = clamp_min + 1.0

        pred_pre = pred_depth
        gt_pre = gt_depth
        pred_depth = torch.clamp(pred_depth, min=clamp_min, max=clamp_max)
        gt_depth = torch.clamp(gt_depth, min=clamp_min, max=clamp_max)

        # 3) 마스크 적용 후 텐서 확보
        pred_pre_masked = pred_pre[mask]
        gt_pre_masked = gt_pre[mask]
        pred_depth_masked = pred_depth[mask]
        gt_depth_masked = gt_depth[mask]

        # 3.5) Silog 계산 준비 및 계산
        log_pred = torch.log(pred_depth_masked * self.silog_ratio)
        log_gt = torch.log(gt_depth_masked * self.silog_ratio)
        log_diff = log_pred - log_gt
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.silog_ratio2 * (log_diff.mean() ** 2)
        silog_var = silog1 - silog2
        if torch.any(silog_var < 0):
            silog_var = torch.abs(silog_var)
        silog_loss = torch.sqrt(silog_var + 1e-8) * self.silog_ratio
        # Expose internals for debugging/metrics
        try:
            self.add_metric('silog1', silog1)
            self.add_metric('silog2', silog2)
            self.add_metric('silog_var', silog_var)
        except Exception:
            pass

        # 4) 로깅(메트릭/옵션 콘솔)
        try:
            # 기본 범위 메트릭
            self.add_metric('silog_clamp_min', float(clamp_min))
            self.add_metric('silog_clamp_max', float(clamp_max))
            # 실제 사용된 깊이 범위 메트릭 (post-clamp)
            if pred_depth_masked.numel() > 0:
                self.add_metric('pred_depth_min', pred_depth_masked.min())
                self.add_metric('pred_depth_max', pred_depth_masked.max())
            if gt_depth_masked.numel() > 0:
                self.add_metric('gt_depth_min', gt_depth_masked.min())
                self.add_metric('gt_depth_max', gt_depth_masked.max())

            # Pre-clamp 범위/백분위수
            def _add_stats(prefix: str, t: torch.Tensor):
                if t.numel() == 0:
                    return
                t_f = t[torch.isfinite(t)]
                if t_f.numel() == 0:
                    return
                self.add_metric(f'{prefix}_min_pre', t_f.min())
                self.add_metric(f'{prefix}_max_pre', t_f.max())
                self.add_metric(f'{prefix}_mean_pre', t_f.mean())
                self.add_metric(f'{prefix}_std_pre', t_f.std())
                try:
                    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=t_f.device)
                    qv = torch.quantile(t_f, qs)
                    self.add_metric(f'{prefix}_p01_pre', qv[0])
                    self.add_metric(f'{prefix}_p05_pre', qv[1])
                    self.add_metric(f'{prefix}_p50_pre', qv[2])
                    self.add_metric(f'{prefix}_p95_pre', qv[3])
                    self.add_metric(f'{prefix}_p99_pre', qv[4])
                except Exception:
                    pass

            _add_stats('pred_depth', pred_pre_masked)
            _add_stats('gt_depth', gt_pre_masked)

            # Pre-clamp 범위 밖 비율
            m = torch.isfinite(pred_pre_masked) & torch.isfinite(gt_pre_masked)
            denom = float(m.sum().item()) if m is not None else 0.0
            if denom > 0:
                frac_pred_below = float((pred_pre_masked[m] < clamp_min).float().mean().item())
                frac_pred_above = float((pred_pre_masked[m] > clamp_max).float().mean().item())
                frac_gt_below = float((gt_pre_masked[m] < clamp_min).float().mean().item())
                frac_gt_above = float((gt_pre_masked[m] > clamp_max).float().mean().item())
            else:
                frac_pred_below = frac_pred_above = frac_gt_below = frac_gt_above = 0.0
            self.add_metric('frac_pred_depth_below_min', frac_pred_below)
            self.add_metric('frac_pred_depth_above_max', frac_pred_above)
            self.add_metric('frac_gt_depth_below_min', frac_gt_below)
            self.add_metric('frac_gt_depth_above_max', frac_gt_above)

            # Post-clamp 경계 포화 비율
            if pred_depth_masked.numel() > 0:
                self.add_metric('frac_pred_depth_at_min', float((pred_depth_masked == clamp_min).float().mean().item()))
                self.add_metric('frac_pred_depth_at_max', float((pred_depth_masked == clamp_max).float().mean().item()))
            if gt_depth_masked.numel() > 0:
                self.add_metric('frac_gt_depth_at_min', float((gt_depth_masked == clamp_min).float().mean().item()))
                self.add_metric('frac_gt_depth_at_max', float((gt_depth_masked == clamp_max).float().mean().item()))

            verbose = os.environ.get('SSI_SILOG_VERBOSE', '0') == '1'
            if os.environ.get('SSI_SILOG_LOG_EVERY', '0') == '1' or os.environ.get('SSI_SILOG_LOG_ONCE', '0') == '1':
                if os.environ.get('SSI_SILOG_LOG_ONCE', '0') == '1':
                    os.environ['SSI_SILOG_LOG_ONCE'] = '0'
                if not verbose:
                    print(
                        f"[SSI-SILOG] clamp=[{clamp_min:.4g}, {clamp_max:.4g}] "
                        f"pred_below={frac_pred_below:.4f} pred_above={frac_pred_above:.4f} "
                        f"gt_below={frac_gt_below:.4f} gt_above={frac_gt_above:.4f}"
                    )
                else:
                    def _fmt_stats(prefix: str, pre: torch.Tensor, post: torch.Tensor):
                        if pre.numel() == 0:
                            return f"{prefix}: n=0"
                        pre_f = pre[torch.isfinite(pre)]
                        post_f = post[torch.isfinite(post)] if post is not None else pre_f
                        try:
                            qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=pre_f.device)
                            qv = torch.quantile(pre_f, qs)
                            pstr = f"p01={qv[0]:.4g} p05={qv[1]:.4g} med={qv[2]:.4g} p95={qv[3]:.4g} p99={qv[4]:.4g}"
                        except Exception:
                            pstr = ""
                        return (
                            f"{prefix}: pre[min={pre_f.min():.4g} max={pre_f.max():.4g} mean={pre_f.mean():.4g} std={pre_f.std():.4g} {pstr}] "
                            f"post[min={post_f.min():.4g} max={post_f.max():.4g}]"
                        )
                    pred_at_min = float((pred_depth_masked == clamp_min).float().mean().item()) if pred_depth_masked.numel() else 0.0
                    pred_at_max = float((pred_depth_masked == clamp_max).float().mean().item()) if pred_depth_masked.numel() else 0.0
                    gt_at_min = float((gt_depth_masked == clamp_min).float().mean().item()) if gt_depth_masked.numel() else 0.0
                    gt_at_max = float((gt_depth_masked == clamp_max).float().mean().item()) if gt_depth_masked.numel() else 0.0

                    print("[SSI-SILOG]\n"
                          f"  clamp=[{clamp_min:.4g}, {clamp_max:.4g}]\n"
                          f"  pre-out-of-range: pred<={frac_pred_below:.4f}, pred>={frac_pred_above:.4f}, gt<={frac_gt_below:.4f}, gt>={frac_gt_above:.4f}\n"
                          f"  post-at-boundary: pred@min={pred_at_min:.4f}, pred@max={pred_at_max:.4f}, gt@min={gt_at_min:.4f}, gt@max={gt_at_max:.4f}\n"
                          f"  {_fmt_stats('pred_depth', pred_pre_masked, pred_depth_masked)}\n"
                          f"  {_fmt_stats('gt_depth', gt_pre_masked, gt_depth_masked)}"
                          )
        except Exception:
            pass

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