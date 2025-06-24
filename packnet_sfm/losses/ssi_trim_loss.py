import torch
import torch.nn as nn

class SSITrimLoss(nn.Module):
    """
    Scale- and Shift-Invariant Trimmed L1-Loss (MiDaS `L_ssitrim`)

    Args
    ----
    trim : float, optional
        Fraction (0‒1) of highest-error pixels to discard. 0 → no trimming.
    epsilon : float, optional
        Small number to stabilise pseudo-inverse when var(pred) ≈ 0.
    """
    def __init__(self, trim: float = 0.2, epsilon: float = 1e-6):
        super().__init__()
        assert 0.0 <= trim < 1.0, "trim must be in [0,1)"
        self.trim = trim
        self.eps = epsilon

    @staticmethod
    def _solve_scale_shift(pred, gt, mask, eps):
        """Closed-form α, β solving ‖α d + β – z‖² (least-squares)."""
        # flatten valid pixels
        d = pred[mask]                # (N,)
        z = gt[mask]
        if d.numel() == 0:            # empty mask → return neutral α,β
            return torch.tensor(1.0, device=pred.device), torch.tensor(0.0, device=pred.device)
        
        # 🆕 더 안정적인 스케일-시프트 계산
        n = d.numel()
        if n < 10:  # 너무 적은 픽셀이면 neutral 반환
            return torch.tensor(1.0, device=pred.device), torch.tensor(0.0, device=pred.device)
        
        # 🆕 Robust statistics로 outlier에 덜 민감하게
        mean_d, mean_z = d.mean(), z.mean()
        
        # 분산 계산을 더 안정적으로
        var_d = torch.var(d, unbiased=False) + eps
        cov_dz = torch.mean((d - mean_d) * (z - mean_z))
        
        # 🆕 더 안정적인 alpha 계산
        alpha = cov_dz / var_d
        beta = mean_z - alpha * mean_d
        
        # 🆕 비정상적인 스케일 방지
        alpha = torch.clamp(alpha, 0.1, 10.0)  # 스케일 제한
        
        return alpha, beta

    def forward(self, pred, gt, mask=None):
        """
        pred, gt : (B, 1, H, W) -or- (B, H, W)
        mask     : bool/int same shape, True/1 = valid GT.  None → all valid.
        """
        if pred.dim() == 4:     # squeeze channel dim if present
            pred, gt = pred.squeeze(1), gt.squeeze(1)
            if mask is not None and mask.dim() == 4:
                mask = mask.squeeze(1)

        B = pred.shape[0]
        total = 0.0
        
        for b in range(B):
            mb = torch.ones_like(gt[b], dtype=torch.bool) if mask is None else mask[b] > 0
            
            # 🆕 유효한 픽셀이 충분한지 확인
            if mb.sum() < 100:  # 최소 100개 픽셀 필요
                total += torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                continue
            
            α, β = self._solve_scale_shift(pred[b], gt[b], mb, self.eps)
            
            # 🆕 aligned prediction 계산
            aligned_pred = α * pred[b] + β
            
            # aligned residuals
            res = torch.abs(aligned_pred - gt[b])
            res = res[mb]                     # apply mask

            if self.trim > 0 and res.numel() > 0:
                k = int((1.0 - self.trim) * res.numel())
                if k > 0:
                    # 정렬을 사용해서 상위 trim% 제거
                    res_sorted, _ = torch.sort(res)
                    res = res_sorted[:k]  # 작은 k개만 선택 (상위 trim% 제거)
                elif k == 0:
                    res = torch.tensor([], device=res.device, dtype=res.dtype)
            
            # 빈 텐서 처리
            if res.numel() > 0:
                total += res.mean()
            else:
                total += torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        return total / B