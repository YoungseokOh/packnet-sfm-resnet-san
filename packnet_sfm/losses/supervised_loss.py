# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import inspect
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.utils.depth import inv2depth

from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling
# from packnet_sfm.losses.scale_adaptive_loss import ScaleAdaptiveLoss  # ❌ Module missing
from packnet_sfm.losses.ssi_loss import SSILoss
from packnet_sfm.losses.ssi_trim_loss import SSITrimLoss
from packnet_sfm.losses.ssi_loss_enhanced import EnhancedSSILoss, ProgressiveEnhancedSSILoss
from packnet_sfm.losses.ssi_silog_loss import SSISilogLoss
# from packnet_sfm.losses.ssi_silog_nearfield_loss import SSISilogNearFieldLoss, SSISilogNearFieldLossWrapper  # ❌ Module missing
# from packnet_sfm.losses.ssi_silog_road_aware_loss import SSISilogRoadAwareLoss, SSISilogRoadAwareLossWrapper  # ❌ Module missing

########################################################################################################################


class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.

        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold
        
    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.

        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted inverse depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth map

        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = (pred - gt).abs()

        # Remove
        # mask = (gt > 0).detach()
        # diff = gt - pred
        # diff = diff[mask]
        # diff = diff.abs()

        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        return torch.cat((diff, diff2)).mean()

class SilogLoss(nn.Module):
    def __init__(self, ratio=10, ratio2=0.85):
        super().__init__()
        self.ratio = ratio  # Not used in corrected formula
        self.ratio2 = ratio2

    def forward(self, pred, gt):
        # ✅ CRITICAL FIX: Correct Silog formula from original paper
        # sqrt(E[log_diff^2] - lambda * E[log_diff]^2)
        log_diff = torch.log(pred) - torch.log(gt)
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.ratio2 * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2)  # ✅ No multiplication by ratio!
        return silog_loss

########################################################################################################################

def get_loss_func(supervised_method, **kwargs):
    """Determines the supervised loss to be used, given the supervised method."""
    print(f"🔍 Loading loss function for: {supervised_method}")
    
    if supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('ssi-silog'):
        # 🆕 클래스 기반 SSI-Silog 손실 (YAML에서 파라미터 전달 가능)
        return SSISilogLoss(
            min_depth=kwargs.get('min_depth', None),
            max_depth=kwargs.get('max_depth', None),
            ssi_weight=kwargs.get('ssi_weight', 0.7),
            silog_weight=kwargs.get('silog_weight', 0.3),
            alpha=kwargs.get('alpha', 0.85),
            silog_ratio=kwargs.get('silog_ratio', 10),
            silog_ratio2=kwargs.get('silog_ratio2', 0.85),
            gradient_weight=kwargs.get('gradient_weight', 0.0),  # 🆕 Gradient Loss
            gradient_scales=kwargs.get('gradient_scales', 4),     # 🆕 Gradient scales
        )
    # elif supervised_method.endswith('ssi-silog-nearfield'):
    #     # 🆕 SSI-Silog with Optional Near-Field Weighting
    #     # Can enable/disable near-field weighting via enable_near_field_weighting parameter
    #     # ❌ NOT IMPLEMENTED - module missing
    #     from packnet_sfm.losses.ssi_silog_nearfield_loss import SSISilogNearFieldLossWrapper
    #     return SSISilogNearFieldLossWrapper(
    #         enable_near_field_weighting=kwargs.get('enable_near_field_weighting', True),
    #         alpha=kwargs.get('alpha', 10.0),
    #         silog_ratio=kwargs.get('silog_ratio', 0.85),
    #         silog_ratio2=kwargs.get('silog_ratio2', 0.0),
    #         ssi_weight=kwargs.get('ssi_weight', 0.7),
    #         silog_weight=kwargs.get('silog_weight', 0.3),
    #         min_depth=kwargs.get('min_depth', None),
    #         max_depth=kwargs.get('max_depth', None),
    #     )
    # elif supervised_method.endswith('ssi-silog-road-aware'):
    #     # 🛣️ SSI-Silog with Road-Aware and Near-Field Weighting
    #     # Prioritizes road pixels + near-field distance for autonomous driving
    #     # ❌ NOT IMPLEMENTED - module missing
    #     from packnet_sfm.losses.ssi_silog_road_aware_loss import SSISilogRoadAwareLossWrapper
    #     return SSISilogRoadAwareLossWrapper(
    #         enable_road_weighting=kwargs.get('enable_road_weighting', True),
    #         alpha=kwargs.get('alpha', 10.0),
    #         silog_ratio=kwargs.get('silog_ratio', 0.85),
    #         silog_ratio2=kwargs.get('silog_ratio2', 0.0),
    #         ssi_weight=kwargs.get('ssi_weight', 0.7),
    #         silog_weight=kwargs.get('silog_weight', 0.3),
    #         min_depth=kwargs.get('min_depth', None),
    #         max_depth=kwargs.get('max_depth', None),
    #         near_field_threshold=kwargs.get('near_field_threshold', 1.0),
    #         road_weight=kwargs.get('road_weight', 5.0),
    #         road_nearfield_weight=kwargs.get('road_nearfield_weight', 10.0),
    #         nonroad_nearfield_weight=kwargs.get('nonroad_nearfield_weight', 3.0),
    #     )
    elif supervised_method.endswith('silog'):
        return SilogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    elif supervised_method.endswith('ssi'):
        return SSILoss()
    elif supervised_method.endswith('enhanced-ssi'):
        return EnhancedSSILoss()
    elif supervised_method.endswith('progressive-ssi'):
        return ProgressiveEnhancedSSILoss()
    elif supervised_method.endswith('ssi-trim'):
        return SSITrimLoss(trim=0.2, epsilon=1e-6)
    # elif supervised_method.endswith('scale-adaptive'):
    #     # ❌ NOT IMPLEMENTED - module missing
    #     return ScaleAdaptiveLoss(
    #         lambda_sg=kwargs.get('lambda_sg', 0.5),
    #         epsilon=kwargs.get('epsilon', 1e-8),
    #         num_scales=kwargs.get('num_scales', 4),
    #         use_absolute=kwargs.get('use_absolute', True),
    #         use_inv_depth=kwargs.get('use_inv_depth', False),
    #         min_depth=kwargs.get('min_depth', None),
    #         max_depth=kwargs.get('max_depth', None),
    #     )
    # elif supervised_method.endswith('adaptive-multi-domain'):
    #     # 🆕 Adaptive Multi-Domain Loss with uncertainty-based weighting
    #     # Combines ScaleAdaptiveLoss (structure) + SSISilogLoss (scale)
    #     # Import here to avoid circular dependency
    #     # ❌ NOT IMPLEMENTED - module missing
    #     from packnet_sfm.losses.adaptive_multi_domain_loss import AdaptiveMultiDomainLossWrapper
    #     return AdaptiveMultiDomainLossWrapper(
    #         # ScaleAdaptiveLoss parameters
    #         lambda_sg=kwargs.get('lambda_sg', 0.1),
    #         num_scales=kwargs.get('num_scales', 4),
    #         # SSISilogLoss parameters
    #         ssi_weight=kwargs.get('ssi_weight', 0.7),
    #         silog_weight=kwargs.get('silog_weight', 0.3),
    #         alpha_ssi=kwargs.get('alpha_ssi', 0.85),
    #         beta_silog=kwargs.get('beta_silog', 0.15),
    #         # Common parameters
    #         min_depth=kwargs.get('min_depth', 0.05),
    #         max_depth=kwargs.get('max_depth', 100.0),
    #     )
    # elif supervised_method.endswith('fixed-multi-domain'):
    #     # 🆕 Fixed-Weight Multi-Domain Loss (STABLE BASELINE)
    #     # Combines ScaleAdaptiveLoss (structure) + SSISilogLoss (scale)
    #     # with fixed, manually tuned weights (no uncertainty learning)
    #     # Import here to avoid circular dependency
    #     # ❌ NOT IMPLEMENTED - module missing
    #     from packnet_sfm.losses.fixed_multi_domain_loss import FixedMultiDomainLossWrapper
    #     return FixedMultiDomainLossWrapper(
    #         # Fixed weights (can be tuned via grid search)
    #         w_structure=kwargs.get('w_structure', 0.4),
    #         w_scale=kwargs.get('w_scale', 0.6),
    #         # ScaleAdaptiveLoss parameters
    #         lambda_sg=kwargs.get('lambda_sg', 0.1),
    #         num_scales=kwargs.get('num_scales', 4),
    #         use_absolute=kwargs.get('use_absolute', True),
    #         use_inv_depth=kwargs.get('use_inv_depth', False),
    #         # SSISilogLoss parameters
    #         ssi_weight=kwargs.get('ssi_weight', 0.7),
    #         silog_weight=kwargs.get('silog_weight', 0.3),
    #         alpha_ssi=kwargs.get('alpha_ssi', 0.85),
    #         beta_silog=kwargs.get('beta_silog', 0.15),
    #         # Common parameters
    #         min_depth=kwargs.get('min_depth', 0.05),
    #         max_depth=kwargs.get('max_depth', 100.0),
    #     )
    # elif supervised_method.endswith('distance-aware-multi-domain'):
    #     # 🆕 Distance-Aware Multi-Domain Loss (PHASE 2: NEAR-FIELD FOCUSED)
    #     # Applies distance-based weighting on top of fixed multi-domain loss
    #     # CRITICAL: 5x weight for D < 1m (safety-critical range)
    #     # Expected: D<1m abs_rel 0.028 → 0.018 (36% improvement)
    #     # ❌ NOT IMPLEMENTED - module missing
    #     from packnet_sfm.losses.distance_aware_loss import DistanceAwareMultiDomainLossWrapper
    #     return DistanceAwareMultiDomainLossWrapper(
    #         # Fixed domain weights (from Phase 1 best result: w40_60)
    #         w_structure=kwargs.get('w_structure', 0.4),
    #         w_scale=kwargs.get('w_scale', 0.6),
    #         # Distance-based weights (optional custom ranges)
    #         weight_ranges=kwargs.get('weight_ranges', None),  # None = use default
    #         # ScaleAdaptiveLoss parameters
    #         lambda_sg=kwargs.get('lambda_sg', 0.1),
    #         num_scales=kwargs.get('num_scales', 4),
    #         use_absolute=kwargs.get('use_absolute', True),
    #         use_inv_depth=kwargs.get('use_inv_depth', False),
    #         # SSISilogLoss parameters
    #         ssi_weight=kwargs.get('ssi_weight', 0.7),
    #         silog_weight=kwargs.get('silog_weight', 0.3),
    #         alpha_ssi=kwargs.get('alpha_ssi', 0.85),
    #         beta_silog=kwargs.get('beta_silog', 0.15),
    #         # Common parameters
    #         min_depth=kwargs.get('min_depth', 0.05),
    #         max_depth=kwargs.get('max_depth', 100.0),
    #     )
    
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))

########################################################################################################################

class SupervisedLoss(LossBase):
    """
    Supervised loss for inverse depth maps.

    Parameters
    ----------
    supervised_method : str
        Which supervised method will be used
    supervised_num_scales : int
        Number of scales used by the supervised loss
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_method='sparse-l1',
                 supervised_num_scales=4, progressive_scaling=0.0, **kwargs):
        super().__init__()
        # 보존: get_loss_func에 kwargs로 min_depth/max_depth 등을 넘겨 SSI-Silog에서 사용할 수 있게 함
        self.loss_func = get_loss_func(supervised_method, **kwargs)
        self.supervised_method = supervised_method
        self.n = supervised_num_scales
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)

        # Instantiate-once logging for selected loss function
        if not hasattr(SupervisedLoss, '_logged_methods'):
            SupervisedLoss._logged_methods = set()
        if supervised_method not in SupervisedLoss._logged_methods:
            loss_cls = getattr(self.loss_func, '__class__', type(self.loss_func))
            params_of_interest = ['lambda_sg', 'num_scales', 'use_absolute',
                                  'use_inv_depth', 'epsilon', 'min_depth', 'max_depth']
            captured = {k: kwargs[k] for k in params_of_interest if k in kwargs}
            msg = f"[SupervisedLoss] Using '{supervised_method}' -> {loss_cls.__name__}"
            if captured:
                pretty = ', '.join(f"{k}={v}" for k, v in captured.items())
                msg += f" ({pretty})"
            print(msg)
            SupervisedLoss._logged_methods.add(supervised_method)

    ########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'supervised_num_scales': self.n,
        }

########################################################################################################################

    def calculate_loss(self, inv_depths, gt_inv_depths, masks=None):
        """
        Calculate the supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps
        masks : list of torch.Tensor, optional
            List of binary masks to apply. Defaults to None.

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """
        num_scales = self.n
        
        if self.supervised_method.startswith('sparse'):
            total_loss = 0.0
            eps = 1e-6

            for i in range(num_scales):
                # 기존 마스크: Ground Truth가 0보다 큰 픽셀
                valid_mask = (gt_inv_depths[i] > 0.).detach()
                
                # 추가 Binary mask 로드 및 결합
                current_mask = valid_mask
                if masks is not None and i < len(masks) and masks[i] is not None:
                    # Ensure mask dimensions match for broadcasting
                    if masks[i].shape != valid_mask.shape:
                        # Resize mask if necessary (assuming masks[i] is already a tensor)
                        masks[i] = torch.nn.functional.interpolate(
                            masks[i].float().unsqueeze(0).unsqueeze(0), # Add batch and channel dims
                            size=valid_mask.shape[2:], # Target H, W
                            mode='nearest'
                        ).squeeze(0).squeeze(0) # Remove batch and channel dims
                        masks[i] = (masks[i] > 0).float() # Ensure it's a binary mask (0 or 1)

                    current_mask = valid_mask & masks[i].to(valid_mask.device) # Combine masks

                pred_filled = inv_depths[i].masked_fill(~current_mask, eps)
                gt_filled = gt_inv_depths[i].masked_fill(~current_mask, eps)
                
                # Build kwargs based on the loss forward signature
                loss_kwargs = {}
                if hasattr(self.loss_func, 'forward'):
                    sig = inspect.signature(self.loss_func.forward)
                    params = sig.parameters
                    if 'mask' in params:
                        loss_kwargs['mask'] = current_mask
                    if 'progress' in params:
                        loss_kwargs['progress'] = getattr(self, '_progress', 0.0)
                    if 'epoch' in params:
                        loss_kwargs['epoch'] = getattr(self, '_epoch', 0)
                # Call with filtered kwargs only
                loss_i = self.loss_func(pred_filled, gt_filled, **loss_kwargs)

                # ===== Per-scale logging for analysis (especially sparse-ssi-silog) =====
                try:
                    # Per-scale loss value
                    self.add_metric(f's{i}/loss', loss_i)
                    # Valid pixel stats
                    valid_px = int(current_mask.sum().item())
                    total_px = int(current_mask.numel())
                    self.add_metric(f's{i}/valid_px', valid_px)
                    self.add_metric(f's{i}/valid_ratio', valid_px / max(1, total_px))
                    # If underlying loss exposes metrics (e.g., SSISilogLoss), copy them with scale prefix
                    if isinstance(self.loss_func, LossBase):
                        for k, v in self.loss_func.metrics.items():
                            self.add_metric(f's{i}/{k}', v)

                    # Optional concise per-step console debug for scale 0
                    import os as _os
                    if i == 0 and (_os.environ.get('SSI_SILOG_LOG_EVERY', '0') == '1' or _os.environ.get('SSI_SILOG_LOG_ONCE', '0') == '1'):
                        if _os.environ.get('SSI_SILOG_LOG_ONCE', '0') == '1':
                            _os.environ['SSI_SILOG_LOG_ONCE'] = '0'
                        # Pull a few key metrics if present
                        m = self.loss_func.metrics if isinstance(self.loss_func, LossBase) else {}
                        ssi_comp = m.get('ssi_component', None)
                        silog_comp = m.get('silog_component', None)
                        ssi_mean = m.get('ssi_mean', None)
                        ssi_var = m.get('ssi_var', None)
                        silog1 = m.get('silog1', None)
                        silog2 = m.get('silog2', None)
                        silog_var = m.get('silog_var', None)
                        frac_pb = m.get('frac_pred_depth_below_min', None)
                        frac_pa = m.get('frac_pred_depth_above_max', None)
                        frac_gb = m.get('frac_gt_depth_below_min', None)
                        frac_ga = m.get('frac_gt_depth_above_max', None)
                        pred_min = m.get('pred_depth_min', None)
                        pred_max = m.get('pred_depth_max', None)
                        gt_min = m.get('gt_depth_min', None)
                        gt_max = m.get('gt_depth_max', None)
                        # Format helper
                        def _f(x):
                            try:
                                return float(x.detach().item() if hasattr(x, 'detach') else x)
                            except Exception:
                                return None
                        print("[SupervisedLoss s0]",
                              f"loss={_f(loss_i):.6f}",
                              f"ssi={_f(ssi_comp):.6f}" if ssi_comp is not None else "ssi=NA",
                              f"silog={_f(silog_comp):.6f}" if silog_comp is not None else "silog=NA",
                              f"ssi_mean={_f(ssi_mean):.4e}" if ssi_mean is not None else "",
                              f"ssi_var={_f(ssi_var):.4e}" if ssi_var is not None else "",
                              f"silog1={_f(silog1):.4e}" if silog1 is not None else "",
                              f"silog2={_f(silog2):.4e}" if silog2 is not None else "",
                              f"silog_var={_f(silog_var):.4e}" if silog_var is not None else "",
                              f"pred[min={_f(pred_min):.4g} max={_f(pred_max):.4g}]" if pred_min is not None and pred_max is not None else "",
                              f"gt[min={_f(gt_min):.4g} max={_f(gt_max):.4g}]" if gt_min is not None and gt_max is not None else "",
                              f"out-of-range pred<={_f(frac_pb):.4f} pred>={_f(frac_pa):.4f} gt<={_f(frac_gb):.4f} gt>={_f(frac_ga):.4f}" if frac_pb is not None else "",
                              )
                except Exception:
                    pass

                total_loss += loss_i

            return total_loss / float(num_scales)
        else:
            # Dense loss handling remains the same
            return sum([
                self.loss_func(inv_depths[i], gt_inv_depths[i])
                for i in range(num_scales)
            ]) / float(num_scales)
    
    def forward(self, inv_depths, gt_inv_depth, return_logs=False, progress=0.0, masks=None):
        """Forward with progress information for enhanced losses"""
        # Store progress for enhanced losses
        self._progress = progress
        
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)
        # Match predicted scales for ground-truth
        gt_inv_depths = match_scales(gt_inv_depth, inv_depths, self.n,
                                     mode='nearest', align_corners=None)
        
        # Calculate and store supervised loss, passing masks
        loss = self.calculate_loss(inv_depths, gt_inv_depths, masks=masks)
        self.add_metric('supervised_loss', loss)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }