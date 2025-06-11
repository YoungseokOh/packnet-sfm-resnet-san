# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling
from packnet_sfm.losses.ssi_loss import SSILoss

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
        self.ratio = ratio
        self.ratio2 = ratio2

    def forward(self, pred, gt):
        log_diff = torch.log(pred * self.ratio) - \
                   torch.log(gt * self.ratio)
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.ratio2 * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * self.ratio
        return silog_loss

########################################################################################################################

def get_loss_func(supervised_method):
    """Determines the supervised loss to be used, given the supervised method."""
    if supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('silog'):
        return SilogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    elif supervised_method.endswith('ssi'):
        return SSILoss()
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
        self.loss_func = get_loss_func(supervised_method)
        self.supervised_method = supervised_method
        self.n = supervised_num_scales
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)

    ########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'supervised_num_scales': self.n,
        }

########################################################################################################################

    def calculate_loss(self, inv_depths, gt_inv_depths):
        """
        Calculate the supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """
        num_scales = self.n
        if self.supervised_method.startswith('sparse'):
            total_loss = 0.0

            # epsilon: 로그에서 0을 피하기 위해 사용하는 작은 값
            # dtype에 맞게 설정 (예: float32라면 1e-6, float16이면 좀 더 크게)
            eps = 1e-6

            for i in range(num_scales):
                mask = (gt_inv_depths[i] > 0.).detach()  # Bool mask, shape [B,1,H,W]

                # mask=False인 위치를 epsilon으로 채워 줍니다.
                #   - inv_depths[i]도 0인 부분이 있으면 로그 NaN이므로, 동일하게 epsilon으로 처리
                pred_filled = inv_depths[i].masked_fill(~mask, eps)
                gt_filled   = gt_inv_depths[i].masked_fill(~mask, eps)

                # 이제 pred_filled, gt_filled는 “모두 양수(eps 이상)”이므로 로그 손실에서 NaN이 나오지 않습니다.
                loss_i = self.loss_func(pred_filled, gt_filled)
                total_loss += loss_i

            return total_loss / float(num_scales)
        # Dense 계열 손실(e.g., mse, berhu, silog, abs_rel)인 경우는 기존 로직 그대로
        else:
            return sum([
                self.loss_func(inv_depths[i], gt_inv_depths[i])
                for i in range(num_scales)
            ]) / float(num_scales)

    def forward(self, inv_depths, gt_inv_depth,
                return_logs=False, progress=0.0):
        """
        Calculates training supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)
        # Match predicted scales for ground-truth
        gt_inv_depths = match_scales(gt_inv_depth, inv_depths, self.n,
                                     mode='nearest', align_corners=None)
        # Calculate and store supervised loss
        loss = self.calculate_loss(inv_depths, gt_inv_depths)
        self.add_metric('supervised_loss', loss)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }