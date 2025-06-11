# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling


class SSILoss(LossBase):
    """
    Scale-Shift-Invariant depth loss (log-depth version).
    """
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred_inv_depth, gt_inv_depth, mask=None):
        if mask is None: mask = (gt_inv_depth > 0)
        diff = (pred_inv_depth[mask] - gt_inv_depth[mask])
        diff2  = diff ** 2
        n = diff.numel()
        mean = diff.mean()
        var  = diff2.mean() - mean ** 2
        loss = var + self.alpha * mean ** 2
        return loss
