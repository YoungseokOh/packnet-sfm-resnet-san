# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/layers.py

from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        # ReLU
        self.nonlin = nn.ReLU(inplace=True)
        # ELU
        # self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=False):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
            self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        else:
            # 🔧 ONNX 변환 최적화: Conv2d 내장 패딩 사용 (별도 Pad 레이어 불필요)
            self.pad = None
            self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, padding=1)

    def forward(self, x):
        if self.pad is not None:
            out = self.pad(x)
            out = self.conv(out)
        else:
            # Conv2d 내장 패딩 사용
            out = self.conv(x)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


# ==============================================================================
# Dual-Head Helper Functions (ST2 Implementation)
# ==============================================================================

def dual_head_to_depth(integer_sigmoid, fractional_sigmoid, max_depth):
    """
    Convert dual-head sigmoid outputs to depth
    
    Parameters
    ----------
    integer_sigmoid : torch.Tensor [B, 1, H, W]
        Integer part in sigmoid space [0, 1]
    fractional_sigmoid : torch.Tensor [B, 1, H, W]
        Fractional part in sigmoid space [0, 1]
    max_depth : float
        Maximum depth for integer scaling
    
    Returns
    -------
    depth : torch.Tensor [B, 1, H, W]
        Final depth in meters [0, max_depth + 1]
    
    Example
    -------
    >>> integer_sig = torch.tensor([[[[0.333]]]])  # 0.333 * 15 = 5.0
    >>> fractional_sig = torch.tensor([[[[0.5]]]])  # 0.5m
    >>> depth = dual_head_to_depth(integer_sig, fractional_sig, 15.0)
    >>> print(depth)  # 5.5m
    """
    # Integer part: [0, 1] → [0, max_depth]
    integer_part = integer_sigmoid * max_depth
    
    # Fractional part: already [0, 1]m
    fractional_part = fractional_sigmoid
    
    # Combine
    depth = integer_part + fractional_part
    
    return depth


def decompose_depth(depth_gt, max_depth):
    """
    Decompose ground truth depth into integer and fractional parts
    
    Parameters
    ----------
    depth_gt : torch.Tensor [B, 1, H, W]
        Ground truth depth in meters
    max_depth : float
        Maximum depth for integer normalization
    
    Returns
    -------
    integer_gt : torch.Tensor [B, 1, H, W]
        Integer part in sigmoid space [0, 1]
    fractional_gt : torch.Tensor [B, 1, H, W]
        Fractional part [0, 1]m
    
    Example
    -------
    >>> depth = torch.tensor([[[[5.7]]]])  # 5.7m
    >>> integer_gt, frac_gt = decompose_depth(depth, 15.0)
    >>> print(integer_gt)  # 5.0 / 15.0 = 0.333
    >>> print(frac_gt)     # 0.7m
    """
    import torch
    
    # Integer part: floor(depth)
    integer_meters = torch.floor(depth_gt)
    integer_gt = integer_meters / max_depth  # Normalize to [0, 1]
    
    # Fractional part: depth - floor(depth)
    fractional_gt = depth_gt - integer_meters  # Already [0, 1]m
    
    return integer_gt, fractional_gt


def dual_head_to_inv_depth(integer_sigmoid, fractional_sigmoid, max_depth, min_depth=0.5):
    """
    Convert dual-head outputs to inverse depth (for compatibility with existing code)
    
    Parameters
    ----------
    integer_sigmoid : torch.Tensor
        Integer part sigmoid output
    fractional_sigmoid : torch.Tensor
        Fractional part sigmoid output
    max_depth : float
        Maximum depth
    min_depth : float
        Minimum depth
    
    Returns
    -------
    inv_depth : torch.Tensor
        Inverse depth [1/max_depth, 1/min_depth]
    """
    # First convert to depth
    depth = dual_head_to_depth(integer_sigmoid, fractional_sigmoid, max_depth)
    
    # Clamp to valid range
    import torch
    depth = torch.clamp(depth, min=min_depth, max=max_depth + 1.0)
    
    # Convert to inverse depth
    inv_depth = 1.0 / depth
    
    return inv_depth
