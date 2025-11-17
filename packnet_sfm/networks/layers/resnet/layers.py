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

def dual_head_to_depth(integer_normalized, fractional_sigmoid, max_depth, n_integer_levels=256):
    """
    Convert dual-head outputs to depth
    
    🆕 PTQ Quantization: Integer is now discretized to 256 levels
    
    Parameters
    ----------
    integer_normalized : torch.Tensor [B, 1, H, W]
        Integer part normalized to [0, 1] representing quantization level
        (e.g., 0.0 = level 0, 0.5 = level 128, 1.0 = level 255)
    fractional_sigmoid : torch.Tensor [B, 1, H, W]
        Fractional part in sigmoid space [0, 1]m
    max_depth : float
        Maximum depth for integer scaling
    n_integer_levels : int
        Number of quantization levels (default: 256 for 8-bit PTQ)
    
    Returns
    -------
    depth : torch.Tensor [B, 1, H, W]
        Final depth in meters [0, max_depth + 1]
    
    Example
    -------
    >>> integer_norm = torch.tensor([[[[0.333]]]])  # Normalized level
    >>> fractional_sig = torch.tensor([[[[0.5]]]])  # 0.5m
    >>> depth = dual_head_to_depth(integer_norm, fractional_sig, 15.0, n_integer_levels=256)
    """
    # Integer part: normalize to actual meters
    # integer_meters = normalized * max_depth
    integer_part = integer_normalized * max_depth
    
    # Fractional part: already [0, 1]m
    fractional_part = fractional_sigmoid
    
    # Combine: final depth = integer_part + fractional_part
    depth = integer_part + fractional_part
    
    return depth


def decompose_depth(depth_gt, max_depth, n_integer_levels=256):
    """
    Decompose ground truth depth into integer and fractional parts
    
    🆕 PTQ Quantization: Integer part is quantized to 256 discrete levels
    
    Parameters
    ----------
    depth_gt : torch.Tensor [B, 1, H, W]
        Ground truth depth in meters
    max_depth : float
        Maximum depth for integer normalization
    n_integer_levels : int
        Number of quantization levels for integer head (default: 256 for 8-bit PTQ)
    
    Returns
    -------
    integer_gt : torch.Tensor [B, 1, H, W]
        Integer part normalized to [0, 1] for n_integer_levels
        (e.g., level 0 = 0m, level 128 = max_depth*128/256, level 255 = max_depth)
    fractional_gt : torch.Tensor [B, 1, H, W]
        Fractional part [0, 1]m
    
    Example
    -------
    >>> depth = torch.tensor([[[[5.7]]]])  # 5.7m, max_depth=15.0
    >>> integer_gt, frac_gt = decompose_depth(depth, 15.0, n_integer_levels=256)
    >>> # Integer level ≈ (5.0 / 15.0) * 255 ≈ 85 level
    >>> # Fractional = 0.7m
    """
    import torch
    
    # Integer part: Quantize to n_integer_levels discrete levels
    # depth_in_levels = (depth_gt / max_depth) * (n_integer_levels - 1)
    integer_levels = torch.round((depth_gt / max_depth) * (n_integer_levels - 1))
    integer_levels = torch.clamp(integer_levels, min=0, max=n_integer_levels - 1)
    
    # Normalize integer levels to [0, 1]
    integer_gt = integer_levels / (n_integer_levels - 1)  # [B, 1, H, W]
    
    # Fractional part: depth - (integer_level * max_depth / (n_integer_levels - 1))
    integer_meters = (integer_levels / (n_integer_levels - 1)) * max_depth
    fractional_gt = depth_gt - integer_meters  # [B, 1, H, W] in [0, 1]m range
    
    # Clamp fractional part to valid range [0, 1]
    fractional_gt = torch.clamp(fractional_gt, min=0, max=1.0)
    
    return integer_gt, fractional_gt


def dual_head_to_inv_depth(integer_normalized, fractional_sigmoid, max_depth, min_depth=0.5, n_integer_levels=256):
    """
    Convert dual-head outputs to inverse depth (for compatibility with existing code)
    
    Parameters
    ----------
    integer_normalized : torch.Tensor
        Integer part normalized [0, 1]
    fractional_sigmoid : torch.Tensor
        Fractional part sigmoid output
    max_depth : float
        Maximum depth
    min_depth : float
        Minimum depth
    n_integer_levels : int
        Number of quantization levels (default: 256)
    
    Returns
    -------
    inv_depth : torch.Tensor
        Inverse depth [1/max_depth, 1/min_depth]
    """
    # First convert to depth
    depth = dual_head_to_depth(integer_normalized, fractional_sigmoid, max_depth, n_integer_levels)
    
    # Clamp to valid range
    import torch
    depth = torch.clamp(depth, min=min_depth, max=max_depth + 1.0)
    
    # Convert to inverse depth
    inv_depth = 1.0 / depth
    
    return inv_depth
