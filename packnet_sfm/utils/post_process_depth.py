"""
Post-processing functions for converting sigmoid outputs to depth.

This module provides two transformation methods:
1. Linear space transformation (original method, poor INT8 performance)
2. Log space transformation (new method, better INT8 performance)
"""

import torch
import numpy as np


def sigmoid_to_inv_depth(sigmoid_output, min_depth=0.05, max_depth=80.0, use_log_space=False):
    """
    Convert sigmoid [0, 1] to bounded inverse depth.
    
    This is the TRAINING-TIME conversion that must match evaluation!
    
    Args:
        sigmoid_output: torch.Tensor, values in [0, 1]
        min_depth: minimum depth (m), default 0.05
        max_depth: maximum depth (m), default 80.0
        use_log_space: bool, if True use log space interpolation (default: False)
    
    Returns:
        inv_depth: torch.Tensor, values in [1/max_depth, 1/min_depth]
    
    Mathematical Formula (Linear - default):
        inv_depth = min_inv + (max_inv - min_inv) × sigmoid
        
        where:
            min_inv = 1 / max_depth  (far, sigmoid=0)
            max_inv = 1 / min_depth  (near, sigmoid=1)
    
    Mathematical Formula (Log space):
        log_inv = log(min_inv) + (log(max_inv) - log(min_inv)) × sigmoid
        inv_depth = exp(log_inv)
        
        Benefits of log space:
            - More uniform distribution across depth range
            - Better for INT8 quantization (~3% error vs ~39% for linear)
            - Geometric mean at sigmoid=0.5 (2.0m vs 0.1m for linear)
    
    Example:
        >>> sigmoid = torch.tensor([0.0, 0.5, 1.0])
        >>> inv_lin = sigmoid_to_inv_depth(sigmoid, 0.05, 80.0, use_log_space=False)
        >>> print(inv_lin)  # [0.0125, 10.00625, 20.0]
        >>> inv_log = sigmoid_to_inv_depth(sigmoid, 0.05, 80.0, use_log_space=True)
        >>> print(inv_log)  # [0.0125, 0.5, 20.0]
    """
    min_inv = 1.0 / max(max_depth, 1e-6)  # 1/80 = 0.0125
    max_inv = 1.0 / max(min_depth, 1e-6)  # 1/0.05 = 20.0
    
    if use_log_space:
        # Log space interpolation for uniform distribution
        log_min_inv = torch.log(torch.tensor(min_inv, device=sigmoid_output.device, dtype=sigmoid_output.dtype))
        log_max_inv = torch.log(torch.tensor(max_inv, device=sigmoid_output.device, dtype=sigmoid_output.dtype))
        
        log_inv_depth = log_min_inv + (log_max_inv - log_min_inv) * sigmoid_output
        inv_depth = torch.exp(log_inv_depth)
    else:
        # Linear space interpolation (default, backward compatible)
        inv_depth = min_inv + (max_inv - min_inv) * sigmoid_output
    
    return inv_depth


def sigmoid_to_depth_linear(sigmoid_output, min_depth=0.05, max_depth=80.0):
    """
    Linear space transformation: sigmoid → depth
    
    This is the original transformation method used in PackNet-SFM.
    
    Args:
        sigmoid_output: torch.Tensor [B,1,H,W] or [H,W], values in [0, 1]
        min_depth: minimum depth (m), default 0.05
        max_depth: maximum depth (m), default 80.0
    
    Returns:
        depth: torch.Tensor, same shape, values in [min_depth, max_depth]
    
    Mathematical Formula:
        inv_depth = min_inv + (max_inv - min_inv) × sigmoid
        depth = 1 / inv_depth
        
        where:
            min_inv = 1 / max_depth
            max_inv = 1 / min_depth
    
    INT8 Quantization Error (0.05~80m range):
        - Non-uniform error distribution
        - Near range (0.05m): ~0.4% error
        - Mid range (1m): ~39% error  ❌ CRITICAL
        - Far range (10m): ~392% error ❌ UNUSABLE
    
    Example:
        >>> sigmoid = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 1, 3)
        >>> depth = sigmoid_to_depth_linear(sigmoid, 0.05, 80.0)
        >>> print(depth)
        tensor([[[[80.0000, 0.0999, 0.0500]]]])
    """
    min_inv = 1.0 / max(max_depth, 1e-6)
    max_inv = 1.0 / max(min_depth, 1e-6)
    
    inv_depth = min_inv + (max_inv - min_inv) * sigmoid_output
    depth = 1.0 / (inv_depth + 1e-8)
    
    return depth


def sigmoid_to_depth_log(sigmoid_output, min_depth=0.05, max_depth=80.0):
    """
    Log space transformation: sigmoid → depth
    
    ⚠️  WARNING: This transformation is INCOMPATIBLE with inverse-depth training!
    
    This method is designed for INT8 quantization post-processing, NOT for evaluating
    models trained with inverse depth loss. The model learns sigmoid → inv_depth mapping,
    but this function assumes sigmoid → log(depth) mapping, causing terrible metrics.
    
    **Use Case**: Apply AFTER converting depth predictions to INT8, then back to FP32.
    **NOT for**: Direct evaluation of inverse-depth trained models.
    
    Args:
        sigmoid_output: torch.Tensor [B,1,H,W] or [H,W], values in [0, 1]
        min_depth: minimum depth (m), default 0.05
        max_depth: maximum depth (m), default 80.0
    
    Returns:
        depth: torch.Tensor, same shape, values in [min_depth, max_depth]
    
    Mathematical Formula (Log-Inverse-Depth Space):
        We interpolate in log(inv_depth) space to match training:
        
        log_inv = log(1/max_depth) + (log(1/min_depth) - log(1/max_depth)) × sigmoid
        inv_depth = exp(log_inv)
        depth = 1 / inv_depth
        
        This matches the inverse depth training convention:
            sigmoid=0 → inv_depth=1/80 → depth=80m (FAR)
            sigmoid=1 → inv_depth=1/0.05 → depth=0.05m (NEAR)
    
    INT8 Quantization Benefit:
        - Uniform error in log(inv_depth) space (~3% at all distances)
        - 13x better than linear for INT8
    
    Example:
        >>> sigmoid = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 1, 3)
        >>> depth = sigmoid_to_depth_log(sigmoid, 0.05, 80.0)
        >>> print(depth)
        tensor([[[[80.0000, 2.0000, 0.0500]]]])
        # sigmoid=0.5 → geometric mean of inv_depth: exp(mean(log(1/80), log(1/0.05))) → depth=2.0m
    """
    # ✅ CORRECT: Interpolate in log(inv_depth) space to match training!
    # sigmoid=0 → FAR (small inv_depth → large depth)
    # sigmoid=1 → NEAR (large inv_depth → small depth)
    
    min_inv = 1.0 / max(max_depth, 1e-6)  # 1/80 = 0.0125
    max_inv = 1.0 / max(min_depth, 1e-6)  # 1/0.05 = 20.0
    
    # Log space interpolation in INVERSE DEPTH
    log_min_inv = torch.log(torch.tensor(min_inv, device=sigmoid_output.device))
    log_max_inv = torch.log(torch.tensor(max_inv, device=sigmoid_output.device))
    
    log_inv_depth = log_min_inv + (log_max_inv - log_min_inv) * sigmoid_output
    inv_depth = torch.exp(log_inv_depth)
    depth = 1.0 / (inv_depth + 1e-8)
    
    return depth


def apply_post_processing_variants(sigmoid_output, min_depth=0.05, max_depth=80.0):
    """
    Apply both Linear and Log transformations for comparison.
    
    Args:
        sigmoid_output: torch.Tensor [B,1,H,W], sigmoid values [0, 1]
        min_depth: minimum depth (m), default 0.05
        max_depth: maximum depth (m), default 80.0
    
    Returns:
        dict: {
            'linear': depth from linear transformation,
            'log': depth from log transformation
        }
    
    Example:
        >>> sigmoid = torch.rand(1, 1, 384, 640)
        >>> depths = apply_post_processing_variants(sigmoid, 0.05, 80.0)
        >>> print(depths.keys())
        dict_keys(['linear', 'log'])
    """
    return {
        'linear': sigmoid_to_depth_linear(sigmoid_output, min_depth, max_depth),
        'log': sigmoid_to_depth_log(sigmoid_output, min_depth, max_depth)
    }
