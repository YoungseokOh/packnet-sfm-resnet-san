# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Dual-Head Depth Decoder for Integer-Fractional depth prediction.

이 Decoder는 기존 DepthDecoder와 동일한 인터페이스를 유지하면서,
두 개의 독립적인 출력 헤드를 추가합니다.

Key Features:
- Integer Head: 정수부 예측 (0 ~ max_depth meters)
- Fractional Head: 소수부 예측 (0 ~ 1 meter)
- INT8 양자화 친화적 설계 (±2mm precision for fractional)
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from .layers import ConvBlock, Conv3x3, upsample


class DualHeadDepthDecoder(nn.Module):
    """
    Integer-Fractional Dual-Head Depth Decoder
    
    기존 DepthDecoder와 동일한 upsampling 구조를 사용하되,
    최종 출력 헤드만 2개로 분리합니다.
    
    Parameters
    ----------
    num_ch_enc : list of int
        Encoder channel counts (e.g., [64, 64, 128, 256, 512])
    scales : list of int
        Which scales to produce outputs (default: [0, 1, 2, 3])
    max_depth : float
        Maximum depth for integer head (default: 15.0)
    use_skips : bool
        Whether to use skip connections (default: True)
    
    Outputs
    -------
    - ("integer", scale): [B, 1, H, W] sigmoid [0, 1] → represents [0, max_depth]
    - ("fractional", scale): [B, 1, H, W] sigmoid [0, 1] → represents [0, 1]m
    
    Example
    -------
    >>> decoder = DualHeadDepthDecoder(num_ch_enc=[64, 64, 128, 256, 512], max_depth=15.0)
    >>> outputs = decoder(encoder_features)
    >>> integer_sigmoid = outputs[("integer", 0)]  # [B, 1, H, W]
    >>> fractional_sigmoid = outputs[("fractional", 0)]  # [B, 1, H, W]
    """
    
    def __init__(self, num_ch_enc, scales=range(4), max_depth=15.0, use_skips=True):
        super(DualHeadDepthDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.max_depth = max_depth
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        
        # Decoder channel counts (기존 DepthDecoder와 동일)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # ========================================
        # 공통 Upsampling Layers (기존과 100% 동일)
        # ========================================
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0: channel reduction
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1: skip connection fusion
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # ========================================
        # Dual-Head: 각 스케일별로 2개의 출력 헤드
        # ========================================
        for s in self.scales:
            # Integer Head (정수부 예측: 0~max_depth)
            self.convs[("integer_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)
            
            # Fractional Head (소수부 예측: 0~1m)
            self.convs[("fractional_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        
        print(f"🔧 DualHeadDepthDecoder initialized:")
        print(f"   Max depth: {max_depth}m")
        print(f"   Scales: {list(scales)}")
        print(f"   Integer quantization interval: {max_depth/255:.4f}m ({max_depth/255*1000:.2f}mm)")
        print(f"   Fractional quantization interval: {1.0/255:.4f}m ({1.0/255*1000:.2f}mm = 3.92mm)")

    def forward(self, input_features):
        """
        Forward pass
        
        Parameters
        ----------
        input_features : list of torch.Tensor
            Encoder features [feat0, feat1, ..., feat4]
            각 feature의 shape: [B, C, H, W]
        
        Returns
        -------
        outputs : dict
            {
                ("integer", scale): [B, 1, H, W] sigmoid [0,1],
                ("fractional", scale): [B, 1, H, W] sigmoid [0,1]
            }
        """
        self.outputs = {}

        # ========================================
        # 공통 Decoder Processing (기존 DepthDecoder와 동일)
        # ========================================
        x = input_features[-1]
        for i in range(4, -1, -1):
            # Upsample
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            
            # Skip connection
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            
            # ========================================
            # Dual-Head Outputs
            # ========================================
            if i in self.scales:
                # Integer Head: [0, 1] sigmoid → represents [0, max_depth]
                integer_raw = self.convs[("integer_conv", i)](x)
                self.outputs[("integer", i)] = self.sigmoid(integer_raw)
                
                # Fractional Head: [0, 1] sigmoid → represents [0, 1]m
                fractional_raw = self.convs[("fractional_conv", i)](x)
                self.outputs[("fractional", i)] = self.sigmoid(fractional_raw)

        return self.outputs
