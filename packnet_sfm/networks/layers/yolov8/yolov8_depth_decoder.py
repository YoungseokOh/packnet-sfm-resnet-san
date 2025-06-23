# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDepthHead(nn.Module):
    """간단한 깊이 예측 헤드"""
    def __init__(self, c1, c2=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 2, c2, 3, padding=1),
            nn.Sigmoid()  # 0-1 범위로 제한
        )
    
    def forward(self, x):
        return self.conv(x)


class YOLOv8DepthDecoder(nn.Module):
    """
    🔧 안정성 개선된 YOLOv8 depth decoder
    """
    def __init__(self, encoder_channels, scales=range(4), verbose=False):
        super().__init__()
        
        self.scales = scales
        self.encoder_channels = encoder_channels
        self.verbose = verbose
        
        if verbose:
            print(f"🏗️ YOLOv8DepthDecoder: {encoder_channels}")
        
        # 고정된 디코더 채널
        self.decoder_channel = 64
        
        # Feature 변환 레이어들
        self.feature_convs = nn.ModuleList()
        for ch in encoder_channels:
            self.feature_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, self.decoder_channel, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.decoder_channel),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 융합 레이어들
        self.fusion_convs = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.decoder_channel * 2, self.decoder_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.decoder_channel),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 깊이 예측 헤드들
        self.depth_heads = nn.ModuleDict()
        for scale in self.scales:
            self.depth_heads[f"scale_{scale}"] = SimpleDepthHead(self.decoder_channel)
    
    def forward(self, features):
        # 1) 디버깅: 입력 feature 크기
        if self.verbose:
            shapes = [f.shape for f in features]
            print(f"▶️ Decoder input feature shapes: {shapes}")

        # Convert features
        converted = [conv(f) for conv, f in zip(self.feature_convs, features)]

        outputs = {}
        x = converted[-1]
        # Deepest level 예측 (scale=len−1)
        if (len(converted)-1) in self.scales:
            depth = self.depth_heads[f"scale_{len(converted)-1}"](x)
            depth = torch.clamp(depth, 1e-6, 1.0 - 1e-6)
            outputs[f"disp_{len(converted)-1}"] = depth

        # Top-down
        for i in range(len(converted)-2, -1, -1):
            x = self.fusion_convs[len(converted)-2-i](
                torch.cat([
                    F.interpolate(x, size=converted[i].shape[-2:], mode='nearest'),
                    converted[i]
                ], dim=1)
            )
            if i in self.scales:
                depth = self.depth_heads[f"scale_{i}"](x)
                # 2) clamp sigmoid boundary
                depth = torch.clamp(depth, 1e-6, 1.0 - 1e-6)
                outputs[f"disp_{i}"] = depth

        # 3) 전체 출력에 nan/inf 체크
        for k, v in outputs.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"⚠️ Decoder out {k} has NaN/Inf: "
                      f"nan={int(torch.isnan(v).sum())}, inf={int(torch.isinf(v).sum())}")
                outputs[k] = torch.nan_to_num(v, nan=1e-3, posinf=1-1e-6, neginf=1e-6)

        return outputs