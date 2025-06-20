# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDepthHead(nn.Module):
    """Simplified depth prediction head"""
    def __init__(self, c1, c2=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 2, c2, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv(x)


class YOLOv8DepthDecoder(nn.Module):
    """
    🔧 Print 제거된 YOLOv8 depth decoder
    """
    def __init__(self, encoder_channels, scales=range(4), verbose=False):
        super().__init__()
        
        self.scales = scales
        self.encoder_channels = encoder_channels
        self.verbose = verbose
        
        if verbose:
            print(f"🏗️ YOLOv8DepthDecoder")
            print(f"   Encoder channels: {encoder_channels}")
            print(f"   Prediction scales: {list(scales)}")
        
        # Fixed decoder channel
        self.decoder_channel = 64
        
        # Feature conversion layers
        self.feature_convs = nn.ModuleList()
        for ch in encoder_channels:
            self.feature_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, self.decoder_channel, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.decoder_channel),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Fusion layers for top-down processing
        self.fusion_convs = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.decoder_channel * 2, self.decoder_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.decoder_channel),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Depth prediction heads
        self.depth_heads = nn.ModuleDict()
        for scale in self.scales:
            self.depth_heads[f"scale_{scale}"] = SimpleDepthHead(self.decoder_channel)
    
    def forward(self, features):
        """
        🔧 Print 제거된 forward pass
        """
        # Convert all features to decoder channels
        converted_features = []
        for i, feat in enumerate(features):
            converted = self.feature_convs[i](feat)
            converted_features.append(converted)
        
        # Top-down processing
        outputs = {}
        x = converted_features[-1]  # Start from P5
        
        # Generate prediction at deepest level if requested
        if (len(converted_features) - 1) in self.scales:
            depth = self.depth_heads[f"scale_{len(converted_features) - 1}"](x)
            outputs[f"disp_{len(converted_features) - 1}"] = depth
        
        # Process from P4 to P1
        for i in range(len(converted_features) - 2, -1, -1):
            # Upsample current feature
            x_up = F.interpolate(x, size=converted_features[i].shape[-2:], mode='nearest')
            
            # Fuse features
            fused = torch.cat([x_up, converted_features[i]], dim=1)
            x = self.fusion_convs[len(converted_features) - 2 - i](fused)
            
            # Generate depth prediction at this scale
            scale_idx = i
            if scale_idx in self.scales:
                depth = self.depth_heads[f"scale_{scale_idx}"](x)
                outputs[f"disp_{scale_idx}"] = depth
        
        return outputs