# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional


def autopad(k, p=None, d=1):
    """Auto-padding calculation"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution"""
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions - YOLOv8 핵심 블록"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv8"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class YOLOv8Backbone(nn.Module):
    """
    YOLOv8 Backbone for depth estimation
    """
    # YOLOv8 model configurations [depth_multiple, width_multiple, max_channels]
    model_configs = {
        'n': [0.33, 0.25, 1024],  # YOLOv8n
        's': [0.33, 0.50, 1024],  # YOLOv8s  
        'm': [0.67, 0.75, 576],   # YOLOv8m  <-- pretrained weight 에 맞춘 값
        'l': [1.00, 1.00, 512],   # YOLOv8l
        'x': [1.00, 1.25, 512],   # YOLOv8x
    }

    def __init__(self, variant='s', pretrained=True):
        """
        Initialize YOLOv8 backbone
        
        Parameters
        ----------
        variant : str
            YOLOv8 variant ('n', 's', 'm', 'l', 'x')
        pretrained : bool
            Whether to load ImageNet pretrained weights
        """
        super().__init__()
        
        if variant not in self.model_configs:
            raise ValueError(f"Unsupported YOLOv8 variant: {variant}. Choose from {list(self.model_configs.keys())}")
        
        self.variant = variant
        depth_multiple, width_multiple, max_channels = self.model_configs[variant]
        
        # Function to calculate channels
        def make_divisible(x, divisor=8):
            return math.ceil(x / divisor) * divisor
        
        def scale_channels(channels):
            return make_divisible(min(channels * width_multiple, max_channels))
        
        def scale_depth(depth):
            return max(round(depth * depth_multiple), 1)
        
        # YOLOv8 backbone structure
        # Input: 3 channels
        self.conv1 = Conv(3, scale_channels(64), 3, 2)  # P1/2
        self.conv2 = Conv(scale_channels(64), scale_channels(128), 3, 2)  # P2/4
        self.c2f1 = C2f(scale_channels(128), scale_channels(128), scale_depth(3), True)
        
        self.conv3 = Conv(scale_channels(128), scale_channels(256), 3, 2)  # P3/8
        self.c2f2 = C2f(scale_channels(256), scale_channels(256), scale_depth(6), True)
        
        self.conv4 = Conv(scale_channels(256), scale_channels(512), 3, 2)  # P4/16
        self.c2f3 = C2f(scale_channels(512), scale_channels(512), scale_depth(6), True)
        
        self.conv5 = Conv(scale_channels(512), scale_channels(1024), 3, 2)  # P5/32
        self.c2f4 = C2f(scale_channels(1024), scale_channels(1024), scale_depth(3), True)
        
        self.sppf = SPPF(scale_channels(1024), scale_channels(1024), 5)
        
        # Store output channels for decoder
        self.out_channels = [
            scale_channels(64),   # P1
            scale_channels(128),  # P2  
            scale_channels(256),  # P3
            scale_channels(512),  # P4
            scale_channels(1024), # P5
        ]
        
        print(f"🎯 YOLOv8{variant.upper()} Backbone initialized")
        print(f"   Output channels: {self.out_channels}")
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load ImageNet pretrained weights from ultralytics"""
        try:
            import ultralytics
            from ultralytics import YOLO
            
            # Download and load YOLOv8 pretrained model
            model_name = f'yolov8{self.variant}.pt'
            print(f"🔄 Loading {model_name} pretrained weights...")
            
            yolo_model = YOLO(model_name)
            pretrained_state = yolo_model.model.state_dict()
            
            # Create mapping for backbone weights
            backbone_state = {}
            current_state = self.state_dict()
            
            # Mapping YOLOv8 backbone to our backbone
            mapping = {
                'model.0': 'conv1',
                'model.1': 'conv2', 
                'model.2': 'c2f1',
                'model.3': 'conv3',
                'model.4': 'c2f2',
                'model.5': 'conv4',
                'model.6': 'c2f3',
                'model.7': 'conv5',
                'model.8': 'c2f4',
                'model.9': 'sppf',
            }
            
            for yolo_key, our_key in mapping.items():
                for param_name, param_value in pretrained_state.items():
                    if param_name.startswith(yolo_key + '.'):
                        new_key = param_name.replace(yolo_key + '.', our_key + '.')
                        if new_key in current_state:
                            backbone_state[new_key] = param_value
            
            # Load compatible weights
            missing_keys, unexpected_keys = self.load_state_dict(backbone_state, strict=False)
            
            print(f"✅ Loaded pretrained weights")
            print(f"   Loaded: {len(backbone_state)} parameters")
            if missing_keys:
                print(f"   Missing: {len(missing_keys)} parameters")
            
        except ImportError:
            print("⚠️ ultralytics not installed. Install with: pip install ultralytics")
            print("🔧 Initializing with random weights")
        except Exception as e:
            print(f"⚠️ Failed to load pretrained weights: {e}")
            print("🔧 Initializing with random weights")
    
    def forward(self, x):
        """
        Forward pass
        
        Returns
        -------
        features : list of torch.Tensor
            Multi-scale features [P1, P2, P3, P4, P5]
        """
        features = []
        
        # P1/2
        x = self.conv1(x)
        features.append(x)
        
        # P2/4
        x = self.conv2(x)
        x = self.c2f1(x)
        features.append(x)
        
        # P3/8  
        x = self.conv3(x)
        x = self.c2f2(x)
        features.append(x)
        
        # P4/16
        x = self.conv4(x)
        x = self.c2f3(x)
        features.append(x)
        
        # P5/32
        x = self.conv5(x)
        x = self.c2f4(x)
        x = self.sppf(x)
        features.append(x)
        
        return features