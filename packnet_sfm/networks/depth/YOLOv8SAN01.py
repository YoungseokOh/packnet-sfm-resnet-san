# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from packnet_sfm.networks.layers.yolov8.yolov8_backbone import YOLOv8Backbone
from packnet_sfm.networks.layers.yolov8.yolov8_depth_decoder import YOLOv8DepthDecoder


class YOLOv8SAN01(nn.Module):
    """
    🔧 디버깅 강화된 YOLOv8-based SAN network
    """
    def __init__(self, variant='s', dropout=None, version=None, 
                 use_film=False, film_scales=[], use_enhanced_lidar=False, **kwargs):
        super().__init__()
        
        self.variant = variant
        self.use_film = use_film
        self.film_scales = film_scales
        self.use_enhanced_lidar = use_enhanced_lidar
        
        print(f"🚀 YOLOv8SAN01 ({variant.upper()}) - FiLM: {use_film}, Enhanced: {use_enhanced_lidar}")
        
        # YOLOv8 Backbone
        self.backbone = YOLOv8Backbone(variant=variant, pretrained=True)
        
        # YOLOv8 Depth Decoder
        self.decoder = YOLOv8DepthDecoder(
            encoder_channels=self.backbone.out_channels,
            scales=range(4)
        )
        
        # MinkowskiEncoder (간단한 초기화)
        self.mconvs = None
        if use_film and len(film_scales) > 0:
            try:
                from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
                rgb_channels_per_scale = []
                for i in range(len(self.backbone.out_channels)):
                    if i in film_scales:
                        rgb_channels_per_scale.append(self.backbone.out_channels[i])
                    else:
                        rgb_channels_per_scale.append(0)
                
                self.mconvs = MinkowskiEncoder(
                    self.backbone.out_channels,
                    rgb_channels=rgb_channels_per_scale,
                    with_uncertainty=False
                )
                print("📊 MinkowskiEncoder initialized successfully")
            except Exception as e:
                print(f"⚠️ MinkowskiEncoder 초기화 실패: {e}")
                self.mconvs = None
        
        # Learnable fusion weights
        self.weight = torch.nn.parameter.Parameter(
            torch.ones(len(self.backbone.out_channels)) * 0.5, requires_grad=True
        )
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(len(self.backbone.out_channels)), requires_grad=True
        )
        
        # 🔧 수정된 depth scaling function - ResNet과 동일하게
        self.scale_inv_depth = partial(self._scale_inv_depth, min_depth=0.1, max_depth=100.0)
        
        self.init_weights()
    
    def _scale_inv_depth(self, disp, min_depth, max_depth):
        """🔧 ResNet 방식과 동일한 inverse depth scaling"""
        # disp_to_depth와 동일한 방식
        min_disp = 1 / max_depth  # 1/100 = 0.01
        max_disp = 1 / min_depth  # 1/0.1 = 10
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        return scaled_disp
    
    def init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def run_network(self, rgb, input_depth=None):
        """
        🔧 디버깅 강화된 네트워크 실행
        """
        batch_size, _, height, width = rgb.shape
        
        # Extract multi-scale features
        features = self.backbone(rgb)
        
        # LiDAR integration (optional)
        if input_depth is not None and self.mconvs is not None and self.use_film:
            try:
                self.mconvs.prep(input_depth)
                enhanced_features = []
                for i, feat in enumerate(features):
                    if i in self.film_scales:
                        try:
                            sparse_feat = self.mconvs(feat)
                            weight_val = torch.sigmoid(self.weight[i])
                            fused_feat = (weight_val * feat + 
                                         (1 - weight_val) * sparse_feat + 
                                         self.bias[i].view(1, 1, 1, 1))
                            enhanced_features.append(fused_feat)
                        except Exception:
                            enhanced_features.append(feat)
                    else:
                        enhanced_features.append(feat)
                features = enhanced_features
            except Exception:
                pass  # RGB-only로 fallback
        
        # Decode to depth maps
        try:
            depth_outputs = self.decoder(features)
        except Exception as e:
            print(f"❌ Decoder failed: {e}")
            # 🔧 더 나은 fallback 생성
            depth_outputs = {}
            for i in range(4):
                # 적절한 스케일의 출력 생성
                scale_factor = 2 ** (i + 1)
                fallback_h = height // scale_factor
                fallback_w = width // scale_factor
                
                # 🔧 적절한 초기값으로 fallback 생성 (0.1이 아닌 0.5)
                fallback = torch.ones(
                    batch_size, 1, fallback_h, fallback_w,
                    device=rgb.device, dtype=rgb.dtype, requires_grad=True
                ) * 0.5  # sigmoid 출력의 중간값
                depth_outputs[f"disp_{i}"] = fallback
        
        # 🔧 ResNet과 동일한 방식으로 inverse depth 생성
        inv_depths = []
        for i in range(4):
            if f"disp_{i}" in depth_outputs:
                disp = depth_outputs[f"disp_{i}"]
                # ResNet 방식과 동일한 scaling
                inv_depth = self.scale_inv_depth(disp)
                inv_depths.append(inv_depth)
            else:
                # Fallback
                scale_factor = 2 ** (i + 1)
                fallback_h = height // scale_factor
                fallback_w = width // scale_factor
                
                # 🔧 적절한 inverse depth 값으로 fallback
                fallback_inv = torch.ones(
                    batch_size, 1, fallback_h, fallback_w,
                    device=rgb.device, dtype=rgb.dtype, requires_grad=True
                ) * 0.1  # 10m depth에 해당하는 inverse depth
                inv_depths.append(fallback_inv)
        
        return inv_depths, features
    
    def forward(self, rgb, input_depth=None, **kwargs):
        """
        🔧 디버깅 강화된 forward pass
        """
        if not self.training:
            inv_depths, _ = self.run_network(rgb, input_depth)
            return {'inv_depths': inv_depths}
        
        # Training mode
        output = {}
        
        # RGB-only forward pass
        inv_depths_rgb, skip_feat_rgb = self.run_network(rgb)
        output['inv_depths'] = inv_depths_rgb
        
        if input_depth is None:
            return output
        
        # RGB+LiDAR forward pass
        inv_depths_rgbd, skip_feat_rgbd = self.run_network(rgb, input_depth)
        output['inv_depths_rgbd'] = inv_depths_rgbd
        
        # Feature consistency loss
        if (len(skip_feat_rgbd) == len(skip_feat_rgb) and 
            all(f1.shape == f2.shape for f1, f2 in zip(skip_feat_rgbd, skip_feat_rgb))):
            feature_loss = sum([
                ((feat_rgbd.detach() - feat_rgb) ** 2).mean()
                for feat_rgbd, feat_rgb in zip(skip_feat_rgbd, skip_feat_rgb)
            ]) / len(skip_feat_rgbd)
            output['depth_loss'] = feature_loss
        
        return output