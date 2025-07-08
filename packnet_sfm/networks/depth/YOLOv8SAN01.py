import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder


class YOLOv8Neck(nn.Module):
    """YOLOv8 Neck (feature pyramid) part."""
    
    def __init__(self, backbone_channels, variant='s'):
        super().__init__()
        
        # YOLOv8 variant-specific settings
        depth_multiple, width_multiple, max_channels = {
            'n': [0.33, 0.25, 1024],
            's': [0.33, 0.50, 1024],
            'm': [0.67, 0.75, 576],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.25, 512],
        }[variant]
        
        def make_divisible(x, divisor=8):
            return math.ceil(x / divisor) * divisor
        
        def scale_channels(channels):
            return make_divisible(min(channels * width_multiple, max_channels))
        
        # YOLOv8 Neck Feature Processing
        self.neck_convs = nn.ModuleDict()
        
        # P3 (1/8) processing
        self.neck_convs['P3'] = nn.Sequential(
            Conv(backbone_channels[2], scale_channels(256), 3, 1),
            C2f(scale_channels(256), scale_channels(256), 1, True),
            Conv(scale_channels(256), scale_channels(256), 3, 1)
        )
        
        # P4 (1/16) processing  
        self.neck_convs['P4'] = nn.Sequential(
            Conv(backbone_channels[3], scale_channels(512), 3, 1),
            C2f(scale_channels(512), scale_channels(512), 1, True),
            Conv(scale_channels(512), scale_channels(512), 3, 1)
        )
        
        # P5 (1/32) processing
        self.neck_convs['P5'] = nn.Sequential(
            Conv(backbone_channels[4], scale_channels(1024), 3, 1),
            C2f(scale_channels(1024), scale_channels(1024), 1, True),
            Conv(scale_channels(1024), scale_channels(1024), 3, 1)
        )
        
        # Additional layers for P1, P2 levels
        self.neck_convs['P1'] = nn.Sequential(
            Conv(backbone_channels[0], scale_channels(64), 3, 1),
            C2f(scale_channels(64), scale_channels(64), 1, True)
        )
        
        self.neck_convs['P2'] = nn.Sequential(
            Conv(backbone_channels[1], scale_channels(128), 3, 1),
            C2f(scale_channels(128), scale_channels(128), 1, True)
        )
        
        # Output channels (ResNet compatible)
        self.output_channels = [
            scale_channels(64),   # P1
            scale_channels(128),  # P2
            scale_channels(256),  # P3
            scale_channels(512),  # P4
            scale_channels(1024), # P5
        ]
        
        print(f"🎯 YOLOv8 Neck:")
        print(f"   Output channels: {self.output_channels}")
    
    def forward(self, backbone_features):
        neck_features = []
        
        # P1, P2 simple processing
        neck_features.append(self.neck_convs['P1'](backbone_features[0]))  # P1
        neck_features.append(self.neck_convs['P2'](backbone_features[1]))  # P2
        
        # P3, P4, P5 detection-style processing
        neck_features.append(self.neck_convs['P3'](backbone_features[2]))  # P3
        neck_features.append(self.neck_convs['P4'](backbone_features[3]))  # P4
        neck_features.append(self.neck_convs['P5'](backbone_features[4]))  # P5
        
        return neck_features


class DepthNeck(nn.Module):
    """A simple FPN-style neck for depth estimation."""
    def __init__(self, backbone_channels, variant='s'):
        super().__init__()

        # YOLOv8 variant-specific settings for channel scaling
        depth_multiple, width_multiple, max_channels = {
            'n': [0.33, 0.25, 1024],
            's': [0.33, 0.50, 1024],
            'm': [0.67, 0.75, 576],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.25, 512],
        }[variant]

        def make_divisible(x, divisor=8):
            return math.ceil(x / divisor) * divisor

        def scale_channels(channels):
            return make_divisible(min(channels * width_multiple, max_channels))

        # Output channels (must match YOLOv8Neck for compatibility)
        self.output_channels = [
            scale_channels(64),
            scale_channels(128),
            scale_channels(256),
            scale_channels(512),
            scale_channels(1024),
        ]

        # Lateral connections to process backbone features
        self.lateral_convs = nn.ModuleList()
        for i in range(5):
            self.lateral_convs.append(
                Conv(backbone_channels[i], self.output_channels[i], k=1)
            )

        # Top-down pathway with upsampling and fusion
        self.top_down_convs = nn.ModuleList()
        for i in range(4, 0, -1):
            self.top_down_convs.append(
                Conv(self.output_channels[i] + self.output_channels[i-1], self.output_channels[i-1], k=3)
            )
            
        print(f"🎯 DepthNeck (FPN-style):")
        print(f"   Output channels: {self.output_channels}")

    def forward(self, backbone_features):
        # 1. Apply lateral connections to backbone features
        lateral_features = [
            self.lateral_convs[i](backbone_features[i]) for i in range(5)
        ]

        # 2. Top-down pathway with upsampling and fusion
        fused_features = [lateral_features[4]]
        for i in range(3, -1, -1):
            # Upsample the higher-level feature
            upsampled_feat = F.interpolate(fused_features[-1], 
                                           size=lateral_features[i].shape[2:], 
                                           mode='bilinear', 
                                           align_corners=False)
            # Concatenate with the current level's lateral feature
            concat_feat = torch.cat([upsampled_feat, lateral_features[i]], dim=1)
            
            # Fuse and refine
            fused_feat = self.top_down_convs[3-i](concat_feat)
            fused_features.append(fused_feat)
        
        # Reverse the list to be from P1 to P5
        fused_features = fused_features[::-1]
        
        return fused_features


# YOLOv8 basic modules
class Conv(nn.Module):
    """Standard convolution"""
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2f(nn.Module):
    """Simplified YOLOv8 C2f block"""
    def __init__(self, c1, c2, n=1, shortcut=False):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([nn.Identity() for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class YOLOv8SAN01(nn.Module):
    """
    🆕 YOLOv8-based SAN network with Neck Features and ImageNet support
    """
    def __init__(self,
                 variant='s',
                 use_film=False,
                 film_scales=[0],
                 use_neck_features=False,
                 use_depth_neck=False,
                 use_imagenet_pretrained=False,
                 **kwargs):  # neck_type 제거
        super().__init__()
    
        self.variant = variant
        self.use_neck_features = use_neck_features
        self.use_depth_neck = use_depth_neck
        self.use_imagenet_pretrained = use_imagenet_pretrained
 
        print(f"🏗️ Initializing YOLOv8SAN01 with YOLOv8{variant}")
        print(f"   Use ImageNet pretrained: {use_imagenet_pretrained}")
        print(f"   Use YOLO Neck Features: {use_neck_features}")
        print(f"   Use Depth Neck: {use_depth_neck}")
 
         # Load backbone
        try:
            if use_imagenet_pretrained:
                model_name = f'yolov8{variant}-cls.pt'
                print(f"🔄 Loading ImageNet classification model: {model_name}")
            else:
                model_name = f'yolov8{variant}.pt'
                print(f"🔄 Loading COCO detection model: {model_name}")
        
            temp_model = YOLO(model_name)
            self.backbone = temp_model.model.model
            del temp_model
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            print("🔄 Falling back to COCO detection model...")
            temp_model = YOLO(f'yolov8{variant}.pt')
            self.backbone = temp_model.model.model
            del temp_model
            self.use_imagenet_pretrained = False
    
        # ResNet compatible channel structure (fixed)
        self.resnet_channels = [64, 64, 128, 256, 512]
        
        # YOLOv8 actual extracted channels
        yolo_channel_configs = {
            'n': [32, 64, 128, 256, 512],
            's': [64, 64, 128, 256, 256],
            'm': [48, 96, 192, 384, 576],
            'l': [64, 128, 256, 512, 512],
            'x': [80, 160, 320, 640, 640],
        }
        self.yolo_channels = yolo_channel_configs.get(variant, [64, 64, 128, 256, 256])
        
        # 🔧 Verify exact channel counts via test extraction
        print(f"🔧 Testing actual YOLOv8 backbone channels...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 352, 1216)
            actual_channels = self._test_backbone_channels(dummy_input)
    
        print(f"🔧 Actual backbone channels: {actual_channels}")
        print(f"🔧 Expected YOLO channels: {self.yolo_channels}")
    
        # 🔧 Update with actual channels
        if len(actual_channels) == 5:
            self.yolo_channels = actual_channels
            print(f"✅ Updated to actual channels: {self.yolo_channels}")
    
        # 🔧 Depth Neck 우선 처리
        if self.use_depth_neck:
            print("✅ Depth Neck enabled: applying DepthNeck for feature fusion")
            self.neck = DepthNeck(self.yolo_channels, variant=self.variant)
            # ── 여기서 output_channels 를 실제 YOLO 채널과 일치시켜 어댑터 mismatch 방지
            self.neck.output_channels = self.yolo_channels
            adapter_input_channels = self.neck.output_channels
        elif self.use_neck_features:
            print("🔧 Applying YOLOv8Neck for feature fusion")
            self.neck = YOLOv8Neck(self.yolo_channels, variant=self.variant)
            adapter_input_channels = self.neck.output_channels
        else:
             adapter_input_channels = self.yolo_channels
    
        print(f"🔧 Final adapter input channels: {adapter_input_channels}")
        print(f"🔧 ResNet target channels: {self.resnet_channels}")
    
        # Create feature adapters
        self.feature_adapters = nn.ModuleList()
        for i, (input_ch, resnet_ch) in enumerate(zip(adapter_input_channels, self.resnet_channels)):
            if input_ch != resnet_ch:
                adapter = nn.Sequential(
                    nn.Conv2d(input_ch, resnet_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(resnet_ch),
                    nn.ReLU(inplace=True)
                )
                print(f"   Adapter {i}: {input_ch} -> {resnet_ch} channels")
            else:
                adapter = nn.Identity()
                print(f"   Adapter {i}: {input_ch} channels (no change)")
            self.feature_adapters.append(adapter)
    
        # ResNet DepthDecoder
        self.decoder = DepthDecoder(num_ch_enc=self.resnet_channels)
    
        # SAN settings
        self.use_film = use_film
        self.film_scales = film_scales
        self.use_enhanced_lidar = kwargs.get('use_enhanced_lidar', False)
    
        # FiLM configuration
        rgb_channels_per_scale = None
        if use_film:
            rgb_channels_per_scale = []
            for i in range(len(self.resnet_channels)):
                if i in film_scales:
                    rgb_channels_per_scale.append(self.resnet_channels[i])
                else:
                    rgb_channels_per_scale.append(0)
    
        # Setup Minkowski encoder
        self._setup_minkowski_encoder(rgb_channels_per_scale)
    
        # Learnable fusion weights
        self.weight = torch.nn.parameter.Parameter(
            torch.ones(5) * 0.5, requires_grad=True
        )
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(5), requires_grad=True
        )
    
        print(f"🎯 Final configuration:")
        print(f"   YOLOv8 channels: {self.yolo_channels}")
        print(f"   Adapter input: {adapter_input_channels}")
        print(f"   ResNet channels: {self.resnet_channels}")
        print(f"   FiLM enabled: {use_film}")
        print(f"   FiLM scales: {film_scales}")
    
        self.init_weights()

    def _test_backbone_channels(self, dummy_input):
        """Test actual channels extracted from the backbone"""
        features = []
        current_x = dummy_input
        feature_indices = [1, 2, 4, 6, 9]
        
        try:
            for i, layer in enumerate(self.backbone):
                try:
                    current_x = layer(current_x)
                    
                    # Handle tuple returns
                    if isinstance(current_x, (tuple, list)):
                        current_x = current_x[0]
                    
                    # Extract at feature points
                    if i in feature_indices and len(features) < 5:
                        if hasattr(current_x, 'shape') and len(current_x.shape) == 4:
                            features.append(current_x.shape[1])  # Store only channel count
                        elif hasattr(current_x, 'shape') and len(current_x.shape) == 2:
                            break
                    
                    if len(features) >= 5:
                        break
                        
                except Exception as e:
                    break
    
        except Exception as e:
            print(f"⚠️ Backbone test failed: {e}")
            return self.yolo_channels  # fallback
    
        # Use default if less than 5 features are extracted
        if len(features) < 5:
            print(f"⚠️ Only extracted {len(features)} features, using default channels")
            return self.yolo_channels
    
        return features

    def _setup_minkowski_encoder(self, rgb_channels_per_scale):
        """Setup Minkowski encoder"""
        from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
        
        if rgb_channels_per_scale is not None:
            self.mconvs = MinkowskiEncoder(
                self.resnet_channels,
                rgb_channels=rgb_channels_per_scale,
                with_uncertainty=False
            )
        else:
            self.mconvs = MinkowskiEncoder(
                self.resnet_channels,
                with_uncertainty=False
            )

    def init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def extract_features(self, x):
        """ResNet decoder compatible feature extraction with Neck support"""
        features = []
        
        try:
            current_x = x
            feature_indices = [1, 2, 4, 6, 9]
            
            for i, layer in enumerate(self.backbone):
                try:
                    current_x = layer(current_x)
                    
                    # Handle tuple returns
                    if isinstance(current_x, (tuple, list)):
                        current_x = current_x[0]
                    
                    # Extract at feature points
                    if i in feature_indices and len(features) < 5:
                        if hasattr(current_x, 'shape') and len(current_x.shape) == 4:
                            features.append(current_x)
                        elif hasattr(current_x, 'shape') and len(current_x.shape) == 2:
                            # Stop for 2D output from ImageNet classification models
                            break
                    
                    if len(features) >= 5:
                        break
                        
                except Exception as e:
                    break
            
            # Reconstruct features in ResNet style
            if len(features) >= 3:
                # Calculate correct resolution based on original input size
                input_h, input_w = x.shape[-2:]
                
                # Exact spatial hierarchy of ResNet encoder
                target_sizes = [
                    (input_h // 2, input_w // 2),    # 176x608  (1/2)
                    (input_h // 4, input_w // 4),    # 88x304   (1/4) 
                    (input_h // 8, input_w // 8),    # 44x152   (1/8)
                    (input_h // 16, input_w // 16),  # 22x76    (1/16)
                    (input_h // 32, input_w // 32),  # 11x38    (1/32)
                ]
                
                resnet_features = []
                
                # Adjust each feature to its target size
                for i, (feat, target_size) in enumerate(zip(features, target_sizes)):
                    current_size = feat.shape[-2:]
                    
                    if current_size != target_size:
                        # Interpolate to correct size
                        adjusted_feat = torch.nn.functional.interpolate(
                            feat, size=target_size, mode='bilinear', align_corners=False
                        )
                        resnet_features.append(adjusted_feat)
                    else:
                        resnet_features.append(feat)
                
                # Fill in missing features by downsampling the last one
                while len(resnet_features) < 5:
                    if resnet_features:
                        last_feat = resnet_features[-1]
                        target_size = target_sizes[len(resnet_features)]
                        
                        # 2x downsampling + channel increase
                        downsampled = torch.nn.functional.avg_pool2d(last_feat, 2)
                        
                        # Increase channel count (e.g., 256 -> 512)
                        if downsampled.shape[1] != self.yolo_channels[len(resnet_features)]:
                            with torch.no_grad():
                                channel_expander = nn.Conv2d(
                                    downsampled.shape[1], 
                                    self.yolo_channels[len(resnet_features)], 
                                    1, bias=False
                                ).to(downsampled.device)
                                downsampled = channel_expander(downsampled)
                        
                        # Adjust size
                        if downsampled.shape[-2:] != target_size:
                            downsampled = torch.nn.functional.interpolate(
                                downsampled, size=target_size, mode='bilinear', align_corners=False
                            )
                        
                        resnet_features.append(downsampled)
                    else:
                        break
                
                features = resnet_features[:5]
            
            else:
                # Fallback: sequential ResNet-style downsampling from input
                features = []
                input_h, input_w = x.shape[-2:]
                
                for i in range(5):
                    scale_factor = 2 ** (i + 1)
                    target_h = input_h // scale_factor
                    target_w = input_w // scale_factor
                    
                    # Simple downsampling
                    downsampled = torch.nn.functional.avg_pool2d(x, scale_factor)
                    
                    # Resize to exact target if needed
                    if downsampled.shape[-2:] != (target_h, target_w):
                        downsampled = torch.nn.functional.interpolate(
                            downsampled, size=(target_h, target_w), mode='bilinear', align_corners=False
                        )
                    
                    # Channel adjustment
                    if downsampled.shape[1] != self.yolo_channels[i]:
                        with torch.no_grad():
                            channel_adjuster = nn.Conv2d(
                                downsampled.shape[1], self.yolo_channels[i], 1, bias=False
                            ).to(x.device)
                            downsampled = channel_adjuster(downsampled)
                    
                    features.append(downsampled)
            
            # 🆕 Apply Neck (optional)
            if self.use_neck_features and hasattr(self, 'neck'):
                try:
                    neck_features = self.neck(features[:5])
                    features = neck_features
                    
                except Exception as neck_error:
                    print(f"❌ Neck failed: {neck_error}")
            return features[:5]
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            # Complete fallback
            features = []
            input_h, input_w = x.shape[-2:]
            
            for i in range(5):
                scale_factor = 2 ** (i + 1)
                height = input_h // scale_factor
                width = input_w // scale_factor
                channels = self.yolo_channels[i]
                
                dummy_feat = torch.randn(x.shape[0], channels, height, width, device=x.device)
                features.append(dummy_feat)
            
            return features

    def run_network(self, rgb, input_depth=None):
        """Network execution"""
        # YOLOv8 feature extraction
        yolo_features = self.extract_features(rgb)
        
        # Feature adaptation to ResNet format
        adapted_features = []
        for i, (feat, adapter) in enumerate(zip(yolo_features, self.feature_adapters)):
            adapted_feat = adapter(feat)
            adapted_features.append(adapted_feat)
        
        # LiDAR integration
        if input_depth is not None:
            self.mconvs.prep(input_depth)
            
            fused_features = []
            for i, feat in enumerate(adapted_features):
                if self.use_film and i in self.film_scales:
                    result = self.mconvs(feat)
                    
                    if isinstance(result, tuple) and len(result) == 3:
                        sparse_feat, gamma, beta = result
                        modulated_feat = gamma * feat + beta
                        fusion_weight = torch.sigmoid(self.weight[i])
                        fused_feat = (fusion_weight * modulated_feat + 
                                     (1 - fusion_weight) * sparse_feat + 
                                     self.bias[i].view(1, 1, 1, 1))
                    else:
                        sparse_feat = result
                        fusion_weight = torch.sigmoid(self.weight[i])
                        fused_feat = (fusion_weight * feat + 
                                     (1 - fusion_weight) * sparse_feat + 
                                     self.bias[i].view(1, 1, 1, 1))
                else:
                    sparse_feat = self.mconvs(feat)
                    fusion_weight = torch.sigmoid(self.weight[i])
                    fused_feat = (fusion_weight * feat + 
                                 (1 - fusion_weight) * sparse_feat + 
                                 self.bias[i].view(1, 1, 1, 1))
                
                fused_features.append(fused_feat)
            
            adapted_features = fused_features
        
        # Decode to get inverse depth maps
        inv_depths_dict = self.decoder(adapted_features)
        
        # Convert to list format
        if self.training:
            inv_depths = [
                inv_depths_dict[("disp", 0)],
                inv_depths_dict[("disp", 1)],
                inv_depths_dict[("disp", 2)],
                inv_depths_dict[("disp", 3)],
            ]
        else:
            inv_depths = [inv_depths_dict[("disp", 0)]]
        
        return inv_depths, adapted_features

    def forward(self, rgb, input_depth=None, **kwargs):
        """Forward pass"""
        if not self.training:
            inv_depths, _ = self.run_network(rgb, input_depth)
            return {'inv_depths': inv_depths}

        output = {}
        
        # RGB-only forward pass
        inv_depths_rgb, skip_feat_rgb = self.run_network(rgb)
        output['inv_depths'] = inv_depths_rgb
        
        if input_depth is None:
            return {'inv_depths': inv_depths_rgb}
        
        # RGB+D forward pass
        inv_depths_rgbd, skip_feat_rgbd = self.run_network(rgb, input_depth)
        output['inv_depths_rgbd'] = inv_depths_rgbd
        
        # Consistency loss
        if len(skip_feat_rgbd) == len(skip_feat_rgb) and len(skip_feat_rgb) > 0:
            feature_weights = torch.softmax(torch.abs(self.weight), dim=0)
            weighted_loss = sum([
                weight * ((feat_rgbd.detach() - feat_rgb) ** 2).mean()
                for weight, feat_rgbd, feat_rgb in zip(feature_weights, skip_feat_rgbd, skip_feat_rgb)
            ]) / len(skip_feat_rgbd)
            
            output['depth_loss'] = weighted_loss
        
        return output