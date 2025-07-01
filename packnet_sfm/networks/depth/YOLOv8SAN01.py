import math
import torch
import torch.nn as nn
from ultralytics import YOLO
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder


class YOLOv8HeadFeatureExtractor(nn.Module):
    """YOLOv8 Detection Head에서 feature extraction 부분만 추출"""
    
    def __init__(self, backbone_channels, variant='s'):
        super().__init__()
        
        # YOLOv8 variant별 설정
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
        
        # YOLOv8 Head Feature Processing (detection 제외)
        self.head_convs = nn.ModuleDict()
        
        # P3 (1/8) processing
        self.head_convs['P3'] = nn.Sequential(
            Conv(backbone_channels[2], scale_channels(256), 3, 1),
            C2f(scale_channels(256), scale_channels(256), 1, True),
            Conv(scale_channels(256), scale_channels(256), 3, 1)
        )
        
        # P4 (1/16) processing  
        self.head_convs['P4'] = nn.Sequential(
            Conv(backbone_channels[3], scale_channels(512), 3, 1),
            C2f(scale_channels(512), scale_channels(512), 1, True),
            Conv(scale_channels(512), scale_channels(512), 3, 1)
        )
        
        # P5 (1/32) processing
        self.head_convs['P5'] = nn.Sequential(
            Conv(backbone_channels[4], scale_channels(1024), 3, 1),
            C2f(scale_channels(1024), scale_channels(1024), 1, True),
            Conv(scale_channels(1024), scale_channels(1024), 3, 1)
        )
        
        # P1, P2 레벨 생성을 위한 추가 레이어
        self.head_convs['P1'] = nn.Sequential(
            Conv(backbone_channels[0], scale_channels(64), 3, 1),
            C2f(scale_channels(64), scale_channels(64), 1, True)
        )
        
        self.head_convs['P2'] = nn.Sequential(
            Conv(backbone_channels[1], scale_channels(128), 3, 1),
            C2f(scale_channels(128), scale_channels(128), 1, True)
        )
        
        # 출력 채널 (ResNet 호환)
        self.output_channels = [
            scale_channels(64),   # P1
            scale_channels(128),  # P2
            scale_channels(256),  # P3
            scale_channels(512),  # P4
            scale_channels(1024), # P5
        ]
        
        print(f"🎯 YOLOv8 Head Feature Extractor:")
        print(f"   Output channels: {self.output_channels}")
    
    def forward(self, backbone_features):
        head_features = []
        
        # P1, P2는 단순 processing
        head_features.append(self.head_convs['P1'](backbone_features[0]))  # P1
        head_features.append(self.head_convs['P2'](backbone_features[1]))  # P2
        
        # P3, P4, P5는 detection head style processing
        head_features.append(self.head_convs['P3'](backbone_features[2]))  # P3
        head_features.append(self.head_convs['P4'](backbone_features[3]))  # P4
        head_features.append(self.head_convs['P5'](backbone_features[4]))  # P5
        
        return head_features


# YOLOv8 기본 모듈들
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
    """YOLOv8 C2f block 간단 버전"""
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
    🆕 YOLOv8-based SAN network with Head Features and ImageNet support
    """
    def __init__(self, variant='s', use_film=False, film_scales=[0], 
             use_head_features=False, use_imagenet_pretrained=False, **kwargs):
        super().__init__()
    
        self.variant = variant
        self.use_head_features = use_head_features
        self.use_imagenet_pretrained = use_imagenet_pretrained
        
        print(f"🏗️ Initializing YOLOv8SAN01 with YOLOv8{variant}")
        print(f"   Use ImageNet pretrained: {use_imagenet_pretrained}")
        print(f"   Use Head Features: {use_head_features}")
        
        # 백본 로드
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
    
        # ResNet 호환 채널 구조 (고정)
        self.resnet_channels = [64, 64, 128, 256, 512]
        
        # YOLOv8에서 실제 추출되는 채널
        yolo_channel_configs = {
            'n': [32, 64, 128, 256, 512],
            's': [64, 64, 128, 256, 256],
            'm': [48, 96, 192, 384, 576],
            'l': [64, 128, 256, 512, 512],
            'x': [80, 160, 320, 640, 640],
        }
        self.yolo_channels = yolo_channel_configs.get(variant, [64, 64, 128, 256, 256])
        
        # 🔧 실제 추출 테스트를 통해 정확한 채널 수 확인
        print(f"🔧 Testing actual YOLOv8 backbone channels...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 352, 1216)
            actual_channels = self._test_backbone_channels(dummy_input)
    
        print(f"🔧 Actual backbone channels: {actual_channels}")
        print(f"🔧 Expected YOLO channels: {self.yolo_channels}")
    
        # 🔧 실제 채널로 업데이트
        if len(actual_channels) == 5:
            self.yolo_channels = actual_channels
            print(f"✅ Updated to actual channels: {self.yolo_channels}")
    
        # 🔧 Head Feature Extractor 사용 시 채널 구성 변경
        if self.use_head_features:
            try:
                # Head Feature Extractor 초기화
                self.head_feature_extractor = YOLOv8HeadFeatureExtractor(
                    self.yolo_channels, variant=variant
                )
                # 실제 Head Feature Extractor의 출력 채널 사용
                adapter_input_channels = self.head_feature_extractor.output_channels
                print(f"🎯 Head Feature Extractor output channels: {adapter_input_channels}")
                
            except Exception as head_error:
                print(f"❌ Head Feature Extractor failed: {head_error}")
                print("🔧 Falling back to backbone features")
                self.use_head_features = False
                adapter_input_channels = self.yolo_channels
        else:
            adapter_input_channels = self.yolo_channels
    
        print(f"🔧 Final adapter input channels: {adapter_input_channels}")
        print(f"🔧 ResNet target channels: {self.resnet_channels}")
    
        # Feature adapter 생성
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
    
        # SAN 설정
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
    
        # Minkowski encoder 설정
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
        """실제 backbone에서 추출되는 채널 수를 테스트"""
        features = []
        current_x = dummy_input
        feature_indices = [1, 2, 4, 6, 9]
        
        try:
            for i, layer in enumerate(self.backbone):
                try:
                    current_x = layer(current_x)
                    
                    # 튜플 반환 처리
                    if isinstance(current_x, (tuple, list)):
                        current_x = current_x[0]
                    
                    # Feature extraction points에서 추출
                    if i in feature_indices and len(features) < 5:
                        if hasattr(current_x, 'shape') and len(current_x.shape) == 4:
                            features.append(current_x.shape[1])  # 채널 수만 저장
                        elif hasattr(current_x, 'shape') and len(current_x.shape) == 2:
                            break
                    
                    if len(features) >= 5:
                        break
                        
                except Exception as e:
                    break
    
        except Exception as e:
            print(f"⚠️ Backbone test failed: {e}")
            return self.yolo_channels  # fallback
    
        # 5개 미만이면 기본값 사용
        if len(features) < 5:
            print(f"⚠️ Only extracted {len(features)} features, using default channels")
            return self.yolo_channels
    
        return features

    def _setup_minkowski_encoder(self, rgb_channels_per_scale):
        """Minkowski encoder 설정"""
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
        """ResNet decoder 호환 feature extraction with Head Features support"""
        features = []
        
        try:
            current_x = x
            feature_indices = [1, 2, 4, 6, 9]
            
            for i, layer in enumerate(self.backbone):
                try:
                    current_x = layer(current_x)
                    
                    # 튜플 반환 처리
                    if isinstance(current_x, (tuple, list)):
                        current_x = current_x[0]
                    
                    # Feature extraction points에서 추출
                    if i in feature_indices and len(features) < 5:
                        if hasattr(current_x, 'shape') and len(current_x.shape) == 4:
                            features.append(current_x)
                        elif hasattr(current_x, 'shape') and len(current_x.shape) == 2:
                            # ImageNet classification model의 경우 2D 출력에서 중단
                            break
                    
                    if len(features) >= 5:
                        break
                        
                except Exception as e:
                    break
            
            # ResNet 스타일로 feature 재구성
            if len(features) >= 3:
                # 원본 입력 크기 기준으로 올바른 해상도 계산
                input_h, input_w = x.shape[-2:]
                
                # ResNet encoder의 정확한 spatial hierarchy
                target_sizes = [
                    (input_h // 2, input_w // 2),    # 176x608  (1/2)
                    (input_h // 4, input_w // 4),    # 88x304   (1/4) 
                    (input_h // 8, input_w // 8),    # 44x152   (1/8)
                    (input_h // 16, input_w // 16),  # 22x76    (1/16)
                    (input_h // 32, input_w // 32),  # 11x38    (1/32)
                ]
                
                resnet_features = []
                
                # 각 feature를 target size로 조정
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
                
                # 부족하면 마지막 feature를 downsampling해서 보충
                while len(resnet_features) < 5:
                    if resnet_features:
                        last_feat = resnet_features[-1]
                        target_size = target_sizes[len(resnet_features)]
                        
                        # 2x downsampling + 채널 증가
                        downsampled = torch.nn.functional.avg_pool2d(last_feat, 2)
                        
                        # 채널 수 증가 (256 -> 512)
                        if downsampled.shape[1] != self.yolo_channels[len(resnet_features)]:
                            with torch.no_grad():
                                channel_expander = nn.Conv2d(
                                    downsampled.shape[1], 
                                    self.yolo_channels[len(resnet_features)], 
                                    1, bias=False
                                ).to(downsampled.device)
                                downsampled = channel_expander(downsampled)
                        
                        # Size 조정
                        if downsampled.shape[-2:] != target_size:
                            downsampled = torch.nn.functional.interpolate(
                                downsampled, size=target_size, mode='bilinear', align_corners=False
                            )
                        
                        resnet_features.append(downsampled)
                    else:
                        break
                
                features = resnet_features[:5]
            
            else:
                # Fallback: 입력에서 순차적으로 ResNet-style downsampling
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
            
            # 🆕 Head Feature Extractor 적용 (선택적)
            if self.use_head_features and hasattr(self, 'head_feature_extractor'):
                try:
                    head_features = self.head_feature_extractor(features[:5])
                    features = head_features
                    
                except Exception as head_error:
                    print(f"❌ Head Feature Extractor failed: {head_error}")
            return features[:5]
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            # 완전 Fallback
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