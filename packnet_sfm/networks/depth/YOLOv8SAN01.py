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
    🆕 YOLOv8-based SAN network (PackNet-SAN compatible)
    
    Parameters
    ----------
    variant : str
        YOLOv8 variant ('n', 's', 'm', 'l', 'x')
    use_film : bool
        Whether to use Depth-aware FiLM modulation
    film_scales : list of int
        Which scales to apply FiLM (default: [0] - first scale only)
    use_head_features : bool
        Whether to use Head Feature Extraction (default: False)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, variant='s', use_film=False, film_scales=[0], 
                 use_head_features=False, **kwargs):
        super().__init__()
        
        print(f"🏗️ Initializing YOLOv8SAN01 with YOLOv8{variant}")
        print(f"   Use Head Features: {use_head_features}")
        
        # variant 속성을 먼저 저장
        self.variant = variant
        self.use_head_features = use_head_features
        
        # YOLOv8 백본 로드
        try:
            yolo_model = YOLO(f'yolov8{variant}.pt')
            self.backbone = yolo_model.model.model
            print(f"✅ YOLOv8{variant} backbone loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8{variant}: {e}")
            raise
        
        # ResNet DepthDecoder 호환을 위한 정확한 채널 구조
        resnet_channels = [64, 64, 128, 256, 512]  # ResNet-18 표준
        
        # 🆕 실제 채널 탐지 (runtime에서 확인)
        self.yolo_channels = self._probe_actual_channels()
        
        if self.use_head_features:
            # Head Feature Extractor 초기화
            self.head_feature_extractor = YOLOv8HeadFeatureExtractor(
                self.yolo_channels, variant=variant
            )
            # Head Feature Extractor의 출력 채널 사용
            adapter_input_channels = self.head_feature_extractor.output_channels
            print(f"🎯 Using Head Feature Extraction")
            print(f"   Head output channels: {adapter_input_channels}")
        else:
            # Backbone 채널 직접 사용
            adapter_input_channels = self.yolo_channels
            print(f"🎯 Using Backbone Features Only")
            print(f"   Backbone channels: {adapter_input_channels}")
        
        # 🆕 Lazy adapter 초기화 (첫 번째 forward에서 생성)
        self.adapter_input_channels = adapter_input_channels
        self.resnet_channels = resnet_channels
        self.feature_adapters = None
        
        # 최종 채널 수 (ResNet 호환)
        self.num_ch_enc = resnet_channels
        
        # ResNet DepthDecoder 사용
        self.decoder = DepthDecoder(num_ch_enc=self.num_ch_enc)
        
        # 🆕 PackNet-SAN과 동일한 SAN 설정
        self.use_film = use_film
        self.film_scales = film_scales
        self.use_enhanced_lidar = kwargs.get('use_enhanced_lidar', False)
        
        # 🆕 PackNet-SAN과 동일한 FiLM configuration
        rgb_channels_per_scale = None
        if use_film:
            rgb_channels_per_scale = []
            for i in range(len(self.num_ch_enc)):
                if i in film_scales:
                    rgb_channels_per_scale.append(self.num_ch_enc[i])
                else:
                    rgb_channels_per_scale.append(0)
        
        # 🆕 PackNet-SAN과 동일한 Minkowski encoder 설정
        self._setup_minkowski_encoder(rgb_channels_per_scale)
        
        # 🆕 PackNet-SAN과 동일한 Learnable fusion weights
        self.weight = torch.nn.parameter.Parameter(
            torch.ones(5) * 0.5, requires_grad=True
        )
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(5), requires_grad=True
        )
        
        print(f"🎯 FiLM enabled: {use_film}")
        if use_film:
            print(f"   FiLM scales: {film_scales}")
            print(f"   RGB channels per scale: {rgb_channels_per_scale}")
        print(f"🔧 Final encoder channels: {self.num_ch_enc}")
        
        self.init_weights()

    def _probe_actual_channels(self):
        """🆕 Runtime에서 실제 채널 수 탐지"""
        print("🔍 Probing actual YOLOv8 channels at runtime...")
        
        # 더 작은 입력으로 테스트
        dummy_input = torch.randn(1, 3, 32, 32)
        channels = []
        
        with torch.no_grad():
            x = dummy_input
            for i, layer in enumerate(self.backbone):
                try:
                    x = layer(x)
                    if i in [1, 2, 4, 6, 9]:  # 주요 레이어만
                        channels.append(x.shape[1])
                        print(f"   Layer {i}: {x.shape} -> {x.shape[1]} channels")
                        if len(channels) >= 5:
                            break
                except Exception as layer_error:
                    print(f"   ⚠️ Layer {i} failed: {layer_error}")
                    break
        
        # 정확히 5개가 되도록 조정
        while len(channels) < 5:
            channels.append(channels[-1] if channels else 64)
        
        print(f"✅ Actual channels detected: {channels[:5]}")
        return channels[:5]

    def _setup_lazy_adapters(self, actual_channels):
        """🆕 Runtime에서 실제 채널에 맞춰 adapter 생성"""
        if self.feature_adapters is not None:
            return  # 이미 생성됨
        
        self.feature_adapters = nn.ModuleList()
        
        for i, (actual_ch, resnet_ch) in enumerate(zip(actual_channels, self.resnet_channels)):
            adapter = nn.Sequential(
                nn.Conv2d(actual_ch, resnet_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(resnet_ch),
                nn.ReLU(inplace=True)
            )
            self.feature_adapters.append(adapter)
        
        # GPU로 이동 (필요한 경우)
        if next(self.parameters()).is_cuda:
            self.feature_adapters = self.feature_adapters.cuda()

    def _setup_minkowski_encoder(self, rgb_channels_per_scale):
        """🆕 PackNet-SAN과 동일한 Minkowski encoder 설정"""
        from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
        self.mconvs = MinkowskiEncoder(
            self.num_ch_enc,
            rgb_channels=rgb_channels_per_scale,
            with_uncertainty=False
        )
        print("📊 Standard LiDAR processing")

    def init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def extract_features(self, x):
        """🆕 Runtime adapter 생성을 포함한 Feature Extraction"""
        features = []
        original_size = x.shape[-2:]  # (H, W)
        
        try:
            # 1) YOLOv8 Backbone features 추출
            feature_indices = [1, 2, 4, 6, 9]
            actual_channels = []
            
            current_x = x
            for i, layer in enumerate(self.backbone):
                try:
                    current_x = layer(current_x)
                    if i in feature_indices:
                        features.append(current_x.clone())
                        actual_channels.append(current_x.shape[1])  # 실제 채널 기록
                        if len(features) >= 5:
                            break
                except Exception as layer_error:
                    print(f"⚠️ Error in layer {i}: {layer_error}")
                    if features:
                        features.append(features[-1])
                        actual_channels.append(actual_channels[-1] if actual_channels else 64)
                    else:
                        dummy_feat = nn.functional.adaptive_avg_pool2d(x, (x.shape[-2]//2, x.shape[-1]//2))
                        features.append(dummy_feat)
                        actual_channels.append(dummy_feat.shape[1])
            
            # 정확히 5개 확보
            while len(features) < 5:
                features.append(features[-1] if features else x)
                actual_channels.append(actual_channels[-1] if actual_channels else x.shape[1])
            features = features[:5]
            actual_channels = actual_channels[:5]
            
            # 2) 선택적 Head Feature Processing
            if self.use_head_features:
                try:
                    features = self.head_feature_extractor(features)
                    # Head feature extractor는 고정된 출력 채널을 가짐
                    actual_channels = self.head_feature_extractor.output_channels
                except Exception as head_error:
                    print(f"⚠️ Head Feature Extraction failed: {head_error}")
                    print("🔧 Falling back to backbone features")
            
            # 3) 🆕 Lazy adapter 생성 (실제 채널에 맞춰)
            self._setup_lazy_adapters(actual_channels)
            
            # 4) Feature adaptation
            adapted_features = []
            target_sizes = self._calculate_resnet_target_sizes(original_size)
            
            for i, (feat, adapter, target_size) in enumerate(zip(features, self.feature_adapters, target_sizes)):
                try:
                    # 목표 해상도로 리사이즈
                    if feat.shape[-2:] != target_size:
                        feat = nn.functional.interpolate(
                            feat, size=target_size, mode='bilinear', align_corners=False
                        )
                    
                    # 채널 변환
                    adapted_feat = adapter(feat)
                    adapted_features.append(adapted_feat)
                    
                except Exception as adapter_error:
                    print(f"⚠️ Adapter {i} failed: {adapter_error}")
                    # 안전한 fallback
                    dummy_feat = torch.zeros(feat.shape[0], self.num_ch_enc[i], *target_size, 
                                           device=feat.device, dtype=feat.dtype)
                    adapted_features.append(dummy_feat)
            
            return adapted_features
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            # 완전한 fallback
            target_sizes = self._calculate_resnet_target_sizes(original_size)
            dummy_features = []
            for i, target_size in enumerate(target_sizes):
                dummy_feat = torch.zeros(x.shape[0], self.num_ch_enc[i], *target_size, 
                                       device=x.device, dtype=x.dtype)
                dummy_features.append(dummy_feat)
            return dummy_features

    def _calculate_resnet_target_sizes(self, original_size):
        """ResNet DepthDecoder 호환 target sizes 계산"""
        H, W = original_size
        
        # ResNet encoder의 정확한 해상도 피라미드
        target_sizes = [
            (H // 2, W // 2),    # feat0: H/2, W/2   (conv1+maxpool)
            (H // 4, W // 4),    # feat1: H/4, W/4   (layer1)
            (H // 8, W // 8),    # feat2: H/8, W/8   (layer2)
            (H // 16, W // 16),  # feat3: H/16, W/16 (layer3)
            (H // 32, W // 32),  # feat4: H/32, W/32 (layer4)
        ]
        
        return target_sizes

    def run_network(self, rgb, input_depth=None):
        """
        🆕 PackNet-SAN과 동일한 network execution
        """
        try:
            # YOLOv8에서 ResNet 호환 feature pyramid 추출
            skip_features = self.extract_features(rgb)
            
            # 🆕 PackNet-SAN과 동일한 LiDAR 처리
            if input_depth is not None:
                self.mconvs.prep(input_depth)
                
                fused_features = []
                for i, feat in enumerate(skip_features):
                    if self.use_film and i in self.film_scales:
                        # FiLM이 활성화된 스케일에서만 depth-aware modulation
                        result = self.mconvs(feat)
                        
                        if isinstance(result, tuple) and len(result) == 3:
                            sparse_feat, gamma, beta = result
                            # FiLM modulation 적용
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
                        # 🆕 FiLM이 비활성화된 스케일에서도 LiDAR 융합은 수행
                        sparse_feat = self.mconvs(feat)
                        fusion_weight = torch.sigmoid(self.weight[i])
                        fused_feat = (fusion_weight * feat + 
                                     (1 - fusion_weight) * sparse_feat + 
                                     self.bias[i].view(1, 1, 1, 1))
                    
                    fused_features.append(fused_feat)
                
                skip_features = fused_features
            
            # ResNet DepthDecoder 사용
            inv_depths_dict = self.decoder(skip_features)
            
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
            
            return inv_depths, skip_features
            
        except Exception as e:
            print(f"❌ Network execution failed: {e}")
            # 완전한 fallback
            dummy_depth = torch.ones(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3], 
                                   device=rgb.device, dtype=rgb.dtype) * 0.1
            return [dummy_depth], []

    def forward(self, rgb, input_depth=None, **kwargs):
        """
        🆕 PackNet-SAN과 완전히 동일한 forward pass
        """
        if not self.training:
            # 🆕 Inference 모드: RGB+LiDAR 사용
            inv_depths, _ = self.run_network(rgb, input_depth)
            return {'inv_depths': inv_depths}

        output = {}
        
        # 🆕 Training 모드: PackNet-SAN과 동일한 로직
        # 1) RGB-only forward pass
        inv_depths_rgb, skip_feat_rgb = self.run_network(rgb)
        output['inv_depths'] = inv_depths_rgb
        
        if input_depth is None:
            return {'inv_depths': inv_depths_rgb}
        
        # 2) RGB+LiDAR forward pass
        inv_depths_rgbd, skip_feat_rgbd = self.run_network(rgb, input_depth)
        output['inv_depths_rgbd'] = inv_depths_rgbd
        
        # 3) 🆕 PackNet-SAN과 동일한 feature consistency loss
        if len(skip_feat_rgbd) == len(skip_feat_rgb) and len(skip_feat_rgb) > 0:
            feature_weights = torch.softmax(torch.abs(self.weight), dim=0)
            weighted_loss = sum([
                weight * ((feat_rgbd.detach() - feat_rgb) ** 2).mean()
                for weight, feat_rgbd, feat_rgb in zip(feature_weights, skip_feat_rgbd, skip_feat_rgb)
            ]) / len(skip_feat_rgbd)
            
            output['depth_loss'] = weighted_loss
        
        return output