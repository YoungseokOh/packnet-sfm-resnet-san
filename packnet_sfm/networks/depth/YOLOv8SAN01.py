import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth


class YOLOv8SAN01(nn.Module):
    """
    🆕 YOLOv8-based SAN network with ResNet DepthDecoder compatibility
    
    Parameters
    ----------
    variant : str
        YOLOv8 variant ('n', 's', 'm', 'l', 'x')
    use_film : bool
        Whether to use Depth-aware FiLM modulation
    film_scales : list of int
        Which scales to apply FiLM (default: [0] - first scale only)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, variant='s', use_film=True, film_scales=[0], **kwargs):
        super().__init__()
        
        # 🆕 Enhanced LiDAR processing 설정
        use_enhanced_lidar = kwargs.get('use_enhanced_lidar', True)
        
        print(f"🏗️  Initializing YOLOv8SAN01 with YOLOv8{variant}")
        
        # YOLOv8 backbone 로드
        yolo_model = YOLO(f'yolov8{variant}.pt')
        self.backbone = yolo_model.model.model
        
        # 🔍 ResNet DepthDecoder 호환을 위한 정확한 채널 구조
        resnet_channels = [64, 64, 128, 256, 512]  # ResNet-18 표준
        
        # YOLOv8 feature extraction과 adaptation 설정
        self._setup_feature_extraction()
        
        # 🆕 실제 YOLOv8 채널 확인 후 adapter 설정 (일치시킴)
        self.yolo_channels = self._probe_yolo_channels()
        
        self._setup_feature_adapters(self.yolo_channels, resnet_channels)
        
        # 최종 채널 수 (ResNet 호환)
        self.num_ch_enc = resnet_channels
        
        # ResNet DepthDecoder 사용
        self.decoder = DepthDecoder(num_ch_enc=self.num_ch_enc)
        
        # SAN 설정
        self.use_film = use_film
        self.film_scales = film_scales
        self.use_enhanced_lidar = use_enhanced_lidar
        
        # FiLM configuration
        rgb_channels_per_scale = None
        if use_film:
            rgb_channels_per_scale = []
            for i in range(len(self.num_ch_enc)):
                if i in film_scales:
                    rgb_channels_per_scale.append(self.num_ch_enc[i])
                else:
                    rgb_channels_per_scale.append(0)
        
        # Minkowski encoder 설정
        self._setup_minkowski_encoder(rgb_channels_per_scale, use_enhanced_lidar)
        
        # 🆕 Learnable fusion weights
        self.weight = torch.nn.parameter.Parameter(
            torch.ones(5) * 0.5, requires_grad=True
        )
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(5), requires_grad=True
        )
        
        # 🆕 Feature refinement layers
        if use_enhanced_lidar:
            self.feature_refinement = nn.ModuleDict()
            for i, ch in enumerate(self.num_ch_enc):
                self.feature_refinement[str(i)] = nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, ch, kernel_size=1),
                    nn.Sigmoid()
                )
        
        print(f"🎯 FiLM enabled: {use_film}")
        if use_film:
            print(f"   FiLM scales: {film_scales}")
            print(f"   RGB channels per scale: {rgb_channels_per_scale}")
        print(f"🔧 Final encoder channels: {self.num_ch_enc}")
        
        self.init_weights()

    def _setup_feature_extraction(self):
        """YOLOv8 feature extraction points 설정"""
        # 🔧 ResNet과 정확히 맞추기 위해 5개만 추출
        self.feature_extraction_points = [1, 2, 4, 6, 9]  # 🔧 0 제거, 5개만 추출

    def _probe_yolo_channels(self):
        """YOLOv8 backbone의 실제 채널 수 탐지"""
        dummy_input = torch.randn(1, 3, 64, 64)
        
        channels = []
        x = dummy_input
        
        print("🔍 Probing YOLOv8 channels:")
        with torch.no_grad():
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                
                if i in self.feature_extraction_points:
                    channels.append(x.shape[1])
                    if len(channels) >= 5:  # 🔧 정확히 5개만
                        break
        
        # 🔧 더 이상 슬라이싱 필요 없음 - 정확히 5개 추출
        # 정확히 5개가 되도록 조정
        while len(channels) < 5:
            channels.append(channels[-1] if channels else 64)
        
        return channels[:5]

    def _setup_feature_adapters(self, yolo_channels, target_channels):
        """Feature adapter 설정 - 정확한 채널 매핑"""
        
        # 채널 변환을 위한 adapter 생성
        self.feature_adapters = nn.ModuleList()
        for i, (yolo_ch, resnet_ch) in enumerate(zip(yolo_channels, target_channels)):
            adapter = nn.Sequential(
                nn.Conv2d(yolo_ch, resnet_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(resnet_ch),
                nn.ReLU(inplace=True)
            )
            self.feature_adapters.append(adapter)
            print(f"   Adapter {i}: {yolo_ch} -> {resnet_ch}")

    def _setup_minkowski_encoder(self, rgb_channels_per_scale, use_enhanced_lidar):
        """Minkowski encoder 설정"""
        if use_enhanced_lidar:
            try:
                from packnet_sfm.networks.layers.enhanced_minkowski_encoder import EnhancedMinkowskiEncoder
                self.mconvs = EnhancedMinkowskiEncoder(
                    self.num_ch_enc,
                    rgb_channels=rgb_channels_per_scale,
                    with_uncertainty=False,
                    use_geometry_processor=True
                )
                print("🎯 Enhanced LiDAR processing enabled")
            except ImportError:
                print("⚠️ Enhanced LiDAR not available, using standard version")
                from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
                self.mconvs = MinkowskiEncoder(
                    self.num_ch_enc,
                    rgb_channels=rgb_channels_per_scale,
                    with_uncertainty=False
                )
                print("📊 Standard LiDAR processing")
        else:
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
        """YOLOv8에서 ResNet 호환 feature pyramid 추출"""
        features = []
        original_size = x.shape[-2:]  # (H, W)
        
        # 🔧 YOLOv8 backbone을 통과하면서 feature 추출 (5개만!)
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            
            if i in self.feature_extraction_points:
                features.append(x)
                if len(features) >= 5:  # 🔧 5개만 추출
                    break
        
        # 🔧 정확히 5개 확보
        while len(features) < 5:
            features.append(features[-1])
        features = features[:5]
        
        # 🆕 ResNet과 정확히 동일한 해상도 피라미드 생성
        adapted_features = []
        target_sizes = self._calculate_resnet_target_sizes(original_size)
        
        # 🔧 모든 feature를 동일하게 처리 (첫 번째 특별 처리 제거)
        for i, (feat, adapter, target_size) in enumerate(zip(features, self.feature_adapters, target_sizes)):
            
            # 목표 해상도로 리사이즈
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            # 채널 변환
            adapted_feat = adapter(feat)
            adapted_features.append(adapted_feat)
        
        return adapted_features

    def _calculate_resnet_target_sizes(self, original_size):
        """ResNet DepthDecoder 호환 target sizes 계산"""
        H, W = original_size
        
        # ResNet encoder의 정확한 해상도 피라미드
        # 🔧 실제 ResNet-18과 동일한 해상도 구조
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
        🆕 Enhanced network execution with ResNet compatibility
        """
        # YOLOv8에서 ResNet 호환 feature pyramid 추출
        skip_features = self.extract_features(rgb)
        
        # Enhanced sparse depth processing (ResNetSAN01과 동일)
        if input_depth is not None:
            self.mconvs.prep(input_depth)
            
            fused_features = []
            for i, feat in enumerate(skip_features):
                # Enhanced FiLM application
                if self.use_film and i in self.film_scales:
                    result = self.mconvs(feat)
                    
                    if isinstance(result, tuple) and len(result) == 3:
                        sparse_feat, gamma, beta = result
                        
                        # Improved FiLM with feature refinement
                        if self.use_enhanced_lidar and str(i) in self.feature_refinement:
                            attention_map = self.feature_refinement[str(i)](feat)
                            refined_feat = feat * attention_map
                        else:
                            refined_feat = feat
                        
                        # Enhanced FiLM application
                        modulated_feat = gamma * refined_feat + beta
                        
                        # Adaptive fusion based on feature importance
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
                    # Standard fusion
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

    def forward(self, rgb, input_depth=None, **kwargs):
        """
        🆕 Enhanced forward pass with ResNet DepthDecoder compatibility
        """
        if not self.training:
            inv_depths, _ = self.run_network(rgb, input_depth)
            return {'inv_depths': inv_depths}

        output = {}
        
        # RGB-only forward pass
        inv_depths_rgb, skip_feat_rgb = self.run_network(rgb)
        output['inv_depths'] = inv_depths_rgb
        
        if input_depth is None:
            return {'inv_depths': inv_depths_rgb}
        
        # RGB+D forward pass with enhanced processing
        inv_depths_rgbd, skip_feat_rgbd = self.run_network(rgb, input_depth)
        output['inv_depths_rgbd'] = inv_depths_rgbd
        
        # Enhanced consistency loss with feature-level weighting
        feature_weights = torch.softmax(torch.abs(self.weight), dim=0)
        weighted_loss = sum([
            weight * ((feat_rgbd.detach() - feat_rgb) ** 2).mean()
            for weight, feat_rgbd, feat_rgb in zip(feature_weights, skip_feat_rgbd, skip_feat_rgb)
        ]) / len(skip_feat_rgbd)
        
        output['depth_loss'] = weighted_loss
        
        return output