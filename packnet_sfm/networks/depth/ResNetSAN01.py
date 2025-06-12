import torch
import torch.nn as nn

from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.enhanced_minkowski_encoder import EnhancedMinkowskiEncoder  # 🆕
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth
from functools import partial


class ResNetSAN01(nn.Module):
    """
    🆕 Enhanced ResNet-based SAN network with improved LiDAR feature extraction
    
    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Version string (format: {num_layers}{variant}, e.g., '18A', '34A', '50B')
    use_film : bool
        Whether to use Depth-aware FiLM modulation
    film_scales : list of int
        Which scales to apply FiLM (default: [0] - first scale only)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, use_film=True, film_scales=[0], **kwargs):
        super().__init__()
        
        # 🆕 기존 파라미터만 사용
        # use_enhanced_lidar는 제거하고 기본적으로 True로 설정
        use_enhanced_lidar = kwargs.get('use_enhanced_lidar', True)
        
        # Parse version string
        if version:
            num_layers = int(''.join(filter(str.isdigit, version)))
            self.variant = ''.join(filter(str.isalpha, version)) or 'A'
        else:
            num_layers = 18
            self.variant = 'A'
        
        print(f"🏗️  Initializing ResNetSAN01 with ResNet-{num_layers} (variant {self.variant})")
        
        # ResNet encoder
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=True)
        
        # Standard depth decoder
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        
        # 설정
        self.use_film = use_film
        self.film_scales = film_scales
        self.use_enhanced_lidar = use_enhanced_lidar
        
        # FiLM configuration
        rgb_channels_per_scale = None
        if use_film:
            encoder_channels = self.encoder.num_ch_enc
            rgb_channels_per_scale = []
            
            for i in range(len(encoder_channels)):
                if i in film_scales:
                    rgb_channels_per_scale.append(encoder_channels[i])
                else:
                    rgb_channels_per_scale.append(0)
    
        # Minkowski encoder 선택
        if use_enhanced_lidar:
            # Enhanced 버전이 없다면 일단 기존 버전 사용
            try:
                from packnet_sfm.networks.layers.enhanced_minkowski_encoder import EnhancedMinkowskiEncoder
                self.mconvs = EnhancedMinkowskiEncoder(
                    self.encoder.num_ch_enc, 
                    rgb_channels=rgb_channels_per_scale,
                    with_uncertainty=False,
                    use_geometry_processor=True
                )
                print("🎯 Enhanced LiDAR processing enabled")
            except ImportError:
                # Enhanced 버전이 없으면 기존 버전 사용
                print("⚠️ Enhanced LiDAR not available, using standard version")
                from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
                self.mconvs = MinkowskiEncoder(
                    self.encoder.num_ch_enc, 
                    rgb_channels=rgb_channels_per_scale,
                    with_uncertainty=False
                )
                print("📊 Standard LiDAR processing")
        else:
            # 기존 MinkowskiEncoder 사용
            from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
            self.mconvs = MinkowskiEncoder(
                self.encoder.num_ch_enc, 
                rgb_channels=rgb_channels_per_scale,
                with_uncertainty=False
            )
            print("📊 Standard LiDAR processing")
        
        # 🆕 Learnable fusion weights with improved initialization
        self.weight = torch.nn.parameter.Parameter(
            torch.ones(5) * 0.5, requires_grad=True  # Start with balanced fusion
        )
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(5), requires_grad=True
        )
        
        # 🆕 Feature refinement layers
        if use_enhanced_lidar:
            self.feature_refinement = nn.ModuleDict()
            for i, ch in enumerate(self.encoder.num_ch_enc):
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
        
        self.init_weights()

    def init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def run_network(self, rgb, input_depth=None):
        """
        🆕 Enhanced network execution with improved LiDAR processing
        """
        # Encode RGB features
        skip_features = self.encoder(rgb)
        
        # Enhanced sparse depth processing
        if input_depth is not None:
            self.mconvs.prep(input_depth)
            
            fused_features = []
            for i, feat in enumerate(skip_features):
                # 🆕 Enhanced FiLM application
                if self.use_film and i in self.film_scales:
                    result = self.mconvs(feat)
                    
                    if isinstance(result, tuple) and len(result) == 3:
                        sparse_feat, gamma, beta = result
                        
                        # 🆕 Improved FiLM with feature refinement
                        if self.use_enhanced_lidar and str(i) in self.feature_refinement:
                            attention_map = self.feature_refinement[str(i)](feat)
                            refined_feat = feat * attention_map
                        else:
                            refined_feat = feat
                        
                        # Enhanced FiLM application
                        modulated_feat = gamma * refined_feat + beta
                        
                        # 🆕 Adaptive fusion based on feature importance
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
        
        # Decode to get inverse depth maps
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
        🆕 Enhanced forward pass with improved LiDAR integration
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
        
        # 🆕 Enhanced consistency loss with feature-level weighting
        feature_weights = torch.softmax(torch.abs(self.weight), dim=0)
        weighted_loss = sum([
            weight * ((feat_rgbd.detach() - feat_rgb) ** 2).mean()
            for weight, feat_rgbd, feat_rgb in zip(feature_weights, skip_feat_rgbd, skip_feat_rgb)
        ]) / len(skip_feat_rgbd)
        
        output['depth_loss'] = weighted_loss
        
        return output