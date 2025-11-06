import os
import torch
import torch.nn as nn
from torchvision.utils import save_image  # ← 추가
import json  # ← 추가

from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.enhanced_minkowski_encoder import EnhancedMinkowskiEncoder
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
    def __init__(self, dropout=None, version=None, use_film=False, film_scales=[0],
                 use_enhanced_lidar=False,
                 min_depth=0.5, max_depth=80.0,
                 depth_output_mode='sigmoid',  # 'sigmoid' (default) or 'direct'
                 **kwargs):
        super().__init__()
        
        # 안전 보정
        if max_depth <= 0: max_depth = 80.0
        if min_depth <= 0: min_depth = 0.5
        if max_depth <= min_depth: max_depth = min_depth + 1.0
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.depth_output_mode = depth_output_mode
        
        # 🆕 기존 파라미터만 사용
        use_enhanced_lidar = kwargs.get('use_enhanced_lidar', False)  # 기본값 False로 변경
        
        # Parse version string
        if version:
            num_layers = int(version[:2])
            self.variant = version[2:] if len(version) > 2 else 'A'
        else:
            num_layers = 18
            self.variant = 'A'
        
        print(f"🏗️ Initializing ResNetSAN01 with ResNet-{num_layers} (variant {self.variant})")
        print(f"🎯 Depth range: [{self.min_depth}, {self.max_depth}]m")
        print(f"🎯 Depth output mode: {self.depth_output_mode}")
        if self.depth_output_mode == 'direct':
            print(f"   → Direct Linear Depth (INT8 friendly: ±{(self.max_depth - self.min_depth) / 255 / 2 * 1000:.1f}mm)")
        else:
            print(f"   → Sigmoid → Bounded Inverse (legacy)")
        
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
            rgb_channels_per_scale = []
            for i in range(len(self.encoder.num_ch_enc)):
                if i in film_scales:
                    rgb_channels_per_scale.append(self.encoder.num_ch_enc[i])
                else:
                    rgb_channels_per_scale.append(0)

        # 🔧 Minkowski encoder 선택 (조건부)
        # use_film=False이면 Minkowski encoder 불필요 (추론 전용)
        self.mconvs = None
        if use_film:
            if use_enhanced_lidar:
                print("🔧 Using EnhancedMinkowskiEncoder")
                from packnet_sfm.networks.layers.enhanced_minkowski_encoder import EnhancedMinkowskiEncoder
                self.mconvs = EnhancedMinkowskiEncoder(
                    self.encoder.num_ch_enc,
                    rgb_channels=rgb_channels_per_scale,
                    with_uncertainty=False
                )
                
                # Feature refinement layers (Enhanced용)
                self.feature_refinement = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ch, ch, 3, padding=1, bias=False)
                    ) for ch in self.encoder.num_ch_enc
                ])
            else:
                print("🔧 Using standard MinkowskiEncoder")
                from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
                self.mconvs = MinkowskiEncoder(
                    self.encoder.num_ch_enc,
                    rgb_channels=rgb_channels_per_scale,
                    with_uncertainty=False
                )
        else:
            print("🔧 Minkowski encoder disabled (inference-only mode)")

        
        # Learnable fusion weights
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
        
        self.init_weights()
        
        self._disp_stats_done = False  # ✅ DISP_STATS_ONCE 제어 플래그

    def init_weights(self):
        """Initialize only newly created layers; keep pretrained encoder intact."""
        # Skip encoder (pretrained)
        for name, m in self.named_modules():
            if name.startswith('encoder'):
                continue
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _maybe_log_disp_stats(self, outputs):
        """
        ENV:
          DISP_STATS_ONCE=1  -> 한 번만 disparity 통계 출력
          DISP_STATS_EVERY=1 -> 매 step 출력
          DISP_STATS_DIR     -> 저장 폴더 (기본: disp_stats)
        """
        every = os.environ.get("DISP_STATS_EVERY", "0") == "1"
        once  = os.environ.get("DISP_STATS_ONCE", "0") == "1"
        if not (every or once):
            return
        if once and self._disp_stats_done:
            return

        # scale 0 기준 (없으면 리턴)
        key = ("disp", 0)
        if key not in outputs:
            return
        disp = outputs[key].detach()
        v = disp[disp.isfinite()]
        if v.numel() == 0:
            print("[DISP_STATS] no finite values")
            return
        q = torch.quantile(v, torch.tensor([0.0,0.01,0.05,0.5,0.95,0.99,1.0], device=v.device))
        stats = {
            "min": float(q[0]), "p1": float(q[1]), "p5": float(q[2]),
            "median": float(q[3]), "p95": float(q[4]), "p99": float(q[5]),
            "max": float(q[6]),
            "mean": float(v.mean()), "std": float(v.std()),
            "sat>0.99": float((disp > 0.99).float().mean()),
            "sat<0.01": float((disp < 0.01).float().mean()),
        }
        print(f"[DISP_STATS] scale0:", " ".join(f"{k}={stats[k]:.4g}" for k in stats))

        # ===== 저장 (JSON + PNG 한 장) =====
        try:
            if not hasattr(self, "_disp_stats_idx"):
                self._disp_stats_idx = 0
            out_dir = os.environ.get("DISP_STATS_DIR", "disp_stats")
            os.makedirs(out_dir, exist_ok=True)
            json_path = os.path.join(out_dir, f"disp_stats_{self._disp_stats_idx:05d}.json")
            with open(json_path, "w") as f:
                json.dump(stats, f, indent=2)
            # 첫 배치 첫 샘플 저장 (0~1 값 가정)
            png_path = os.path.join(out_dir, f"disp_{self._disp_stats_idx:05d}.png")
            save_image(disp[0:1], png_path)
            # 인덱스 증가 (EVERY 모드 대비)
            self._disp_stats_idx += 1
        except Exception as e:
            print("[DISP_STATS][SAVE_ERROR]", e)
        # ===============================

        if once:
            self._disp_stats_done = True
            # 이후 저장 반복 방지 (EVERY가 아니면 인덱스 고정)

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
        
        # Decode to get outputs
        outputs = self.decoder(skip_features)  # ("disp", i) is sigmoid output [0, 1]

        # 🆕 Convert to depth based on depth_output_mode
        if self.depth_output_mode == 'direct':
            # Direct Linear Depth Output
            depth_outputs = []
            for i in range(4):
                sigmoid = outputs[("disp", i)]
                # Linear transformation: depth = min + (max - min) * sigmoid
                depth = self.min_depth + (self.max_depth - self.min_depth) * sigmoid
                depth_outputs.append(depth)
            
            if not hasattr(self, "_direct_mode_logged"):
                print(f"\n[ResNetSAN01] Direct Depth Output mode")
                print(f"   Range: [{self.min_depth}, {self.max_depth}]m")
                print(f"   INT8 quantization error: ±{(self.max_depth - self.min_depth) / 255 / 2 * 1000:.1f}mm (uniform)")
                self._direct_mode_logged = True
        else:
            # Sigmoid Output (legacy, for Bounded Inverse transformation)
            depth_outputs = []
            for i in range(4):
                sigmoid = outputs[("disp", i)]
                # Bounded Inverse: inv = inv_min + (inv_max - inv_min) * sigmoid
                inv_min = 1.0 / self.max_depth
                inv_max = 1.0 / self.min_depth
                inv_depth = inv_min + (inv_max - inv_min) * sigmoid
                depth = 1.0 / (inv_depth + 1e-8)
                depth_outputs.append(depth)
            
            if not hasattr(self, "_sigmoid_mode_logged"):
                print(f"\n[ResNetSAN01] Sigmoid → Bounded Inverse mode (legacy)")
                print(f"   Range: [{self.min_depth}, {self.max_depth}]m")
                print(f"   Warning: INT8 error @ {self.max_depth}m: ~{(self.max_depth - self.min_depth) * 434 / 255 * 1000:.0f}mm")
                self._sigmoid_mode_logged = True

        if self.training:
            if hasattr(self, "_maybe_log_disp_stats"):
                self._maybe_log_disp_stats(outputs)
            # Training: return 4 scales of depth outputs
            return depth_outputs, skip_features
        else:
            # Inference: return 1 scale of depth output
            return [depth_outputs[0]], skip_features

    def forward(self, rgb, input_depth=None, **kwargs):
        """
        🆕 Enhanced forward pass with improved LiDAR integration
        
        Returns:
            dict with 'inv_depths' key (name kept for backward compatibility)
            - If depth_output_mode='direct': contains direct depth values
            - If depth_output_mode='sigmoid': contains bounded inverse depth values
        """
        if not self.training:
            depths, _ = self.run_network(rgb, input_depth)
            return {'inv_depths': depths}  # Keep key name for compatibility

        output = {}
        
        # RGB-only forward pass
        depths_rgb, skip_feat_rgb = self.run_network(rgb)
        output['inv_depths'] = depths_rgb  # Keep key name for compatibility
        
        if input_depth is None:
            return output
        
        # RGB+D forward pass with enhanced processing
        depths_rgbd, skip_feat_rgbd = self.run_network(rgb, input_depth)
        output['inv_depths_rgbd'] = depths_rgbd  # Keep key name for compatibility
        
        # 🆕 Enhanced consistency loss with feature-level weighting
        feature_weights = torch.softmax(torch.abs(self.weight), dim=0)
        weighted_loss = sum([
            weight * ((feat_rgbd.detach() - feat_rgb) ** 2).mean()
            for weight, feat_rgbd, feat_rgb in zip(feature_weights, skip_feat_rgbd, skip_feat_rgb)
        ]) / len(skip_feat_rgbd)
        
        output['depth_loss'] = weighted_loss
        
        return output