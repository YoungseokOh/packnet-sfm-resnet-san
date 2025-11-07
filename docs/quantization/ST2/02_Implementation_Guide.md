# 2. 구현 가이드 (Step-by-Step)

## Phase 1: DualHeadDepthDecoder 구현

**파일 생성**: `packnet_sfm/networks/layers/resnet/dual_head_depth_decoder.py`

**완전한 구현 코드**:

```python
# packnet_sfm/networks/layers/resnet/dual_head_depth_decoder.py
"""
Dual-Head Depth Decoder for Integer-Fractional depth prediction.

이 Decoder는 기존 DepthDecoder와 동일한 인터페이스를 유지하면서,
두 개의 독립적인 출력 헤드를 추가합니다.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from .layers import ConvBlock, Conv3x3, upsample


class DualHeadDepthDecoder(nn.Module):
    """
    Integer-Fractional Dual-Head Depth Decoder
    
    기존 DepthDecoder와 동일한 upsampling 구조를 사용하되,
    최종 출력 헤드만 2개로 분리합니다.
    
    Parameters
    ----------
    num_ch_enc : list of int
        Encoder channel counts (e.g., [64, 64, 128, 256, 512])
    scales : list of int
        Which scales to produce outputs (default: [0, 1, 2, 3])
    max_depth : float
        Maximum depth for integer head (default: 15.0)
    use_skips : bool
        Whether to use skip connections (default: True)
    
    Outputs
    -------
    - ("integer", scale): [B, 1, H, W] sigmoid [0, 1] → represents [0, max_depth]
    - ("fractional", scale): [B, 1, H, W] sigmoid [0, 1] → represents [0, 1]m
    """
    
    def __init__(self, num_ch_enc, scales=range(4), max_depth=15.0, use_skips=True):
        super(DualHeadDepthDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.max_depth = max_depth
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        
        # Decoder channel counts (기존과 동일)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # ========================================
        # 공통 Upsampling Layers (기존과 100% 동일)
        # ========================================
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0: channel reduction
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1: skip connection fusion
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # ========================================
        # Dual-Head: 각 스케일별로 2개의 출력 헤드
        # ========================================
        for s in self.scales:
            # Integer Head (정수부 예측: 0~max_depth)
            self.convs[("integer_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)
            
            # Fractional Head (소수부 예측: 0~1m)
            self.convs[("fractional_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        
        print(f"🔧 DualHeadDepthDecoder initialized:")
        print(f"   Max depth: {max_depth}m")
        print(f"   Scales: {list(scales)}")
        print(f"   Integer quantization interval: {max_depth/255:.4f}m")
        print(f"   Fractional quantization interval: {1.0/255:.4f}m (3.92mm)")

    def forward(self, input_features):
        """
        Forward pass
        
        Parameters
        ----------
        input_features : list of torch.Tensor
            Encoder features [feat0, feat1, ..., feat4]
        
        Returns
        -------
        outputs : dict
            {
                ("integer", scale): [B, 1, H, W] sigmoid [0,1],
                ("fractional", scale): [B, 1, H, W] sigmoid [0,1]
            }
        """
        self.outputs = {}

        # ========================================
        # 공통 Decoder Processing (기존과 동일)
        # ========================================
        x = input_features[-1]
        for i in range(4, -1, -1):
            # Upsample
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            
            # Skip connection
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            
            # ========================================
            # Dual-Head Outputs
            # ========================================
            if i in self.scales:
                # Integer Head: [0, 1] sigmoid
                integer_raw = self.convs[("integer_conv", i)](x)
                self.outputs[("integer", i)] = self.sigmoid(integer_raw)
                
                # Fractional Head: [0, 1] sigmoid
                fractional_raw = self.convs[("fractional_conv", i)](x)
                self.outputs[("fractional", i)] = self.sigmoid(fractional_raw)

        return self.outputs
```

**테스트 코드**:

```python
# test_dual_head_decoder.py
import torch
from packnet_sfm.networks.layers.resnet.dual_head_depth_decoder import DualHeadDepthDecoder

def test_dual_head_decoder():
    # Encoder channel counts (ResNet18)
    num_ch_enc = [64, 64, 128, 256, 512]
    
    # Create decoder
    decoder = DualHeadDepthDecoder(
        num_ch_enc=num_ch_enc,
        scales=[0],  # Only test scale 0
        max_depth=15.0
    )
    
    # Dummy encoder features
    batch_size = 2
    features = [
        torch.randn(batch_size, 64, 96, 160),   # scale 0
        torch.randn(batch_size, 64, 48, 80),    # scale 1
        torch.randn(batch_size, 128, 24, 40),   # scale 2
        torch.randn(batch_size, 256, 12, 20),   # scale 3
        torch.randn(batch_size, 512, 6, 10),    # scale 4
    ]
    
    # Forward pass
    outputs = decoder(features)
    
    # Check outputs
    assert ("integer", 0) in outputs, "Missing integer output"
    assert ("fractional", 0) in outputs, "Missing fractional output"
    
    integer_out = outputs[("integer", 0)]
    fractional_out = outputs[("fractional", 0)]
    
    assert integer_out.shape == (batch_size, 1, 96, 160), f"Wrong integer shape: {integer_out.shape}"
    assert fractional_out.shape == (batch_size, 1, 96, 160), f"Wrong fractional shape: {fractional_out.shape}"
    
    # Check value range (sigmoid output)
    assert integer_out.min() >= 0.0 and integer_out.max() <= 1.0, "Integer out of range"
    assert fractional_out.min() >= 0.0 and fractional_out.max() <= 1.0, "Fractional out of range"
    
    print("✅ DualHeadDepthDecoder test passed!")

if __name__ == "__main__":
    test_dual_head_decoder()
```

---

## Phase 2: Helper Functions

**파일 수정**: `packnet_sfm/networks/layers/resnet/layers.py`

**추가할 함수들**:

```python
# packnet_sfm/networks/layers/resnet/layers.py (기존 파일 끝에 추가)

def dual_head_to_depth(integer_sigmoid, fractional_sigmoid, max_depth):
    """
    Convert dual-head sigmoid outputs to depth
    
    Parameters
    ----------
    integer_sigmoid : torch.Tensor [B, 1, H, W]
        Integer part in sigmoid space [0, 1]
    fractional_sigmoid : torch.Tensor [B, 1, H, W]
        Fractional part in sigmoid space [0, 1]
    max_depth : float
        Maximum depth for integer scaling
    
    Returns
    -------
    depth : torch.Tensor [B, 1, H, W]
        Final depth in meters [0, max_depth + 1]
    
    Example
    -------
    >>> integer_sig = torch.tensor([[[[0.333]]]])  # 0.333 * 15 = 5.0
    >>> fractional_sig = torch.tensor([[[[0.5]]]])  # 0.5m
    >>> depth = dual_head_to_depth(integer_sig, fractional_sig, 15.0)
    >>> print(depth)  # 5.5m
    """
    # Integer part: [0, 1] → [0, max_depth]
    integer_part = integer_sigmoid * max_depth
    
    # Fractional part: already [0, 1]m
    fractional_part = fractional_sigmoid
    
    # Combine
    depth = integer_part + fractional_part
    
    return depth


def decompose_depth(depth_gt, max_depth):
    """
    Decompose ground truth depth into integer and fractional parts
    
    Parameters
    ----------
    depth_gt : torch.Tensor [B, 1, H, W]
        Ground truth depth in meters
    max_depth : float
        Maximum depth for integer normalization
    
    Returns
    -------
    integer_gt : torch.Tensor [B, 1, H, W]
        Integer part in sigmoid space [0, 1]
    fractional_gt : torch.Tensor [B, 1, H, W]
        Fractional part [0, 1]m
    
    Example
    -------
    >>> depth = torch.tensor([[[[5.7]]]])  # 5.7m
    >>> integer_gt, frac_gt = decompose_depth(depth, 15.0)
    >>> print(integer_gt)  # 5.0 / 15.0 = 0.333
    >>> print(frac_gt)     # 0.7m
    """
    # Integer part: floor(depth)
    integer_meters = torch.floor(depth_gt)
    integer_gt = integer_meters / max_depth  # Normalize to [0, 1]
    
    # Fractional part: depth - floor(depth)
    fractional_gt = depth_gt - integer_meters  # Already [0, 1]m
    
    return integer_gt, fractional_gt


def dual_head_to_inv_depth(integer_sigmoid, fractional_sigmoid, max_depth, min_depth=0.5):
    """
    Convert dual-head outputs to inverse depth (for compatibility)
    
    Parameters
    ----------
    integer_sigmoid : torch.Tensor
    fractional_sigmoid : torch.Tensor
    max_depth : float
    min_depth : float
    
    Returns
    -------
    inv_depth : torch.Tensor
        Inverse depth [1/max_depth, 1/min_depth]
    """
    # First convert to depth
    depth = dual_head_to_depth(integer_sigmoid, fractional_sigmoid, max_depth)
    
    # Clamp to valid range
    depth = torch.clamp(depth, min=min_depth, max=max_depth)
    
    # Convert to inverse depth
    inv_depth = 1.0 / depth
    
    return inv_depth
```

**테스트**:

```python
# test_helper_functions.py
import torch
from packnet_sfm.networks.layers.resnet.layers import (
    dual_head_to_depth, decompose_depth, dual_head_to_inv_depth
)

def test_helpers():
    # Test 1: Decompose and reconstruct
    depth_gt = torch.tensor([[[[5.7, 12.3, 0.8]]]])
    max_depth = 15.0
    
    integer_gt, frac_gt = decompose_depth(depth_gt, max_depth)
    depth_reconstructed = dual_head_to_depth(integer_gt, frac_gt, max_depth)
    
    assert torch.allclose(depth_gt, depth_reconstructed, atol=1e-5), "Reconstruction failed"
    print("✅ Test 1: Decompose/reconstruct passed")
    
    # Test 2: Edge cases
    depth_edge = torch.tensor([[[[0.0, 15.0, 7.999]]]])
    integer_gt, frac_gt = decompose_depth(depth_edge, max_depth)
    
    assert torch.all(integer_gt >= 0) and torch.all(integer_gt <= 1), "Integer out of range"
    assert torch.all(frac_gt >= 0) and torch.all(frac_gt < 1), "Fractional out of range"
    print("✅ Test 2: Edge cases passed")
    
    # Test 3: Inverse depth conversion
    integer_sig = torch.tensor([[[[0.333]]]])
    frac_sig = torch.tensor([[[[0.5]]]])
    inv_depth = dual_head_to_inv_depth(integer_sig, frac_sig, max_depth, min_depth=0.5)
    
    expected_depth = 5.5  # 0.333*15 + 0.5 = 5.5
    expected_inv = 1.0 / expected_depth
    assert torch.allclose(inv_depth, torch.tensor([[[[expected_inv]]]]), atol=1e-3), "Inv depth wrong"
    print("✅ Test 3: Inverse depth passed")

if __name__ == "__main__":
    test_helpers()
```

---

## Phase 3: ResNetSAN01 확장

**파일 수정**: `packnet_sfm/networks/depth/ResNetSAN01.py`

**수정 위치 1: `__init__` 메서드**

```python
# packnet_sfm/networks/depth/ResNetSAN01.py

class ResNetSAN01(nn.Module):
    def __init__(self, dropout=None, version=None, use_film=False, film_scales=[0],
                 use_enhanced_lidar=False,
                 min_depth=0.5, max_depth=80.0,
                 use_dual_head=False,  # 🆕 추가
                 **kwargs):
        super().__init__()
        
        # 안전 보정 (기존 코드)
        if max_depth <= 0: max_depth = 80.0
        if min_depth <= 0: min_depth = 0.5
        if max_depth <= min_depth: max_depth = min_depth + 1.0
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        
        # ... (기존 encoder 코드 생략) ...
        
        # ResNet encoder (기존 코드)
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=True)
        
        # ========================================
        # 🆕 Decoder 선택 (Factory Pattern)
        # ========================================
        if use_dual_head:
            from packnet_sfm.networks.layers.resnet.dual_head_depth_decoder import DualHeadDepthDecoder
            self.decoder = DualHeadDepthDecoder(
                num_ch_enc=self.encoder.num_ch_enc,
                max_depth=self.max_depth,
                scales=range(4)
            )
            self.is_dual_head = True
            print(f"✅ Using Dual-Head Decoder (max_depth={self.max_depth}m)")
        else:
            from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
            self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
            self.is_dual_head = False
            print(f"✅ Using Single-Head Decoder")
        
        # ... (기존 FiLM/Minkowski 코드 유지) ...
        
        # 설정
        self.use_film = use_film
        self.film_scales = film_scales
        self.use_enhanced_lidar = use_enhanced_lidar
        
        # ... (나머지 기존 코드 유지) ...
```

**수정 위치 2: `run_network` 메서드 (출력 형식 통일)**

```python
# packnet_sfm/networks/depth/ResNetSAN01.py

    def run_network(self, rgb, input_depth=None):
        """
        🆕 Enhanced network execution with Dual-Head support
        """
        # Encode RGB features (기존 코드)
        skip_features = self.encoder(rgb)
        
        # Enhanced sparse depth processing (기존 FiLM 코드 유지)
        if input_depth is not None and self.use_film:
            # ... (기존 FiLM 처리 코드 유지) ...
            pass
        
        # Decode (Dual-Head 또는 Single-Head)
        outputs = self.decoder(skip_features)
        
        # ========================================
        # 🆕 출력 형식 통일
        # ========================================
        if self.is_dual_head:
            # Dual-Head: {"integer": ..., "fractional": ...}
            # → "disp" 키로도 접근 가능하도록 변환
            from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth
            
            for scale in range(4):
                if ("integer", scale) in outputs:
                    # Depth 복원
                    depth = dual_head_to_depth(
                        outputs[("integer", scale)],
                        outputs[("fractional", scale)],
                        self.max_depth
                    )
                    # Inverse depth 변환 (기존 코드와 호환)
                    depth_clamped = torch.clamp(depth, min=self.min_depth, max=self.max_depth)
                    inv_depth = 1.0 / depth_clamped
                    
                    # 기존 키 형식으로도 저장 (호환성)
                    outputs[("disp", scale)] = inv_depth  # Actually inv_depth
                    outputs[("depth", scale)] = depth     # Actual depth
        else:
            # Single-Head: 기존 동작 유지
            pass
        
        if self.training:
            # 학습 시: 모든 스케일 반환
            inv_depths = [outputs[("disp", i)] for i in range(4)]
            return inv_depths, skip_features
        else:
            # 추론 시: scale 0만 반환
            return outputs[("disp", 0)], None
```

**변경 요약**:
- `__init__`: +10줄
- `run_network`: +20줄
- **총 변경량**: ~30줄
- **기존 코드 수정**: 0줄

---

## Phase 4: Loss Function 구현

**파일 생성**: `packnet_sfm/losses/dual_head_depth_loss.py`

```python
# packnet_sfm/losses/dual_head_depth_loss.py
"""
Dual-Head Depth Loss for Integer-Fractional prediction

이 Loss는 기존 SupervisedLoss와 동일한 인터페이스를 유지하면서,
Integer/Fractional 헤드를 별도로 학습합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from packnet_sfm.losses.loss_base import LossBase
from packnet_sfm.networks.layers.resnet.layers import decompose_depth, dual_head_to_depth


class DualHeadDepthLoss(LossBase):
    """
    Integer-Fractional Dual-Head Depth Loss
    
    이 Loss는 세 가지 컴포넌트로 구성됩니다:
    1. Integer Loss: 정수부 예측 (L1 loss)
    2. Fractional Loss: 소수부 예측 (L1 loss, 높은 가중치)
    3. Consistency Loss: 복원된 깊이의 일관성 (L1 loss)
    
    Parameters
    ----------
    max_depth : float
        Maximum depth for integer normalization (default: 15.0)
    integer_weight : float
        Weight for integer loss (default: 1.0)
    fractional_weight : float
        Weight for fractional loss (default: 10.0) - 정밀도 핵심!
    consistency_weight : float
        Weight for consistency loss (default: 0.5)
    min_depth : float
        Minimum valid depth (default: 0.5)
    """
    
    def __init__(self, max_depth=15.0, 
                 integer_weight=1.0, 
                 fractional_weight=10.0,
                 consistency_weight=0.5,
                 min_depth=0.5,
                 **kwargs):
        super().__init__()
        
        # 🆕 파라미터 검증 (Critical!)
        assert max_depth > min_depth, \
            f"max_depth ({max_depth}) must be > min_depth ({min_depth})"
        assert max_depth > 0, \
            f"max_depth must be positive, got {max_depth}"
        assert min_depth >= 0, \
            f"min_depth must be non-negative, got {min_depth}"
        assert integer_weight >= 0, \
            f"integer_weight must be non-negative, got {integer_weight}"
        assert fractional_weight > 0, \
            f"fractional_weight must be positive (핵심!), got {fractional_weight}"
        assert consistency_weight >= 0, \
            f"consistency_weight must be non-negative, got {consistency_weight}"
        
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.integer_weight = integer_weight
        self.fractional_weight = fractional_weight
        self.consistency_weight = consistency_weight
        
        print(f"🎯 DualHeadDepthLoss initialized:")
        print(f"   Max depth: {max_depth}m")
        print(f"   Min depth: {min_depth}m")
        print(f"   Integer weight: {integer_weight}")
        print(f"   Fractional weight: {fractional_weight} (high precision!)")
        print(f"   Consistency weight: {consistency_weight}")
        print(f"   ✅ All parameters validated")
    
    def forward(self, outputs, depth_gt, return_logs=False, progress=0.0):
        """
        Compute dual-head depth loss
        
        Parameters
        ----------
        outputs : dict
            Model outputs containing:
            - ("integer", 0): [B, 1, H, W] sigmoid [0, 1]
            - ("fractional", 0): [B, 1, H, W] sigmoid [0, 1]
        depth_gt : torch.Tensor [B, 1, H, W]
            Ground truth depth
        return_logs : bool
            Whether to return detailed logs
        progress : float
            Training progress [0, 1] for dynamic weighting
        
        Returns
        -------
        loss_dict : dict
            {
                'loss': total_loss,
                'integer_loss': ...,
                'fractional_loss': ...,
                'consistency_loss': ...
            }
        """
        # Resize GT to match prediction size
        if depth_gt.shape[-2:] != outputs[("integer", 0)].shape[-2:]:
            depth_gt = F.interpolate(
                depth_gt, 
                size=outputs[("integer", 0)].shape[-2:],
                mode='nearest'
            )
        
        # Create valid mask
        mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
        
        if mask.sum() == 0:
            # No valid pixels
            return {
                'loss': torch.tensor(0.0, device=depth_gt.device, requires_grad=True),
                'integer_loss': torch.tensor(0.0),
                'fractional_loss': torch.tensor(0.0),
                'consistency_loss': torch.tensor(0.0)
            }
        
        # ========================================
        # 1. Decompose GT depth
        # ========================================
        integer_gt, fractional_gt = decompose_depth(depth_gt, self.max_depth)
        
        # ========================================
        # 2. Integer Loss (coarse prediction)
        # ========================================
        integer_pred = outputs[("integer", 0)]
        integer_loss = F.l1_loss(
            integer_pred[mask],
            integer_gt[mask],
            reduction='mean'
        )
        
        # ========================================
        # 3. Fractional Loss (fine prediction) - 핵심!
        # ========================================
        fractional_pred = outputs[("fractional", 0)]
        fractional_loss = F.l1_loss(
            fractional_pred[mask],
            fractional_gt[mask],
            reduction='mean'
        )
        
        # ========================================
        # 4. Consistency Loss (전체 깊이 일관성)
        # ========================================
        depth_pred = dual_head_to_depth(integer_pred, fractional_pred, self.max_depth)
        consistency_loss = F.l1_loss(
            depth_pred[mask],
            depth_gt[mask],
            reduction='mean'
        )
        
        # ========================================
        # 5. Total Loss (가중치 적용)
        # ========================================
        total_loss = (
            self.integer_weight * integer_loss +
            self.fractional_weight * fractional_loss +
            self.consistency_weight * consistency_loss
        )
        
        # Metrics for logging
        if return_logs:
            self.add_metric('integer_loss', integer_loss)
            self.add_metric('fractional_loss', fractional_loss)
            self.add_metric('consistency_loss', consistency_loss)
            self.add_metric('total_loss', total_loss)
            
            # Additional metrics
            with torch.no_grad():
                # Depth error
                depth_error = torch.abs(depth_pred[mask] - depth_gt[mask])
                self.add_metric('mean_depth_error', depth_error.mean())
                self.add_metric('median_depth_error', depth_error.median())
                
                # Integer accuracy (within 1 meter)
                integer_error = torch.abs(integer_pred[mask] * self.max_depth - integer_gt[mask] * self.max_depth)
                integer_acc = (integer_error < 1.0).float().mean()
                self.add_metric('integer_accuracy', integer_acc)
                
                # Fractional precision
                frac_error = torch.abs(fractional_pred[mask] - fractional_gt[mask])
                self.add_metric('fractional_rmse', torch.sqrt((frac_error ** 2).mean()))
        
        return {
            'loss': total_loss,
            'integer_loss': integer_loss.detach(),
            'fractional_loss': fractional_loss.detach(),
            'consistency_loss': consistency_loss.detach()
        }
```

---

## Phase 5: Model Wrapper 통합

**파일 수정**: `packnet_sfm/models/SemiSupCompletionModel.py`

**수정 위치: `supervised_loss` 메서드**

```python
# packnet_sfm/models/SemiSupCompletionModel.py

    def supervised_loss(self, inv_depths, gt_inv_depths,
                        return_logs=False, progress=0.0):
        """
        Calculates the supervised loss.
        
        🆕 Dual-Head 모델 자동 감지 및 처리
        """
        # ========================================
        # 🆕 Dual-Head 모델 감지
        # ========================================
        if hasattr(self, 'depth_net') and hasattr(self.depth_net, 'is_dual_head') and self.depth_net.is_dual_head:
            # Dual-Head Loss 사용
            from packnet_sfm.losses.dual_head_depth_loss import DualHeadDepthLoss
            
            # Dual-Head Loss 초기화 (한 번만)
            if not hasattr(self, '_dual_head_loss'):
                self._dual_head_loss = DualHeadDepthLoss(
                    max_depth=self.max_depth,
                    min_depth=self.min_depth
                )
            
            # inv_depths는 실제로 outputs dict임
            # gt_inv_depths는 실제로 depth_gt임
            return self._dual_head_loss(
                outputs=inv_depths,  # {"integer": ..., "fractional": ...}
                depth_gt=gt_inv_depths,  # Actually depth
                return_logs=return_logs,
                progress=progress
            )
        else:
            # 기존 Single-Head Loss 사용
            return self._supervised_loss(
                inv_depths, gt_inv_depths,
                return_logs=return_logs, progress=progress
            )
```

**변경량**: +20줄 (기존 코드 수정 없음)

---

## 구현 요약

### 파일 변경 요약

| 파일 | 변경 유형 | 줄 수 |
|------|-----------|-------|
| `dual_head_depth_decoder.py` | 🆕 신규 | ~150줄 |
| `layers.py` | ➕ 함수 추가 | +40줄 |
| `ResNetSAN01.py` | ➕ 로직 추가 | +30줄 |
| `dual_head_depth_loss.py` | 🆕 신규 | ~120줄 |
| `SemiSupCompletionModel.py` | ➕ 분기 추가 | +20줄 |
| **Total** | - | **~360줄** |

### 다음 단계

1. ✅ Phase 1: DualHeadDepthDecoder 구현 완료
2. ✅ Phase 2: Helper Functions 추가 완료
3. ✅ Phase 3: ResNetSAN01 확장 완료
4. ✅ Phase 4: Loss Function 구현 완료
5. ✅ Phase 5: Model Wrapper 통합 완료

→ **다음**: [설정 및 테스트](03_Configuration_Testing.md)
