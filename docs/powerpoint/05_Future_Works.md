# 5. Future Works

---

## 5.1 단기 목표 (3개월)

### 5.1.1 Dual-Head INT8 양자화 완료

**현재 상태**: 아키텍처 구현 완료, INT8 양자화 테스트 진행 중

**목표**:
- INT8 Dual-Head 양자화 검증
- abs_rel < 0.055 달성 (FP32 대비 33% 이내)
- NPU 실배포 검증

```
┌─────────────────────────────────────────────────────────────┐
│              Dual-Head INT8 Optimization Roadmap             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Phase 1: PTQ Baseline (현재)                              │
│   ┌────────────────────────────────────────────────────┐   │
│   │  • Dual-Head ONNX export                           │   │
│   │  • NPU PTQ 적용                                    │   │
│   │  • 성능 측정 및 분석                               │   │
│   └────────────────────────────────────────────────────┘   │
│                                                             │
│   Phase 2: QAT (필요 시)                                    │
│   ┌────────────────────────────────────────────────────┐   │
│   │  • Quantization-Aware Training                     │   │
│   │  • Fine-tuning with fake quantization              │   │
│   │  • 추가 10-20% 성능 개선 예상                      │   │
│   └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.1.2 FiLM + Dual-Head 통합

**목표**: LiDAR 정보를 활용한 Dual-Head 학습

```python
# 향후 구성
model:
    depth_net:
        name: 'ResNetSAN01'
        version: '18A'
        use_dual_head: true   # ✓ Dual-Head
        use_film: true        # ✓ FiLM 활성화
```

**예상 효과**:
- Sparse Depth 가이던스로 정확도 향상
- 특히 에지 영역에서 개선 기대

---

## 5.2 중기 목표 (6개월)

### 5.2.1 Multi-Head 확장

**아이디어**: Dual-Head를 넘어 더 세밀한 범위 분할

```
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Head Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [Current: Dual-Head]                                      │
│   ┌───────────────┬───────────────┐                        │
│   │ Integer [0,15]│ Frac [0,1]m   │                        │
│   │ 16 levels     │ 256 levels    │                        │
│   └───────────────┴───────────────┘                        │
│   Precision: ±1.96mm                                        │
│                                                             │
│   [Future: Tri-Head]                                        │
│   ┌───────────┬───────────┬───────────┐                    │
│   │ Coarse    │ Fine      │ Ultra-Fine│                    │
│   │ [0,15]    │ [0,1]m    │ [0,10]cm  │                    │
│   │ meters    │ decimeters│ centimeters│                   │
│   └───────────┴───────────┴───────────┘                    │
│   Precision: ±0.39mm (5× improvement!)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2.2 Cross-Dataset Generalization

**목표**: KITTI, NYU Depth 등 공개 데이터셋에서의 성능 검증

| Dataset | 특징 | 목표 abs_rel |
|---------|------|--------------|
| **NCDB** (자체) | Near-field, 도심 | 0.041 (달성) |
| **KITTI** | Outdoor, 자율주행 | < 0.10 |
| **NYU Depth v2** | Indoor | < 0.12 |

### 5.2.3 Dynamic Depth Range

**아이디어**: 장면에 따라 동적으로 깊이 범위 조정

```python
class AdaptiveDepthRange(nn.Module):
    """
    장면 특성에 따라 max_depth 동적 조정
    """
    def forward(self, features):
        # Scene classification
        scene_type = self.scene_classifier(features)
        
        # Adaptive range selection
        if scene_type == 'indoor':
            max_depth = 10.0
        elif scene_type == 'urban':
            max_depth = 15.0
        else:  # highway
            max_depth = 80.0
        
        return max_depth
```

---

## 5.3 장기 목표 (1년+)

### 5.3.1 Transformer Integration

**목표**: ViT 기반 Encoder로 전역 컨텍스트 강화

```
┌─────────────────────────────────────────────────────────────┐
│              Hybrid CNN-Transformer Architecture             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐                                           │
│   │   RGB       │                                           │
│   │   Input     │                                           │
│   └──────┬──────┘                                           │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────┐      ┌─────────────┐                     │
│   │ ResNet18    │      │ Swin-Tiny   │                     │
│   │ (Local)     │ ──── │ (Global)    │                     │
│   └─────────────┘      └─────────────┘                     │
│          │                    │                             │
│          └────────┬───────────┘                             │
│                   │                                         │
│                   ▼                                         │
│          ┌─────────────────┐                               │
│          │ Feature Fusion  │                               │
│          └────────┬────────┘                               │
│                   │                                         │
│                   ▼                                         │
│          ┌─────────────────┐                               │
│          │ Dual-Head       │                               │
│          │ Decoder         │                               │
│          └─────────────────┘                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**기대 효과**:
- 넓은 receptive field로 전역 깊이 일관성 향상
- 복잡한 구조물에서의 정확도 개선

### 5.3.2 Self-Supervised + Supervised Hybrid

**목표**: GT 없는 데이터도 활용하는 Semi-supervised 학습

```python
class HybridLoss(nn.Module):
    """
    Supervised + Self-supervised Hybrid Loss
    """
    def forward(self, pred, gt=None, context_frames=None):
        loss = 0
        
        # Supervised component (when GT available)
        if gt is not None:
            loss += self.supervised_loss(pred, gt)
        
        # Self-supervised component (always available)
        if context_frames is not None:
            loss += self.photometric_loss(pred, context_frames)
        
        return loss
```

### 5.3.3 Multi-Task Learning

**목표**: Depth + Semantic Segmentation + Object Detection 통합

```
┌─────────────────────────────────────────────────────────────┐
│                 Multi-Task Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                   Shared Encoder                            │
│                   (ResNet18-SAN)                            │
│                        │                                    │
│         ┌──────────────┼──────────────┐                    │
│         │              │              │                    │
│         ▼              ▼              ▼                    │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐             │
│   │  Depth    │  │ Semantic  │  │ Detection │             │
│   │  Head     │  │   Head    │  │   Head    │             │
│   │ (Dual)    │  │           │  │ (YOLO)    │             │
│   └───────────┘  └───────────┘  └───────────┘             │
│                                                             │
│   Benefits:                                                 │
│   • 공유 특징 추출 → 효율성                                 │
│   • 태스크 간 상보적 정보 활용                              │
│   • 단일 모델로 다중 기능                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5.4 기술적 개선 사항

### 5.4.1 Uncertainty Estimation

**목표**: 깊이 예측의 신뢰도 함께 출력

```python
class UncertaintyHead(nn.Module):
    """
    깊이 예측 불확실성 추정
    """
    def forward(self, features):
        depth = self.depth_head(features)       # [B, 1, H, W]
        log_var = self.uncertainty_head(features)  # [B, 1, H, W]
        
        # Aleatoric uncertainty
        uncertainty = torch.exp(0.5 * log_var)
        
        return depth, uncertainty
```

**활용**:
- 신뢰도 낮은 영역 필터링
- 다중 센서 융합 시 가중치 결정
- 안전 크리티컬 응용에서 신뢰도 기반 의사결정

### 5.4.2 Temporal Consistency

**목표**: 비디오에서 시간적 일관성 확보

```python
class TemporalDepthNet(nn.Module):
    """
    시간적 일관성을 갖춘 깊이 추정
    """
    def __init__(self):
        self.depth_net = ResNetSAN01()
        self.temporal_fusion = ConvLSTM(channels=256)
        
    def forward(self, frames):  # [B, T, C, H, W]
        depths = []
        hidden_state = None
        
        for t in range(frames.shape[1]):
            feat = self.depth_net.encoder(frames[:, t])
            feat, hidden_state = self.temporal_fusion(feat, hidden_state)
            depth = self.depth_net.decoder(feat)
            depths.append(depth)
        
        return torch.stack(depths, dim=1)
```

### 5.4.3 Edge-Aware Refinement

**목표**: 물체 경계에서의 깊이 정확도 향상

```
Edge-aware Loss:
L_edge = L_depth + λ * L_gradient

where:
  L_gradient = |∇d_pred - ∇d_gt| * edge_weight
  edge_weight = ∇I_rgb  (RGB gradient as edge indicator)
```

---

## 5.5 응용 분야 확장

### 5.5.1 Target Applications

| 분야 | 요구 사항 | 현재 상태 |
|------|----------|-----------|
| **ADAS** | 실시간, 높은 정확도 | ✅ 가능 |
| **로보틱스** | 근거리 정밀도 | ✅ 최적화됨 |
| **AR/VR** | 저지연, 일관성 | 🔄 개선 필요 |
| **드론** | 경량, 저전력 | ✅ 가능 |

### 5.5.2 Deployment Platforms

```
┌─────────────────────────────────────────────────────────────┐
│                  Deployment Targets                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [Current]                    [Future]                     │
│   ┌───────────────┐           ┌───────────────┐            │
│   │ Edge NPU      │           │ Mobile SoC    │            │
│   │ (INT8)        │           │ (INT8/FP16)   │            │
│   └───────────────┘           └───────────────┘            │
│                                                             │
│   ┌───────────────┐           ┌───────────────┐            │
│   │ NVIDIA        │           │ Qualcomm      │            │
│   │ Jetson        │           │ Hexagon DSP   │            │
│   └───────────────┘           └───────────────┘            │
│                                                             │
│   ┌───────────────┐           ┌───────────────┐            │
│   │ x86 Server    │           │ Apple Neural  │            │
│   │ (ONNX Runtime)│           │ Engine        │            │
│   └───────────────┘           └───────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5.6 연구 방향 요약

### 단기 (3개월)
- ✅ Dual-Head INT8 양자화 완료
- ✅ FiLM + Dual-Head 통합
- ✅ NPU 실배포 검증

### 중기 (6개월)
- 🔄 Multi-Head 확장 (Tri-Head)
- 🔄 Cross-Dataset 일반화
- 🔄 Dynamic Depth Range

### 장기 (1년+)
- 📋 Transformer 통합
- 📋 Semi-supervised 학습
- 📋 Multi-Task Learning
- 📋 Uncertainty Estimation

---

## 5.7 Conclusion

본 연구에서는 **NPU INT8 양자화에 최적화된 ResNet18-SAN Dual-Head** 아키텍처를 제안하였다.

### 주요 성과
1. **FP32 성능**: abs_rel 0.0414 (목표 0.065 대비 36% 우수)
2. **경량화**: ResNet18 기반 11.7M 파라미터
3. **양자화 최적화**: Dual-Head로 14배 정밀도 향상 설계

### 핵심 기여
- Integer-Fractional 분리 출력으로 Per-tensor 양자화 한계 극복
- 모듈화된 설계로 유연한 기능 확장 가능
- Edge NPU 배포에 최적화된 아키텍처

### 향후 전망
- INT8 Dual-Head 양자화 완료 후 목표 성능 달성 예상
- Multi-Head 확장으로 추가 정밀도 향상 가능
- 다양한 응용 분야로 확장 계획

---

# Thank You

**Questions?**

---

## Contact

- **Project Repository**: `/workspace/packnet-sfm`
- **Documentation**: `docs/quantization/ST2/`
- **Configuration**: `configs/train_resnet_san_ncdb_dual_head_640x384.yaml`
