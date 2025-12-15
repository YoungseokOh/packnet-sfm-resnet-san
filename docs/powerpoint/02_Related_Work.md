# 2. Related Work

---

## 2.1 Monocular Depth Estimation

### 2.1.1 개요

Monocular Depth Estimation은 단일 RGB 이미지로부터 각 픽셀의 깊이(거리) 값을 예측하는 컴퓨터 비전 태스크이다. 이는 본질적으로 ill-posed 문제로, 2D 이미지에서 3D 정보를 복원해야 하는 어려움이 있다.

```
┌─────────────────────────────────────────────────────────────┐
│           Monocular Depth Estimation Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input                    Model                  Output    │
│   ┌─────────┐         ┌───────────┐         ┌──────────┐   │
│   │  RGB    │  ─────▶ │   CNN/    │  ─────▶ │  Depth   │   │
│   │ Image   │         │   ViT     │         │   Map    │   │
│   │ H×W×3   │         │           │         │  H×W×1   │   │
│   └─────────┘         └───────────┘         └──────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.1.2 주요 접근 방법

| 방법 | 대표 논문 | 장점 | 단점 |
|------|-----------|------|------|
| **Supervised** | Eigen et al. [1], DORN [2] | 높은 정확도 | GT 획득 비용 |
| **Self-supervised** | Monodepth [3], PackNet [4] | GT 불필요 | 스케일 모호성 |
| **Stereo-based** | PSMNet [5] | 정확한 깊이 | 스테레오 카메라 필요 |
| **Depth Completion** | S2D [6], SAN [7] | Sparse 입력 활용 | LiDAR 필요 |

### 2.1.3 주요 연구 흐름

```
Timeline of Depth Estimation Research
──────────────────────────────────────────────────────────────

2014    Eigen et al. - Multi-scale CNN for depth prediction
         │
2016    Godard et al. - Monodepth (Self-supervised)
         │
2017    Garg et al. - Unsupervised CNN for depth
         │
2019    Godard et al. - Monodepth2 (Improved self-supervised)
         │
2020    PackNet-SfM - 3D Packing for self-supervised depth
         │ 
2021    MiDaS - Cross-dataset generalization
         │
2022    DPT - Vision Transformer for dense prediction
         │
2023    Depth Anything - Foundation model for depth
         │
2024    Marigold - Diffusion-based depth estimation

```

---

## 2.2 Sparse-to-Dense Depth Completion

### 2.2.1 개요

Depth Completion은 Sparse한 깊이 정보(LiDAR 등)를 Dense한 깊이 맵으로 변환하는 태스크이다. RGB 이미지와 함께 사용하여 정확도를 높인다.

```
┌─────────────────────────────────────────────────────────────┐
│              Sparse-to-Dense Depth Completion               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   RGB Image              Sparse Depth        Dense Depth    │
│   ┌─────────┐           ┌─────────┐         ┌──────────┐   │
│   │         │           │ .  . .  │         │▓▓▓▓▓▓▓▓▓▓│   │
│   │  Scene  │    +      │.    .   │  ─────▶ │▓▓▓▓▓▓▓▓▓▓│   │
│   │         │           │   .   . │         │▓▓▓▓▓▓▓▓▓▓│   │
│   └─────────┘           └─────────┘         └──────────┘   │
│                         (~5% density)       (100% density)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2.2 주요 방법론

**1. Early Fusion**
- RGB와 Sparse Depth를 입력 레벨에서 결합
- 간단하지만 정보 손실 가능

**2. Late Fusion**
- 각각 별도 인코더로 처리 후 결합
- 복잡하지만 정보 보존

**3. Guided Upsampling**
- RGB 특징으로 Sparse Depth 업샘플링 가이드
- 에지 보존에 효과적

### 2.2.3 Sparse Convolution

Sparse 데이터를 효율적으로 처리하기 위해 **Sparse Convolution**이 활용된다.

```python
# Traditional Dense Convolution
for all pixels:
    output[i,j] = conv(input[i,j], kernel)  # 모든 픽셀 연산

# Sparse Convolution (Minkowski Engine)
for valid pixels only:
    output[valid_idx] = conv(input[valid_idx], kernel)  # 유효 픽셀만
```

**장점**:
- 메모리 효율적 (유효 픽셀만 저장)
- 연산 효율적 (필요한 곳만 계산)
- 희소 구조 보존

---

## 2.3 SAN (Sparse Attention Network)

### 2.3.1 PackNet-SAN 개요

**PackNet-SAN** [Toyota Research Institute, 2020]은 Self-supervised Depth Estimation에 Sparse LiDAR 정보를 효과적으로 융합하는 아키텍처이다.

```
┌─────────────────────────────────────────────────────────────┐
│                    PackNet-SAN Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐                    ┌──────────────────┐     │
│   │   RGB    │                    │   Sparse Depth   │     │
│   └────┬─────┘                    └────────┬─────────┘     │
│        │                                   │               │
│        ▼                                   ▼               │
│   ┌──────────┐                    ┌──────────────────┐     │
│   │ PackNet  │                    │   Minkowski      │     │
│   │ Encoder  │                    │   Encoder        │     │
│   └────┬─────┘                    └────────┬─────────┘     │
│        │                                   │               │
│        │     ┌───────────────────┐         │               │
│        └────▶│   FiLM Fusion     │◀────────┘               │
│              │ (Feature-wise     │                         │
│              │  Linear Modulation)│                         │
│              └─────────┬─────────┘                         │
│                        │                                   │
│                        ▼                                   │
│               ┌──────────────┐                             │
│               │   Decoder    │                             │
│               └──────────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3.2 FiLM (Feature-wise Linear Modulation)

FiLM은 조건부 정보를 특징 맵에 주입하는 효과적인 방법이다.

```
FiLM Modulation:

y = γ ⊙ x + β

where:
  x : RGB feature map [B, C, H, W]
  γ : Scale from sparse depth [B, C, H, W]
  β : Shift from sparse depth [B, C, H, W]
  y : Modulated feature [B, C, H, W]
```

**작동 원리**:
1. Minkowski Encoder가 Sparse Depth에서 γ, β 생성
2. RGB 특징에 채널별로 스케일(γ)과 시프트(β) 적용
3. Sparse 정보가 Dense 특징을 조절

### 2.3.3 Minkowski Engine

**MinkowskiEngine**은 Sparse Tensor 연산을 위한 라이브러리로, 3D 포인트 클라우드 처리에 널리 사용된다.

```python
import MinkowskiEngine as ME

class MinkowskiEncoder(nn.Module):
    def __init__(self, channels):
        self.conv = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=channels,
            kernel_size=3,
            dimension=2
        )
    
    def forward(self, sparse_depth):
        # 유효한 픽셀만 처리
        coords, feats = sparse_depth.coordinates, sparse_depth.features
        sparse_tensor = ME.SparseTensor(feats, coords)
        return self.conv(sparse_tensor)
```

---

## 2.4 Neural Network Quantization

### 2.4.1 양자화 개요

**Quantization**은 FP32 가중치와 활성화를 저비트(INT8/INT4)로 변환하여 모델을 경량화하는 기술이다.

```
┌─────────────────────────────────────────────────────────────┐
│                   Quantization Process                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   FP32 Values                        INT8 Values            │
│   ┌─────────────────┐               ┌──────────────┐       │
│   │ -1.5, 0.3, 2.1  │    ─────▶     │  -128 ~ 127  │       │
│   │ (continuous)    │    Quantize   │  (256 levels)│       │
│   └─────────────────┘               └──────────────┘       │
│                                                             │
│   Quantization Formula:                                     │
│   x_q = round((x - zero_point) / scale)                    │
│   x_dq = x_q * scale + zero_point                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.4.2 양자화 유형

| 유형 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **PTQ** (Post-Training) | 학습 후 양자화 | 빠름, 간단 | 정확도 손실 |
| **QAT** (Quantization-Aware) | 양자화 시뮬레이션 학습 | 높은 정확도 | 학습 비용 |
| **Per-tensor** | 텐서당 1개 scale/zp | 하드웨어 효율 | 정밀도 낮음 |
| **Per-channel** | 채널당 scale/zp | 정밀도 높음 | 하드웨어 지원 필요 |

### 2.4.3 NPU 제약사항

본 연구에서 타겟하는 NPU는 다음과 같은 제약이 있다:

```
┌─────────────────────────────────────────────────────────────┐
│                    NPU Constraints                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ Supported:                                              │
│     • INT8 Per-tensor quantization                          │
│     • Basic convolution, pooling                            │
│     • Dual-output (multiple heads)                          │
│                                                             │
│  ❌ Not Supported:                                          │
│     • Per-channel quantization                              │
│     • Dynamic quantization                                  │
│     • Mixed precision (FP16/INT8)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.4.4 깊이 추정에서의 양자화 문제

**문제**: 깊이 값의 넓은 범위가 양자화 정밀도를 저하시킴

```
Single Output (0.5m ~ 15.0m):
─────────────────────────────────────────────────
│        14.5m range ÷ 255 levels = 56.9mm     │
│                     ↓                         │
│            Quantization Error: ±28.4mm        │
─────────────────────────────────────────────────

vs.

Dual Output (Integer + Fractional):
─────────────────────────────────────────────────
│  Integer [0,15]:   16 values (충분)           │
│  Fractional [0,1]: 1m ÷ 255 = 3.92mm         │
│                     ↓                         │
│            Quantization Error: ±1.96mm        │
─────────────────────────────────────────────────
```

---

## 2.5 Encoder Architectures

### 2.5.1 ResNet Family

**ResNet** [He et al., 2015]은 Skip Connection을 통해 깊은 네트워크 학습을 가능하게 한 아키텍처이다.

```
┌─────────────────────────────────────────────────────────────┐
│                   ResNet Residual Block                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input x                                                   │
│      │                                                      │
│      ├────────────────────────┐                             │
│      │                        │                             │
│      ▼                        │                             │
│   ┌──────┐                    │                             │
│   │Conv1 │                    │  (Identity Shortcut)        │
│   │3×3   │                    │                             │
│   └──┬───┘                    │                             │
│      │                        │                             │
│      ▼                        │                             │
│   ┌──────┐                    │                             │
│   │Conv2 │                    │                             │
│   │3×3   │                    │                             │
│   └──┬───┘                    │                             │
│      │                        │                             │
│      ▼                        │                             │
│   ┌──────┐◀───────────────────┘                             │
│   │  +   │   F(x) + x = H(x)                                │
│   └──┬───┘                                                  │
│      │                                                      │
│   Output H(x)                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.5.2 ResNet Variants

| Model | Layers | Params | Top-1 Acc | Use Case |
|-------|--------|--------|-----------|----------|
| **ResNet18** | 18 | 11.7M | 69.8% | **Edge (본 연구)** |
| ResNet34 | 34 | 21.8M | 73.3% | Mobile |
| ResNet50 | 50 | 25.6M | 76.1% | Server |
| ResNet101 | 101 | 44.5M | 77.4% | High-end |

### 2.5.3 왜 ResNet18인가?

본 연구에서 ResNet18을 선택한 이유:

1. **경량성**: 11.7M 파라미터 (ResNet50 대비 54% 감소)
2. **추론 속도**: Edge NPU에서 실시간 처리 가능
3. **충분한 표현력**: Depth Estimation에 적합한 특징 추출
4. **ImageNet Pretrained**: 효과적인 전이 학습

---

## 2.6 Summary

| 분야 | 핵심 연구 | 본 연구 적용 |
|------|----------|--------------|
| **Depth Estimation** | Monodepth, PackNet | Supervised 학습 |
| **Depth Completion** | S2D, SAN | Minkowski Encoder |
| **Feature Fusion** | FiLM | Depth-aware 특징 조절 |
| **Quantization** | PTQ, QAT | Dual-Head 설계 |
| **Encoder** | ResNet family | ResNet18 선택 |

**본 연구의 차별점**:
- 기존 SAN 구조를 **ResNet18 기반**으로 경량화
- **INT8 양자화**에 최적화된 **Dual-Head** 출력 설계
- **Per-tensor 제약** 하에서 정밀도 14배 향상

---

## References

[1] Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", NeurIPS 2014

[2] Fu et al., "Deep Ordinal Regression Network for Monocular Depth Estimation", CVPR 2018

[3] Godard et al., "Unsupervised Monocular Depth Estimation with Left-Right Consistency", CVPR 2017

[4] Guizilini et al., "3D Packing for Self-Supervised Monocular Depth Estimation", CVPR 2020

[5] Chang et al., "Pyramid Stereo Matching Network", CVPR 2018

[6] Ma et al., "Self-supervised Sparse-to-Dense", ICRA 2019

[7] Guizilini et al., "Sparse Auxiliary Networks for Unified Monocular Depth Prediction and Completion", CVPR 2021
