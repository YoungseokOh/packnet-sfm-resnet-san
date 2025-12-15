# Sparse Auxiliary Networks (SAN) - 발표자료

---

## 1. SAN 개요

### 1.1 SAN이란?

**SAN (Sparse Auxiliary Networks)**은 **Sparse LiDAR 데이터**를 보조 정보로 활용하여 **RGB 기반 Depth Estimation**의 정확도를 향상시키는 네트워크 아키텍처이다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SAN의 핵심 아이디어                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   "학습 시에만 LiDAR를 사용하고, 추론 시에는 RGB만으로 예측"          │
│                                                                     │
│   ┌─────────────────┐          ┌─────────────────────────────────┐ │
│   │    학습 시       │          │           추론 시                │ │
│   │  RGB + LiDAR    │   ───▶   │         RGB만 사용               │ │
│   │  (Teacher)      │          │       (LiDAR 불필요)             │ │
│   └─────────────────┘          └─────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 SAN의 핵심 가치

| 특성 | 설명 |
|------|------|
| **학습 효율성** | LiDAR가 제공하는 정확한 깊이 정보로 더 정확한 학습 가능 |
| **추론 독립성** | 학습된 모델은 RGB만으로 동작 (LiDAR 센서 불필요) |
| **비용 절감** | 추론 시 고가의 LiDAR 센서 없이 deployment 가능 |
| **실용성** | 다양한 환경(LiDAR 없는 차량, 모바일 기기)에서 활용 가능 |

---

## 2. SAN 아키텍처

### 2.1 전체 구조

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        ResNet-SAN Architecture                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐                              ┌─────────────────┐       │
│   │    RGB      │                              │  Sparse LiDAR   │       │
│   │   Image     │                              │  Depth (~5%)    │       │
│   │  640×384×3  │                              │   640×384×1     │       │
│   └──────┬──────┘                              └────────┬────────┘       │
│          │                                              │                │
│          ▼                                              ▼                │
│   ┌──────────────┐                           ┌────────────────────┐      │
│   │   ResNet18   │                           │   Minkowski        │      │
│   │   Encoder    │                           │   Encoder          │      │
│   │              │                           │ (Sparse Conv)      │      │
│   └──────┬───────┘                           └─────────┬──────────┘      │
│          │                                             │                 │
│    Scale Features                               FiLM Parameters          │
│    [64, 64, 128, 256, 512]                      (γ, β)                  │
│          │                                             │                 │
│          │         ┌───────────────────────────┐       │                 │
│          └────────▶│      FiLM Modulation      │◀──────┘                 │
│                    │    y = γ × x + β          │                         │
│                    └───────────┬───────────────┘                         │
│                                │                                         │
│                                ▼                                         │
│                    ┌───────────────────────────┐                         │
│                    │   Dual-Head Decoder       │                         │
│                    │  ┌─────────┐ ┌─────────┐  │                         │
│                    │  │ Integer │ │Fractional│  │                         │
│                    │  │  Head   │ │  Head   │  │                         │
│                    │  └────┬────┘ └────┬────┘  │                         │
│                    └───────┼───────────┼───────┘                         │
│                            │           │                                 │
│                            ▼           ▼                                 │
│                    ┌─────────────────────────┐                           │
│                    │ Final Depth = Int + Frac │                          │
│                    │     [0, max_depth]m      │                          │
│                    └─────────────────────────┘                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 핵심 컴포넌트

#### 2.2.1 RGB Encoder (ResNet18)

```python
# ResNet18 Encoder - 5개 스케일의 Feature 추출
class ResnetEncoder:
    def forward(self, rgb):
        features = []
        x = self.conv1(rgb)          # Scale 0: 64 channels
        x = self.layer1(x)           # Scale 1: 64 channels  
        x = self.layer2(x)           # Scale 2: 128 channels
        x = self.layer3(x)           # Scale 3: 256 channels
        x = self.layer4(x)           # Scale 4: 512 channels
        return features  # [64, 64, 128, 256, 512]
```

#### 2.2.2 Minkowski Encoder (Sparse Convolution)

```python
# Sparse Depth 처리를 위한 Minkowski Engine 활용
class MinkowskiEncoder:
    """
    - 유효한 LiDAR 포인트만 처리 (~5% of pixels)
    - Sparse Convolution으로 메모리/연산 효율적
    - FiLM 파라미터 (γ, β) 생성
    """
    def forward(self, sparse_depth):
        # 1. Sparse Conv 처리
        sparse_features = self.mconvs(sparse_depth)
        
        # 2. Dense로 변환
        dense_features = densify_features(sparse_features)
        
        # 3. FiLM 파라미터 생성
        gamma, beta = self.film_generator(dense_features)
        
        return dense_features, gamma, beta
```

#### 2.2.3 FiLM (Feature-wise Linear Modulation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FiLM 동작 원리                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input:  x = RGB Feature [B, C, H, W]                                  │
│   From Sparse: γ (scale), β (shift) [B, C, 1, 1]                       │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │                                                         │          │
│   │    y = γ ⊙ x + β                                       │          │
│   │                                                         │          │
│   │    ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐         │          │
│   │    │  x  │  ×  │  γ  │  +  │  β  │  =  │  y  │         │          │
│   │    │(RGB)│     │(scale)    │(shift)│     │(fused)       │          │
│   │    └─────┘     └─────┘     └─────┘     └─────┘         │          │
│   │                                                         │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                                         │
│   효과:                                                                 │
│   - γ > 1: 해당 채널 강조 (LiDAR가 중요하다고 판단한 특징)              │
│   - γ < 1: 해당 채널 억제                                              │
│   - β ≠ 0: 특징 공간에서의 이동 (Bias 조정)                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 학습 vs 추론 플로우

### 3.1 학습 시 (Training)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         학습 플로우 (Training)                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Step 1: RGB-only Forward (FiLM OFF)                                    │
│   ─────────────────────────────────────                                  │
│   ┌─────────┐    ┌───────────┐    ┌──────────┐                          │
│   │   RGB   │───▶│  Encoder  │───▶│ Decoder  │───▶ Pred_rgb             │
│   └─────────┘    └───────────┘    └──────────┘     (RGB만의 예측)       │
│                                                                          │
│   Step 2: RGB+Depth Forward (FiLM ON)                                    │
│   ───────────────────────────────────                                    │
│   ┌─────────┐    ┌───────────┐                                          │
│   │   RGB   │───▶│  Encoder  │───┐                                      │
│   └─────────┘    └───────────┘   │                                      │
│                                  ├───▶ FiLM ───▶ Decoder ───▶ Pred_film │
│   ┌─────────┐    ┌───────────┐   │     Fusion   (LiDAR 가이드 예측)      │
│   │  LiDAR  │───▶│ Minkowski │───┘                                      │
│   └─────────┘    └───────────┘                                          │
│                                                                          │
│   Step 3: Loss 계산                                                      │
│   ─────────────────                                                      │
│   • Supervised Loss: L1(Pred_rgb, GT) + L1(Pred_film, GT)               │
│   • Feature Consistency: ||Feat_rgb - Feat_film||                       │
│                                                                          │
│   핵심: RGB-only 예측도 GT와 가깝게 학습                                 │
│         → 추론 시 RGB만으로도 정확한 예측 가능                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 추론 시 (Inference)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         추론 플로우 (Inference)                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ✅ LiDAR 불필요! RGB만 사용                                            │
│                                                                          │
│   ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌──────────────┐      │
│   │   RGB   │───▶│  ResNet   │───▶│ Decoder  │───▶│  Depth Map   │      │
│   │  Image  │    │  Encoder  │    │          │    │  640×384×1   │      │
│   └─────────┘    └───────────┘    └──────────┘    └──────────────┘      │
│                                                                          │
│   • Minkowski Encoder: 비활성화 (use_film=False)                        │
│   • FiLM Modulation: 스킵                                               │
│   • 결과: 학습된 지식으로 RGB만으로 정확한 깊이 예측                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Minkowski Engine (Sparse Convolution)

### 4.1 왜 Sparse Convolution이 필요한가?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LiDAR 데이터의 특성                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Dense Depth (카메라)         vs      Sparse Depth (LiDAR)             │
│                                                                         │
│   ┌─────────────────┐                 ┌─────────────────┐               │
│   │█████████████████│                 │.    .   .    .  │               │
│   │█████████████████│                 │  .    .    .    │               │
│   │█████████████████│                 │.   .    .   .   │               │
│   │█████████████████│                 │   .   .    . .  │               │
│   └─────────────────┘                 └─────────────────┘               │
│       100% 밀도                          ~5% 밀도                        │
│                                                                         │
│   문제: 일반 Convolution은 빈 픽셀도 연산 → 95% 낭비!                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Sparse Convolution 동작 방식

```python
# Traditional Dense Convolution (비효율적)
for i in range(H):
    for j in range(W):
        output[i,j] = conv(input[i,j], kernel)  # 모든 픽셀 연산 (95% 낭비)

# Minkowski Sparse Convolution (효율적)
for (i, j) in valid_coordinates:  # 유효한 ~5%만
    output[i,j] = conv(input[i,j], kernel)
```

### 4.3 Minkowski Engine 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Minkowski Encoder 구조                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: Sparse LiDAR Depth [640×384, ~5% valid]                        │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  1. Sparsify (유효 좌표 추출)                                   │     │
│   │     coords = [(x1,y1), (x2,y2), ...]  # ~12,000 points         │     │
│   │     feats  = [d1, d2, ...]            # depth values           │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  2. Sparse Convolution (Multi-scale)                           │     │
│   │     MinkConv2D (5×5) → 64 ch → Pool(2)                         │     │
│   │     MinkConv2D (5×5) → 64 ch → Pool(2)                         │     │
│   │     MinkConv2D (3×3) → 128 ch → Pool(2)                        │     │
│   │     MinkConv2D (3×3) → 256 ch → Pool(2)                        │     │
│   │     MinkConv2D (3×3) → 512 ch                                  │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  3. Densify (Dense Feature로 변환)                              │     │
│   │     Sparse Features → Dense [B, C, H, W]                       │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  4. FiLM Parameter Generation                                  │     │
│   │     AdaptiveAvgPool2d(1) → Conv2d → γ, β                       │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. FiLM이 Scale 0에만 적용되는 이유

### 5.1 Multi-Scale Feature의 역할

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ResNet Feature Scales                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Scale 0 (1/2 해상도)  ─── Low-level features                          │
│   ├── 엣지, 텍스처, 경계                                                │
│   ├── 깊이 불연속점 (객체 경계)                                         │
│   └── LiDAR 포인트와 가장 직접적 대응                                   │
│                                                                         │
│   Scale 1 (1/4 해상도)  ─── Mid-low features                            │
│   ├── 작은 패턴                                                         │
│   └── 국소적 구조                                                       │
│                                                                         │
│   Scale 2 (1/8 해상도)  ─── Mid-level features                          │
│   ├── 객체 부분                                                         │
│   └── 중간 크기 구조                                                    │
│                                                                         │
│   Scale 3 (1/16 해상도) ─── High-level features                         │
│   ├── 전체 객체                                                         │
│   └── 의미론적 정보                                                     │
│                                                                         │
│   Scale 4 (1/32 해상도) ─── Global features                             │
│   ├── 장면 컨텍스트                                                     │
│   └── 전역 정보                                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Scale 0에만 FiLM 적용 이유

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  왜 Scale 0에만 FiLM을 적용하는가?                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. 해상도 일치                                                        │
│      • Scale 0: 320×192 (원본의 1/2)                                   │
│      • LiDAR Sparse Conv 출력도 같은 해상도                             │
│      • 고해상도에서 정밀한 깊이 정보 주입 가능                          │
│                                                                         │
│   2. 깊이 경계 정보                                                     │
│      • Scale 0은 엣지/경계 정보 보유                                    │
│      • LiDAR는 깊이 불연속점에서 정확한 정보 제공                       │
│      • 객체 경계에서의 정확도 향상                                      │
│                                                                         │
│   3. 연산 효율성                                                        │
│      • 모든 스케일에 적용 시 연산량 5배 증가                            │
│      • Scale 0만으로도 충분한 효과                                      │
│      • 효율과 성능의 균형                                               │
│                                                                         │
│   4. 정보 전파                                                          │
│      • Scale 0에서 주입된 정보가 Decoder를 통해 전파                    │
│      • Skip Connection으로 모든 해상도에 영향                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 본 연구에서의 SAN 활용

### 6.1 ResNet18-SAN + Dual-Head

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   우리의 SAN 활용 방식                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   기존 PackNet-SAN                    본 연구 (ResNet18-SAN)            │
│   ┌─────────────────┐                 ┌─────────────────────────────┐   │
│   │ • PackNet Encoder│                │ • ResNet18 Encoder (경량화)  │   │
│   │ • 단일 Depth Head │                │ • Dual-Head (INT8 최적화)    │   │
│   │ • FP32 연산      │                │ • 30m max_depth              │   │
│   │ • GPU 추론      │                │ • NPU 배포 대상              │   │
│   └─────────────────┘                 └─────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Dual-Head와 SAN의 시너지

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Dual-Head + SAN 시너지                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SAN (학습 시)                                                         │
│   ├── LiDAR가 정확한 깊이 정보 제공                                     │
│   ├── FiLM으로 RGB 특징 조정                                            │
│   └── RGB-only 예측도 정확하게 학습                                     │
│                                                                         │
│   Dual-Head (추론 시)                                                   │
│   ├── Integer Head: [0, 30]m → INT8 (31 levels)                        │
│   ├── Fractional Head: [0, 1]m → INT8 (256 levels)                     │
│   └── Final = Integer + Fractional (3.92mm 정밀도)                     │
│                                                                         │
│   결과:                                                                 │
│   • 학습: LiDAR로 정확한 깊이 학습                                      │
│   • 추론: RGB만으로 Dual-Head가 정밀한 깊이 출력                        │
│   • NPU: INT8 양자화로 효율적 배포                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 코드 구현 핵심

### 7.1 ResNetSAN01 클래스

```python
class ResNetSAN01(nn.Module):
    def __init__(self, use_film=False, use_dual_head=False, 
                 max_depth=30.0, ...):
        # RGB Encoder
        self.encoder = ResnetEncoder(num_layers=18, pretrained=True)
        
        # Decoder 선택
        if use_dual_head:
            self.decoder = DualHeadDepthDecoder(max_depth=max_depth)
        else:
            self.decoder = DepthDecoder()
        
        # Minkowski Encoder (학습 시에만 필요)
        if use_film:
            self.mconvs = MinkowskiEncoder(
                channels=self.encoder.num_ch_enc,
                rgb_channels=[64, 0, 0, 0, 0]  # Scale 0만
            )
        else:
            self.mconvs = None  # 추론 시 비활성화
```

### 7.2 학습 시 Forward

```python
def run_network(self, rgb, input_depth=None):
    # 1. RGB Encoding
    features = self.encoder(rgb)
    
    # 2. FiLM 적용 (학습 시 + LiDAR 있을 때)
    if input_depth is not None and self.use_film:
        self.mconvs.prep(input_depth)
        
        for i, feat in enumerate(features):
            if i in self.film_scales:  # Scale 0만
                _, gamma, beta = self.mconvs(feat)
                features[i] = gamma * feat + beta  # FiLM!
    
    # 3. Decoding
    outputs = self.decoder(features)
    return outputs
```

### 7.3 추론 시 Forward

```python
def forward(self, rgb):
    # use_film=False 상태
    # → mconvs가 None이므로 FiLM 스킵
    # → RGB만으로 예측
    
    features = self.encoder(rgb)
    outputs = self.decoder(features)
    return outputs
```

---

## 8. 요약

### 8.1 SAN 핵심 정리

| 항목 | 내용 |
|------|------|
| **목적** | 학습 시 LiDAR 활용, 추론 시 RGB만 사용 |
| **핵심 기술** | FiLM (Feature-wise Linear Modulation) |
| **Sparse 처리** | Minkowski Engine (Sparse Convolution) |
| **FiLM 적용** | Scale 0만 (효율성 + 효과) |
| **장점** | 추론 시 LiDAR 불필요, 배포 용이 |

### 8.2 본 연구 적용

| 항목 | 내용 |
|------|------|
| **Backbone** | ResNet18 (경량화) |
| **Decoder** | Dual-Head (Integer + Fractional) |
| **Max Depth** | 30m (NCDB 데이터 99% 커버) |
| **Target** | NPU 배포 (INT8 양자화) |
| **정밀도** | 3.92mm (INT8 Dual-Head) |

---

## 참고문헌

1. **PackNet-SfM**: Guizilini et al., "3D Packing for Self-Supervised Monocular Depth Estimation", CVPR 2020
2. **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018  
3. **Minkowski Engine**: Choy et al., "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks", CVPR 2019
4. **PackNet-SAN**: Guizilini et al., "Semantically-Guided Representation Learning for Self-Supervised Monocular Depth", ICLR 2020

---

*이 문서는 발표자료 작성을 위한 SAN 관련 내용 정리입니다.*
