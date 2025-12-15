# 3. Method

---

## 3.1 Overall Architecture

### 3.1.1 ResNet18-SAN 전체 구조

본 연구에서 제안하는 **ResNet18-SAN Dual-Head** 아키텍처는 RGB 이미지로부터 깊이를 예측하는 모노큘러 깊이 추정 모델이다. 이 모델의 핵심 아이디어는 **깊이 출력을 정수부(Integer)와 소수부(Fractional)로 분리**하여 NPU INT8 양자화 시 발생하는 정밀도 손실을 최소화하는 것이다.

기존 깊이 추정 모델들은 단일 출력으로 0~30m 전체 범위를 예측하는데, 이를 INT8(256단계)로 양자화하면 약 117mm의 양자화 간격이 발생한다. 반면 Dual-Head 구조에서는 소수부가 0~1m 범위만 담당하므로 양자화 간격이 약 4mm로 줄어들어 **약 30배의 정밀도 향상**을 달성한다.

전체 아키텍처는 다음과 같은 구성요소로 이루어진다:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ResNet18-SAN Dual-Head Architecture               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: RGB [B, 3, 384, 640]     Optional: Sparse Depth [B, 1, H, W]│
│          │                                    │                     │
│          ▼                                    ▼                     │
│   ┌──────────────┐                   ┌────────────────┐            │
│   │   ResNet18   │                   │   Minkowski    │            │
│   │   Encoder    │                   │   Encoder      │            │
│   │              │                   │   (Sparse Conv)│            │
│   │  ┌────────┐  │                   └───────┬────────┘            │
│   │  │ Conv1  │──┼──── Skip[0]: [64, 192, 320]         γ₀, β₀      │
│   │  │7×7, 64 │  │                           │                     │
│   │  └────────┘  │                           ▼                     │
│   │  ┌────────┐  │              ┌─────────────────────────┐        │
│   │  │Layer1  │──┼──── Skip[1]: [64, 96, 160]   │  FiLM   │        │
│   │  │ 2 블록 │  │                           │  Fusion │        │
│   │  └────────┘  │                           │         │        │
│   │  ┌────────┐  │              └──────┬──────────────┘        │
│   │  │Layer2  │──┼──── Skip[2]: [128, 48, 80]        │             │
│   │  │ 2 블록 │  │                                   │             │
│   │  └────────┘  │                                   │             │
│   │  ┌────────┐  │                                   │             │
│   │  │Layer3  │──┼──── Skip[3]: [256, 24, 40]        │             │
│   │  │ 2 블록 │  │                                   │             │
│   │  └────────┘  │                                   │             │
│   │  ┌────────┐  │                                   │             │
│   │  │Layer4  │──┼──── Skip[4]: [512, 12, 20]        │             │
│   │  │ 2 블록 │  │                                   │             │
│   │  └────────┘  │                                   │             │
│   └──────────────┘                                   │             │
│                                                      │             │
│                              Fused Features ◀────────┘             │
│                                    │                               │
│                                    ▼                               │
│                          ┌──────────────────┐                      │
│                          │   Dual-Head      │                      │
│                          │   Decoder        │                      │
│                          └────────┬─────────┘                      │
│                                   │                                │
│                    ┌──────────────┼──────────────┐                 │
│                    ▼              │              ▼                 │
│             ┌───────────┐        │       ┌───────────┐            │
│             │ Integer   │        │       │Fractional │            │
│             │ Head      │        │       │ Head      │            │
│             │ σ([0,1])  │        │       │ σ([0,1])  │            │
│             │ ×max_depth│        │       │ ×1.0m     │            │
│             └─────┬─────┘        │       └─────┬─────┘            │
│                   │              │             │                   │
│                   └──────────────┼─────────────┘                   │
│                                  │                                 │
│                                  ▼                                 │
│                         ┌────────────────┐                         │
│                         │  Final Depth   │                         │
│                         │  = Int + Frac  │                         │
│                         │  [B, 1, H, W]  │                         │
│                         └────────────────┘                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

위 다이어그램에서 데이터 흐름을 살펴보면:

1. **입력 단계**: RGB 이미지(640×384)가 ResNet18 Encoder로 들어가고, 선택적으로 LiDAR sparse depth가 Minkowski Encoder로 입력된다.

2. **특징 추출**: ResNet18은 5개 스케일의 특징 맵을 생성하며, 각 스케일은 이전보다 해상도가 절반으로 줄고 채널 수는 증가한다. 이 특징 맵들은 Skip Connection으로 디코더에 전달된다.

3. **특징 융합 (선택적)**: FiLM 모듈이 활성화된 경우, Minkowski Encoder가 생성한 γ(scale)와 β(bias) 파라미터로 RGB 특징을 조절한다. 이를 통해 LiDAR 정보가 RGB 특징에 주입된다.

4. **깊이 예측**: Dual-Head Decoder는 공유된 업샘플링 경로를 거친 후, Integer Head와 Fractional Head로 분기하여 각각 정수부와 소수부를 예측한다.

5. **최종 출력**: 두 헤드의 출력을 합산하여 최종 깊이 맵을 생성한다.

### 3.1.2 주요 특징

| 구성요소 | 역할 | 출력 채널 |
|----------|------|-----------|
| **ResNet18 Encoder** | RGB 특징 추출 | [64, 64, 128, 256, 512] |
| **Minkowski Encoder** | Sparse Depth 처리 | [64, 64, 128, 256, 512] |
| **FiLM Fusion** | 특징 조절 (선택적) | 동일 |
| **Dual-Head Decoder** | 깊이 예측 | Integer + Fractional |

---

## 3.2 ResNet18 Encoder

### 3.2.1 설계 배경

ResNet18을 인코더로 선택한 이유는 다음과 같다:

1. **경량성**: ResNet18은 약 11M 파라미터로, ResNet50(25M)이나 EfficientNet 대비 가볍다. Edge 디바이스에서의 실시간 추론을 고려할 때 적절한 선택이다.

2. **검증된 성능**: ImageNet에서 사전학습된 가중치를 활용하여 풍부한 시각적 특징을 추출할 수 있다. 깊이 추정 태스크에서도 충분한 표현력을 보인다.

3. **Skip Connection 구조**: ResNet의 잔차 연결(Residual Connection)과 별개로, 인코더의 각 스케일 특징을 디코더로 전달하는 Skip Connection을 통해 고해상도 정보를 보존한다.

### 3.2.2 구조

```python
class ResnetEncoder(nn.Module):
    """
    ResNet18 기반 이미지 인코더
    ImageNet pretrained 가중치 활용
    """
    def __init__(self, num_layers=18, pretrained=True):
        super().__init__()
        
        # ResNet18 로드
        self.encoder = torchvision.models.resnet18(pretrained=pretrained)
        
        # 채널 수: [64, 64, 128, 256, 512]
        self.num_ch_enc = [64, 64, 128, 256, 512]
        
    def forward(self, x):
        features = []
        
        # Conv1: 7×7, stride 2 → [B, 64, H/2, W/2]
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x)
        
        # MaxPool + Layer1 → [B, 64, H/4, W/4]
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        features.append(x)
        
        # Layer2 → [B, 128, H/8, W/8]
        x = self.encoder.layer2(x)
        features.append(x)
        
        # Layer3 → [B, 256, H/16, W/16]
        x = self.encoder.layer3(x)
        features.append(x)
        
        # Layer4 → [B, 512, H/32, W/32]
        x = self.encoder.layer4(x)
        features.append(x)
        
        return features  # 5개 스케일의 특징 맵
```

### 3.2.3 해상도별 특징 맵

인코더는 5개 스케일의 특징 맵을 생성한다. 각 스케일은 이전 스케일보다 해상도가 절반으로 줄어들고, 채널 수는 증가한다. 이는 U-Net 스타일의 인코더-디코더 구조에서 흔히 사용되는 패턴으로, 저해상도에서는 넓은 수용 영역(receptive field)을 통해 전역적 맥락을 포착하고, 고해상도에서는 세밀한 지역적 정보를 보존한다.

입력 해상도 640×384 기준:

| 스케일 | Layer | 채널 | 해상도 | Stride | 설명 |
|--------|-------|------|--------|--------|------|
| 0 | Conv1 | 64 | 320×192 | 2 | 가장 고해상도, 엣지/텍스처 정보 |
| 1 | Layer1 | 64 | 160×96 | 4 | 기본 형태 특징 |
| 2 | Layer2 | 128 | 80×48 | 8 | 중간 수준 의미 특징 |
| 3 | Layer3 | 256 | 40×24 | 16 | 고수준 의미 특징 |
| 4 | Layer4 | 512 | 20×12 | 32 | 가장 추상적, 전역 맥락 |

---

## 3.3 SAN (Sparse Auxiliary Networks)

### 3.3.1 SAN이란?

**SAN (Sparse Auxiliary Networks)**은 본 연구에서 채택한 **학습 전략**으로, 핵심 아이디어는 다음과 같다:

> **"학습 시에는 LiDAR로 가이드하고, 추론 시에는 RGB만으로 예측한다"**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SAN (Sparse Auxiliary Networks)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SAN = Minkowski Encoder + FiLM                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         학습 시 (Training)                          │   │
│   │                                                                     │   │
│   │   Sparse LiDAR ──▶ Minkowski Encoder ──▶ FiLM (γ, β)               │   │
│   │       (~5%)           (Sparse Conv)           │                    │   │
│   │                                               ▼                    │   │
│   │   RGB Features ─────────────────────────▶ γ × x + β                │   │
│   │                                          (변조된 특징)              │   │
│   │                                                                     │   │
│   │   → LiDAR가 RGB 특징에 "깊이 스케일 정보"를 주입                     │   │
│   │   → 모델이 "이 크기의 물체 = 이 거리" 관계를 학습                    │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         추론 시 (Inference)                          │   │
│   │                                                                     │   │
│   │   RGB ──▶ ResNet18 Encoder ──▶ Decoder ──▶ Depth                   │   │
│   │                                                                     │   │
│   │   ✅ LiDAR 불필요!                                                  │   │
│   │   ✅ Minkowski Encoder 비활성화                                     │   │
│   │   ✅ 학습된 지식으로 RGB만으로 정확한 깊이 예측                       │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3.2 SAN의 핵심 가치

| 특성 | 설명 |
|------|------|
| **학습 효율성** | LiDAR가 제공하는 정확한 깊이 정보로 스케일 모호성 해결 |
| **추론 독립성** | 학습된 모델은 RGB만으로 동작 (LiDAR 센서 불필요) |
| **비용 절감** | 추론 시 고가의 LiDAR 센서 없이 배포 가능 |
| **실용성** | 다양한 환경(LiDAR 없는 차량, 모바일 기기)에서 활용 가능 |

### 3.3.3 왜 SAN이 필요한가?

**문제: 스케일 모호성 (Scale Ambiguity)**

RGB 이미지만으로는 절대적인 깊이 스케일을 알기 어렵다:
- 멀리 있는 큰 물체 vs 가까이 있는 작은 물체 → 이미지에서 같은 크기!
- Monocular depth estimation의 본질적 한계

**해결: LiDAR를 "Teacher"로 활용**

```
┌───────────────────────────────────────────────────────────────────┐
│              Teacher-Student Style Learning                       │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Teacher (LiDAR)              Student (RGB)                      │
│   ┌─────────────┐              ┌─────────────┐                   │
│   │ 정확한 깊이 │  ──guide──▶  │ 깊이 추정법 │                   │
│   │ 스케일 정보 │              │    학습     │                   │
│   └─────────────┘              └─────────────┘                   │
│         │                             │                          │
│         │                             ▼                          │
│         │                      ┌─────────────┐                   │
│    학습 후 제거               │ RGB만으로도 │                   │
│                               │ 정확한 예측 │                   │
│                               └─────────────┘                   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3.4 Minkowski Encoder (SAN 구성요소 1)

### 3.4.1 설계 배경

LiDAR 센서에서 얻은 깊이 정보는 매우 희소(sparse)하다. 일반적으로 이미지 전체 픽셀의 약 5% 미만에만 유효한 깊이 값이 존재한다. 이러한 희소 데이터를 일반적인 Dense Convolution으로 처리하면 대부분의 연산이 0에 대해 수행되어 비효율적이다.

**Minkowski Engine**은 이 문제를 해결하기 위해 설계된 희소 텐서(Sparse Tensor) 연산 라이브러리다. 유효한 좌표에서만 연산을 수행하므로 메모리와 연산량을 크게 절약할 수 있다. 본 연구에서는 Minkowski Encoder를 사용하여 LiDAR sparse depth를 효율적으로 처리하고, 이를 FiLM 파라미터로 변환하여 RGB 특징에 주입한다.

### 3.4.2 역할

Sparse Depth 입력(LiDAR)을 처리하여 FiLM 조절 파라미터(γ, β)를 생성한다.

```
┌─────────────────────────────────────────────────────────────┐
│                   Minkowski Encoder                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Sparse Depth Input                                        │
│   ┌─────────────────┐                                       │
│   │ . .   .    .    │  (~5% valid pixels)                   │
│   │    .     .      │                                       │
│   │  .    .      .  │                                       │
│   └────────┬────────┘                                       │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │  Sparsify       │  좌표 + 깊이값 추출                    │
│   │  (coords, feats)│                                       │
│   └────────┬────────┘                                       │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │ MinkConv2D ×5   │  스케일별 Sparse Conv                  │
│   │ (kernel=3)      │                                       │
│   └────────┬────────┘                                       │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │  Densify        │  Sparse → Dense 변환                  │
│   │  + γ, β 생성    │                                       │
│   └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.4.3 Sparse Convolution 상세

```python
class MinkConv2D(nn.Module):
    """
    Minkowski 2D Sparse Convolution Block
    3-layer residual structure
    """
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        
        # 3-layer structure for better feature extraction
        self.layer1 = ME.MinkowskiConvolution(in_planes, out_planes, kernel_size=3)
        self.layer2 = ME.MinkowskiConvolution(in_planes, out_planes*2, kernel_size=3)
        self.layer3 = ME.MinkowskiConvolution(in_planes, out_planes*2, kernel_size=3)
        
        self.bn = ME.MinkowskiBatchNorm(out_planes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        return self.relu(self.bn(x1 + x2 + x3))
```

---

## 3.5 FiLM Fusion (SAN 구성요소 2)

### 3.5.1 FiLM의 역할

Minkowski Encoder가 Sparse LiDAR에서 추출한 정보를 RGB 특징에 **주입하는 방법**이 FiLM이다.

LiDAR와 RGB를 단순히 concatenation하면 희소한 LiDAR 정보가 dense한 RGB 정보에 묻혀버릴 수 있다. **FiLM (Feature-wise Linear Modulation)**은 조건부 정보(LiDAR depth)를 사용하여 메인 특징(RGB)의 각 채널을 **스케일(γ)과 시프트(β)**로 변조한다.

> **"LiDAR가 RGB 특징을 어떻게 해석해야 하는지 가이드한다"**

### 3.5.2 FiLM (Feature-wise Linear Modulation)

FiLM은 조건부 정보(Sparse Depth)를 RGB 특징에 주입하는 방법이다.

```
┌─────────────────────────────────────────────────────────────┐
│                    FiLM Modulation                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   RGB Feature (x)         Sparse Feature (s)                │
│   [B, C, H, W]           [B, C, H, W]                       │
│        │                      │                             │
│        │                      ├──────────────┐              │
│        │                      │              │              │
│        │                      ▼              ▼              │
│        │                 ┌────────┐    ┌────────┐           │
│        │                 │ γ Conv │    │ β Conv │           │
│        │                 └────┬───┘    └────┬───┘           │
│        │                      │              │              │
│        ▼                      ▼              ▼              │
│   ┌─────────────────────────────────────────────┐           │
│   │                                             │           │
│   │        y = γ ⊙ x + β                       │           │
│   │                                             │           │
│   │   (element-wise multiplication + addition)  │           │
│   │                                             │           │
│   └─────────────────────────────────────────────┘           │
│                          │                                  │
│                          ▼                                  │
│                   Modulated Feature (y)                     │
│                   [B, C, H, W]                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.5.3 수식

$$
y_i = \gamma_i \cdot x_i + \beta_i
$$

- $x_i$: RGB 특징의 i번째 채널
- $\gamma_i$: Sparse Depth에서 생성된 Scale (곱해지는 값)
- $\beta_i$: Sparse Depth에서 생성된 Bias (더해지는 값)
- $y_i$: 조절된 특징

γ가 1이고 β가 0이면 원래 특징이 그대로 유지된다. γ가 크면 해당 채널이 강조되고, β가 크면 해당 채널에 새로운 정보가 추가된다. 이렇게 LiDAR 정보가 RGB 특징의 "해석 방향"을 조절한다.

### 3.5.4 적용 스케일

```python
# 기본 설정: 첫 스케일에만 FiLM 적용
film_scales = [0]  # Scale 0 (가장 고해상도)

# 확장 설정: 여러 스케일
film_scales = [0, 1, 2]  # 처음 3개 스케일
```

---

## 3.6 Training & Inference Flow

본 연구의 모델은 **학습 시**와 **추론 시** 다른 경로를 사용한다. 특히 LiDAR sparse depth는 학습에만 사용되고, 추론 시에는 RGB만으로 깊이를 예측한다.

### 3.6.1 전체 플로우 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Training Flow (학습)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Inputs:                                                                   │
│   ┌─────────┐         ┌─────────────┐         ┌─────────┐                  │
│   │  RGB    │         │ Sparse Depth│         │ GT Depth│                  │
│   │ (640×384)│         │  (LiDAR)    │         │  (Dense)│                  │
│   └────┬────┘         └──────┬──────┘         └────┬────┘                  │
│        │                     │                     │                        │
│        ▼                     ▼                     │                        │
│   ┌──────────┐       ┌─────────────┐              │                        │
│   │ ResNet18 │       │  Minkowski  │              │                        │
│   │ Encoder  │       │  Encoder    │              │                        │
│   └────┬─────┘       └──────┬──────┘              │                        │
│        │                    │                      │                        │
│        └────────┬───────────┘                      │                        │
│                 ▼                                  │                        │
│          ┌─────────────┐                           │                        │
│          │ FiLM Fusion │ (optional)                │                        │
│          └──────┬──────┘                           │                        │
│                 ▼                                  │                        │
│          ┌─────────────┐                           │                        │
│          │  Dual-Head  │                           │                        │
│          │   Decoder   │                           │                        │
│          └──────┬──────┘                           │                        │
│                 │                                  │                        │
│        ┌────────┼────────┐                         │                        │
│        ▼                 ▼                         ▼                        │
│   ┌──────────┐    ┌───────────┐            ┌─────────────┐                 │
│   │ Integer  │    │Fractional │            │  GT 분해    │                 │
│   │  σ[0,1]  │    │  σ[0,1]   │            │ (Int, Frac) │                 │
│   └────┬─────┘    └─────┬─────┘            └──────┬──────┘                 │
│        │                │                         │                        │
│        └────────┬───────┘                         │                        │
│                 │                                 │                        │
│                 ▼                                 ▼                        │
│        ┌─────────────────────────────────────────────────┐                 │
│        │              DualHeadDepthLoss                   │                 │
│        │  L_int + 10×L_frac + 0.5×L_consist               │                 │
│        └─────────────────────────────────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Inference Flow (추론)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input:                                                                    │
│   ┌─────────┐                                                              │
│   │  RGB    │    (LiDAR 불필요!)                                            │
│   │ (640×384)│                                                              │
│   └────┬────┘                                                              │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────┐                                                             │
│   │ ResNet18 │                                                             │
│   │ Encoder  │                                                             │
│   └────┬─────┘                                                             │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────┐                                                          │
│   │  Dual-Head  │                                                          │
│   │   Decoder   │                                                          │
│   └──────┬──────┘                                                          │
│          │                                                                  │
│   ┌──────┼──────┐                                                          │
│   ▼             ▼                                                          │
│ ┌────────┐  ┌───────────┐                                                  │
│ │Integer │  │Fractional │                                                  │
│ │ σ[0,1] │  │  σ[0,1]   │                                                  │
│ └───┬────┘  └─────┬─────┘                                                  │
│     │             │                                                         │
│     ▼             ▼                                                         │
│  ┌─────────────────────────────────────┐                                   │
│  │  depth = σ_int × max_depth          │                                   │
│  │        + σ_frac × 1.0               │                                   │
│  └─────────────────────────────────────┘                                   │
│                    │                                                        │
│                    ▼                                                        │
│            ┌────────────┐                                                   │
│            │ Final Depth│                                                   │
│            │ (0~31m)    │                                                   │
│            └────────────┘                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.5.2 핵심 차이점: 학습 vs 추론

| 구분 | 학습 (Training) | 추론 (Inference) |
|------|----------------|-----------------|
| **입력** | RGB + Sparse Depth + GT | RGB만 |
| **Minkowski Encoder** | 활성화 (FiLM용) | 비활성화 |
| **FiLM Fusion** | 활성화 (선택적) | 비활성화 |
| **출력** | Integer/Fractional 분리 | Integer/Fractional 합산 |
| **Loss** | DualHeadDepthLoss | 없음 |

### 3.5.3 왜 학습에만 LiDAR를 사용하는가?

이 설계의 핵심 철학은 **"학습 시에는 LiDAR로 가이드하고, 추론 시에는 RGB만으로 독립적으로 예측"**하는 것이다.

1. **학습 시**: LiDAR sparse depth가 FiLM을 통해 RGB 특징에 깊이 스케일 정보를 주입한다. 이는 모델이 "이 정도 크기의 물체는 이 정도 거리에 있다"는 관계를 학습하게 돕는다.

2. **추론 시**: 학습된 관계를 바탕으로 RGB만으로도 절대 깊이를 예측할 수 있다. 실제 배포 환경에서는 LiDAR가 없거나 고장날 수 있으므로, RGB-only 추론이 가능한 것은 큰 장점이다.

이러한 "Teacher-Student" 스타일의 학습은 LiDAR를 일종의 "지식 증류(Knowledge Distillation)" 소스로 활용한다고 볼 수 있다.

### 3.5.4 코드 레벨 플로우

```python
# ========================================
# SemiSupCompletionModel.forward() 핵심 로직
# ========================================

def forward(self, batch, return_logs=False, progress=0.0, **kwargs):
    if not self.training:
        # 추론 모드: RGB만 사용, SfmModel.forward() 호출
        return SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
    
    # 학습 모드: RGB + Sparse Depth + GT Depth 사용
    # 1. depth_net.forward(rgb, input_depth) 호출
    self_sup_output = SfmModel.forward(self, batch, ...)
    
    # 2. GT depth를 Integer/Fractional로 분해
    gt_depth = batch['depth']
    
    # 3. Dual-Head Loss 계산
    if self.depth_net.is_dual_head:
        sup_output = self.supervised_loss(
            self_sup_output,     # dict with ('integer', i), ('fractional', i)
            depth2inv(gt_depth),
            return_logs=return_logs
        )
    
    return {'loss': loss, ...}
```

```python
# ========================================
# ResNetSAN01.forward() 핵심 로직
# ========================================

def forward(self, rgb, input_depth=None, **kwargs):
    if not self.training:
        # 추론: RGB만 사용 (input_depth 무시)
        outputs, _ = self.run_network(rgb, input_depth=None)
        return outputs
    
    # 학습: RGB 단독 + RGB+Depth 두 번 forward
    # RGB-only forward
    inv_depths_rgb, skip_feat_rgb = self.run_network(rgb)
    output = inv_depths_rgb  # Dual-head dict
    
    if input_depth is None:
        return output
    
    # RGB+Depth forward (FiLM 적용)
    inv_depths_rgbd, skip_feat_rgbd = self.run_network(rgb, input_depth)
    
    # Dual-Head는 RGB-only output만 반환
    # (RGB+D는 FiLM 학습에만 사용)
    return output
```

### 3.5.5 학습 시 두 번의 Forward Pass

학습 시에는 실제로 **두 번**의 forward pass가 수행된다:

1. **RGB-only**: `run_network(rgb, input_depth=None)`
   - LiDAR 없이 RGB만으로 깊이 예측
   - 이 출력이 **최종 Loss 계산에 사용됨**

2. **RGB+Depth**: `run_network(rgb, input_depth)` (FiLM 활성화 시)
   - LiDAR를 FiLM으로 융합하여 깊이 예측
   - Feature-level consistency loss 계산에 사용
   - 모델이 LiDAR 정보를 활용하도록 유도

이 설계는 모델이 **LiDAR가 있을 때와 없을 때 모두** 잘 동작하도록 학습하는 것을 목표로 한다.

---

## 3.6 Dual-Head Decoder (핵심 기여)

### 3.6.1 설계 동기

본 연구의 핵심 기여인 Dual-Head Decoder는 **NPU INT8 양자화의 정밀도 한계**를 극복하기 위해 설계되었다.

#### 문제 상황

대부분의 NPU(Neural Processing Unit)는 **Per-tensor INT8 양자화**만 지원한다. 이는 텐서 전체에 대해 하나의 scale과 zero-point만 사용한다는 의미다. 깊이 추정에서 출력 범위가 0.5~30m라면, INT8의 256 레벨로 이 범위를 표현해야 한다:

$$
\text{양자화 간격} = \frac{30m - 0.5m}{256} \approx 115mm
$$

이는 약 ±57mm의 양자화 오차를 의미하며, 근거리(1~3m)에서 상대 오차가 매우 커진다.

#### 해결책: 정수부-소수부 분리

Dual-Head 구조에서는 깊이를 **정수부(Integer)**와 **소수부(Fractional)**로 분리한다:

- **Integer Head**: 0~30m 범위를 sigmoid [0,1]로 예측 → 정수 단위 깊이
- **Fractional Head**: 0~1m 범위를 sigmoid [0,1]로 예측 → 서브미터 정밀도

소수부는 0~1m 범위만 담당하므로:

$$
\text{소수부 양자화 간격} = \frac{1m}{256} \approx 3.9mm
$$

이는 **약 30배의 정밀도 향상**을 의미한다!

```
┌─────────────────────────────────────────────────────────────┐
│              Single-Head vs Dual-Head Comparison             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Single-Head]                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Output: [0.5m, 15.0m]                                │  │
│  │  INT8: 256 levels for 14.5m range                     │  │
│  │  Quantization interval: 56.9mm                        │  │
│  │  Error: ±28.4mm                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  [Dual-Head]                                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Integer Head: [0, 15] (meter units)                  │  │
│  │  → Sigmoid [0, 1] × 15 = [0, 15]m                     │  │
│  │  → INT8: 256 levels for 15 values (16× oversampling)  │  │
│  │                                                       │  │
│  │  Fractional Head: [0, 1.0m] (sub-meter precision)     │  │
│  │  → Sigmoid [0, 1] × 1 = [0, 1]m                       │  │
│  │  → INT8: 256 levels for 1m range                      │  │
│  │  → Quantization interval: 3.92mm                      │  │
│  │  → Error: ±1.96mm (14× improvement!)                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.5.2 Decoder 구조

디코더는 크게 두 부분으로 구성된다:

1. **공유 업샘플링 경로**: 인코더의 특징을 점진적으로 업샘플링하면서 Skip Connection과 결합
2. **분리된 출력 헤드**: 최종 특징에서 Integer와 Fractional을 각각 예측

공유 경로를 통해 대부분의 연산을 재사용하므로, Single-Head 대비 추가 연산량은 약 5% 미만이다.

```python
class DualHeadDepthDecoder(nn.Module):
    """
    Integer-Fractional Dual-Head Depth Decoder
    
    공유 Upsampling + 분리된 출력 헤드
    """
    def __init__(self, num_ch_enc, max_depth=15.0):
        super().__init__()
        
        self.max_depth = max_depth
        self.num_ch_dec = [16, 32, 64, 128, 256]
        
        # 공유 Upsampling layers (기존 DepthDecoder와 동일)
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # UpConv 0: 채널 감소
            num_ch_in = num_ch_enc[-1] if i == 4 else self.num_ch_dec[i+1]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, self.num_ch_dec[i])
            
            # UpConv 1: Skip connection 결합
            num_ch_in = self.num_ch_dec[i] + (num_ch_enc[i-1] if i > 0 else 0)
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, self.num_ch_dec[i])
        
        # Dual-Head outputs (각 스케일)
        for s in range(4):
            self.convs[("integer_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)
            self.convs[("fractional_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)
        
        self.sigmoid = nn.Sigmoid()
```

### 3.5.3 Forward Pass

```python
def forward(self, input_features):
    outputs = {}
    x = input_features[-1]  # 가장 깊은 특징부터 시작
    
    for i in range(4, -1, -1):
        # Upsample
        x = self.convs[("upconv", i, 0)](x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        # Skip connection
        if i > 0:
            x = torch.cat([x, input_features[i-1]], dim=1)
        
        x = self.convs[("upconv", i, 1)](x)
        
        # Dual-Head outputs
        if i < 4:  # scales 0, 1, 2, 3
            # Integer Head: sigmoid [0,1] → [0, max_depth]
            int_raw = self.convs[("integer_conv", i)](x)
            outputs[("integer", i)] = self.sigmoid(int_raw)
            
            # Fractional Head: sigmoid [0,1] → [0, 1]m
            frac_raw = self.convs[("fractional_conv", i)](x)
            outputs[("fractional", i)] = self.sigmoid(frac_raw)
    
    return outputs
```

### 3.5.4 깊이 복원

최종 깊이는 두 헤드의 출력을 단순히 더해서 복원한다. 이 단순한 후처리는 NPU에서도 쉽게 구현할 수 있다.

```python
def dual_head_to_depth(integer_sigmoid, fractional_sigmoid, max_depth=15.0):
    """
    Dual-Head 출력을 최종 깊이로 변환
    
    Parameters:
        integer_sigmoid: [B, 1, H, W] in [0, 1]
        fractional_sigmoid: [B, 1, H, W] in [0, 1]
        max_depth: float (default 15.0m)
    
    Returns:
        depth: [B, 1, H, W] in [0, max_depth + 1.0]m
    """
    integer_depth = integer_sigmoid * max_depth    # [0, 15]m
    fractional_depth = fractional_sigmoid * 1.0   # [0, 1]m
    
    return integer_depth + fractional_depth
```

**예시**:
```
Input: integer_sigmoid = 0.35, fractional_sigmoid = 0.72
       max_depth = 15.0

Calculation:
  integer_depth = 0.35 × 15.0 = 5.25m
  fractional_depth = 0.72 × 1.0 = 0.72m
  
Final Depth = 5.25 + 0.72 = 5.97m
```

여기서 주목할 점은 Integer Head의 출력이 **실제 정수값이 아니라** sigmoid의 연속적인 출력이라는 것이다. 이름은 "Integer"이지만, 실제로는 0~max_depth 범위의 연속값을 예측한다. "Integer"라는 이름은 이 헤드가 **미터 단위의 대략적인 깊이**를 담당한다는 의미론적 구분이다.

---

## 3.6 Loss Functions

### 3.6.1 설계 원리

Dual-Head 구조를 효과적으로 학습시키려면 두 헤드가 각자의 역할을 잘 수행하도록 유도해야 한다. 단순히 최종 깊이만 supervision하면, 모델이 두 헤드의 역할을 임의로 분배할 수 있다.

따라서 **3-component loss**를 설계했다:
1. **Integer Loss**: 정수부가 대략적인 깊이를 맞추도록
2. **Fractional Loss**: 소수부가 서브미터 디테일을 맞추도록 (높은 가중치!)
3. **Consistency Loss**: 두 헤드의 합이 GT와 일치하도록

### 3.6.2 Dual-Head Depth Loss

```python
class DualHeadDepthLoss(nn.Module):
    """
    3-component loss for Dual-Head training
    
    Components:
    1. Integer Loss (L1): 정수부 예측
    2. Fractional Loss (L1): 소수부 예측 (높은 가중치!)
    3. Consistency Loss (L1): 복원 깊이 일관성
    """
    def __init__(self, max_depth=15.0, 
                 integer_weight=1.0, 
                 fractional_weight=10.0,  # 핵심!
                 consistency_weight=0.5):
        self.max_depth = max_depth
        self.weights = {
            'integer': integer_weight,
            'fractional': fractional_weight,
            'consistency': consistency_weight
        }
```

### 3.6.2 GT 분해

```python
def decompose_depth(depth_gt, max_depth):
    """
    GT 깊이를 Integer/Fractional로 분해
    
    Parameters:
        depth_gt: [B, 1, H, W] ground truth depth
        max_depth: float
    
    Returns:
        integer_gt: [B, 1, H, W] normalized [0, 1]
        fractional_gt: [B, 1, H, W] normalized [0, 1]
    """
    # Clamp to valid range
    depth_gt = torch.clamp(depth_gt, 0, max_depth + 1.0)
    
    # Integer part (floor)
    integer_part = torch.floor(depth_gt)
    integer_gt = integer_part / max_depth  # Normalize to [0, 1]
    
    # Fractional part (remainder)
    fractional_part = depth_gt - integer_part
    fractional_gt = fractional_part  # Already in [0, 1]
    
    return integer_gt, fractional_gt
```

예를 들어 GT depth가 5.72m이고 max_depth가 15m라면:
- `integer_part = floor(5.72) = 5`
- `integer_gt = 5 / 15 = 0.333` (normalized)
- `fractional_gt = 5.72 - 5 = 0.72`

### 3.6.4 Loss 계산

$$
\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{int} + \lambda_2 \mathcal{L}_{frac} + \lambda_3 \mathcal{L}_{consist}
$$

where:
- $\mathcal{L}_{int} = \frac{1}{N}\sum|p_{int} - g_{int}|$ (Integer L1 Loss)
- $\mathcal{L}_{frac} = \frac{1}{N}\sum|p_{frac} - g_{frac}|$ (Fractional L1 Loss)
- $\mathcal{L}_{consist} = \frac{1}{N}\sum|d_{pred} - d_{gt}|$ (Consistency L1 Loss)
- $\lambda_1 = 1.0$, $\lambda_2 = 10.0$ (핵심!), $\lambda_3 = 0.5$

**가중치 설계 이유**:

Fractional 가중치를 10배로 설정한 이유는 **양자화 후 정밀도가 소수부에 의해 결정**되기 때문이다. Integer Head가 1m 정도 틀려도 Fractional Head가 정확하면 최종 깊이는 크게 벗어나지 않는다. 반면 Fractional Head가 부정확하면 INT8 양자화의 이점이 사라진다.

Consistency Loss는 두 헤드가 서로 보완하도록 유도한다. 예를 들어 실제 깊이가 4.9m인데 Integer가 5m, Fractional이 0.1m을 예측하면 합은 5.1m이 된다. Consistency Loss가 이런 불일치를 교정한다.

---

## 3.7 Model Variants

### 3.7.1 지원 모드

본 연구의 아키텍처는 모듈화되어 있어 다양한 조합이 가능하다:

| 모드 | use_dual_head | use_film | 용도 |
|------|---------------|----------|------|
| **Standard** | False | False | RGB-only 추론, 가장 빠름 |
| **FiLM** | False | True | RGB + LiDAR 학습, 스케일 보정 |
| **Dual-Head** | True | False | INT8 최적화 (본 연구의 핵심) |
| **Dual-Head + FiLM** | True | True | Full feature (향후 연구) |

### 3.7.2 Configuration

```yaml
# configs/train_resnet_san_ncdb_dual_head_640x384.yaml

model:
    depth_net:
        name: 'ResNetSAN01'
        version: '18A'          # ResNet18
        use_dual_head: true     # ⭐ Dual-Head 활성화
        use_film: false         # FiLM 비활성화 (단순화)
    
    params:
        min_depth: 0.5          # 최소 깊이
        max_depth: 15.0         # 최대 깊이 (Integer head 범위)
```

---

## 3.8 ONNX Export & NPU Deployment

### 3.8.1 배포 파이프라인 개요

학습된 PyTorch 모델을 NPU에서 실행하려면 다음 단계를 거친다:

1. **PyTorch → ONNX**: 모델을 Open Neural Network Exchange 포맷으로 변환
2. **ONNX → INT8**: Post-Training Quantization(PTQ)으로 8비트 정수 가중치로 변환
3. **INT8 → NPU Binary**: 타겟 NPU SDK로 최종 바이너리 생성

Dual-Head 구조는 이 파이프라인에서 특별한 이점을 제공한다. 두 출력 헤드가 모두 sigmoid [0,1] 범위이므로 양자화 calibration이 간단하고 안정적이다.

### 3.8.2 ONNX Export

```python
# Dual-Head ONNX export
torch.onnx.export(
    model,
    dummy_input,  # [1, 3, 384, 640]
    "dual_head_depth.onnx",
    input_names=['rgb'],
    output_names=['integer', 'fractional'],  # 2개 출력
    opset_version=11
)
```

ONNX opset 11을 사용하는 이유는 대부분의 NPU 컴파일러가 이 버전을 안정적으로 지원하기 때문이다.

### 3.8.3 NPU INT8 양자화

```
┌─────────────────────────────────────────────────────────────┐
│                 NPU INT8 Quantization                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ONNX Model                INT8 Model                      │
│   ┌─────────┐               ┌─────────┐                    │
│   │ FP32    │    ──PTQ──▶   │ INT8    │                    │
│   │ Weights │               │ Weights │                    │
│   └─────────┘               └─────────┘                    │
│                                                             │
│   Integer Head:                                             │
│     Output [0, 1] → INT8 [0, 255]                          │
│     Scale = 1/255 = 0.00392                                │
│                                                             │
│   Fractional Head:                                          │
│     Output [0, 1] → INT8 [0, 255]                          │
│     Scale = 1/255 = 0.00392                                │
│     Depth precision = 1m × 0.00392 = 3.92mm                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

PTQ(Post-Training Quantization)는 학습 없이 calibration 데이터만으로 양자화 파라미터를 결정한다. 일반적으로 100~1000장의 대표 이미지면 충분하다.

### 3.8.4 Inference 파이프라인

```python
def inference(model, rgb_image):
    """
    NPU INT8 inference pipeline
    """
    # 1. Preprocess: RGB 정규화
    rgb = preprocess(rgb_image)  # [1, 3, 384, 640]
    
    # 2. Model inference (NPU에서 실행)
    integer_out, fractional_out = model(rgb)
    
    # 3. Post-process (CPU에서 실행)
    # 두 헤드의 출력을 합산하여 최종 깊이 복원
    depth = integer_out * max_depth + fractional_out * 1.0
    
    return depth
```

후처리(Post-process)는 단순한 곱셈과 덧셈이므로 CPU에서도 무시할 수 있는 오버헤드다. 일부 NPU에서는 이 연산도 NPU에서 수행할 수 있다.

---

## 3.9 Summary

### 주요 설계 결정

| 결정 | 선택 | 이유 |
|------|------|------|
| Encoder | ResNet18 | 경량화(11M params), Edge 배포에 적합 |
| Sparse Conv | Minkowski | 희소 LiDAR 데이터의 효율적 처리 |
| Fusion | FiLM (optional) | 스케일 모호성 해결, 유연한 모달리티 결합 |
| Output | Dual-Head | INT8 양자화에서 30배 정밀도 향상 |
| Loss | 3-component | 정수부/소수부/일관성 동시 학습 |

### 핵심 기여

1. **Dual-Head Architecture**: 깊이 출력을 정수부와 소수부로 분리하여 INT8 양자화 정밀도를 30배 향상. 기존 단일 출력 구조에서 ~117mm였던 양자화 간격을 ~4mm로 줄임.

2. **Modular Design**: `use_dual_head`, `use_film` 플래그로 다양한 구성이 가능. 개발 단계에서는 FP32 Single-Head로 빠르게 실험하고, 배포 시에는 INT8 Dual-Head로 전환 가능.

3. **NPU Compatible**: Per-tensor 양자화 제약을 충족하면서도 높은 정밀도 유지. 별도의 양자화 인식 학습(QAT) 없이도 PTQ만으로 우수한 성능 달성.

