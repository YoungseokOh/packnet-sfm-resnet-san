# Attention Module 기반 모델 개선 보고서

**작성일**: 2024-12-05  
**대상 모델**: ResNetSAN01 (ResNet-18 Encoder + Dual-Head Decoder)  
**목표**: AImotive NPU 제약사항 내에서 적용 가능한 Attention 기법 분석

---

## 1. 현재 모델 아키텍처 분석

### 1.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        ResNetSAN01                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐                                               │
│   │   Input     │ RGB Image (B, 3, 384, 640)                    │
│   └──────┬──────┘                                               │
│          ↓                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              ResNet-18 Encoder (Pretrained)              │   │
│   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │   │
│   │  │ feat0  │ │ feat1  │ │ feat2  │ │ feat3  │ │ feat4  │ │   │
│   │  │ 64ch   │ │ 64ch   │ │ 128ch  │ │ 256ch  │ │ 512ch  │ │   │
│   │  │ 1/2    │ │ 1/4    │ │ 1/8    │ │ 1/16   │ │ 1/32   │ │   │
│   │  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ │   │
│   └───────┼──────────┼──────────┼──────────┼──────────┼─────┘   │
│           │          │          │          │          │         │
│           │          │          │          │          ↓         │
│   ┌───────┼──────────┼──────────┼──────────┼─────────────────┐  │
│   │       │          │          │          │   DualHead      │  │
│   │       │ Skip     │ Skip     │ Skip     │   Decoder       │  │
│   │       │ Connect  │ Connect  │ Connect  │                 │  │
│   │       ↓          ↓          ↓          ↓                 │  │
│   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            │  │
│   │  │ 16ch   │ │ 32ch   │ │ 64ch   │ │ 128ch  │←──┐        │  │
│   │  │ 1/1    │ │ 1/2    │ │ 1/4    │ │ 1/8    │   │256ch   │  │
│   │  └────┬───┘ └────────┘ └────────┘ └────────┘   │1/16    │  │
│   │       │                                         │        │  │
│   │       ↓                                         │        │  │
│   │  ┌─────────────────────────────────────────────┘        │  │
│   │  │ Dual Output Heads                                    │  │
│   │  │  ├── Integer Head:    [B, 1, H, W] sigmoid [0,1]     │  │
│   │  │  └── Fractional Head: [B, 1, H, W] sigmoid [0,1]     │  │
│   │  └──────────────────────────────────────────────────────┘  │
│   └─────────────────────────────────────────────────────────────┘
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Final Depth = Integer_sigmoid × max_depth + Fractional_sigmoid × 1.0
```

### 1.2 현재 문제점 분석

| 구성 요소 | 현재 상태 | 문제점 |
|----------|----------|--------|
| **Encoder** | ResNet-18 (Local convolutions only) | 전역적 컨텍스트 부재 |
| **Skip Connection** | 단순 Concatenation | 정보 선택 능력 없음 |
| **Decoder** | ConvBlock + Upsample | 먼 거리 의존성 모델링 불가 |
| **Feature Fusion** | 동일 비중 결합 | Adaptive weighting 없음 |

### 1.3 Depth Estimation에서 Attention이 필요한 이유

1. **장거리 의존성 (Long-range Dependencies)**
   - 도로 장면에서 소실점(vanishing point)과 근경의 관계 파악 필요
   - CNN의 제한된 receptive field로는 불충분

2. **다중 스케일 컨텍스트 (Multi-scale Context)**
   - 가까운 물체(상세 texture) vs 먼 물체(전역 구조)
   - 스케일별로 다른 특징이 중요

3. **경계 선명도 (Edge Sharpness)**
   - 깊이 불연속점에서 정확한 예측 필요
   - Attention으로 경계 영역에 집중 가능

---

## 2. 적용 가능한 Attention 기법 분석

### 2.1 Self-Attention (Full)

```python
# Standard Self-Attention
# Complexity: O(H×W × H×W × C) = O(N² × C)

Q = Conv1x1(x)  # [B, C, H, W] → [B, C', H, W]
K = Conv1x1(x)  # [B, C, H, W] → [B, C', H, W]
V = Conv1x1(x)  # [B, C, H, W] → [B, C', H, W]

# Reshape: [B, C', HW]
attention = softmax(Q^T × K / sqrt(d))  # [B, HW, HW]  ← 문제!
output = attention × V
```

**⚠️ AImotive NPU 제약사항 위반:**
- `Reshape` 미지원 → [B, C, H, W] → [B, C, HW] 불가
- `MatMul` with spatial dimensions 미지원
- 메모리: 640×384 = 245,760 → Attention map 60GB 필요

**❌ 적용 불가**

---

### 2.2 Squeeze-and-Excitation (SE) Block - ✅ 권장

```
┌─────────────────────────────────────────────────────────────┐
│                    SE Block (NPU Compatible)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input: [B, C, H, W]                                       │
│           │                                                  │
│           ↓                                                  │
│   ┌───────────────────┐                                     │
│   │ GlobalAvgPool     │  [B, C, H, W] → [B, C, 1, 1]        │
│   │ (ReduceMean)      │  ✅ NPU 지원                        │
│   └─────────┬─────────┘                                     │
│             ↓                                                │
│   ┌───────────────────┐                                     │
│   │ Conv1x1 (Reduce)  │  [B, C, 1, 1] → [B, C/r, 1, 1]     │
│   │ + ReLU            │  ✅ NPU 지원                        │
│   └─────────┬─────────┘                                     │
│             ↓                                                │
│   ┌───────────────────┐                                     │
│   │ Conv1x1 (Expand)  │  [B, C/r, 1, 1] → [B, C, 1, 1]     │
│   │ + Sigmoid         │  ✅ NPU 지원                        │
│   └─────────┬─────────┘                                     │
│             ↓                                                │
│   ┌───────────────────┐                                     │
│   │ Mul (Channel-wise)│  Input × Scale = Output            │
│   └─────────┬─────────┘  ✅ NPU 지원                        │
│             ↓                                                │
│   Output: [B, C, H, W]                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**구현 코드:**
```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (AImotive NPU Compatible)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # GlobalAvgPool ✅
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)  # ✅
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)  # ✅
        self.relu = nn.ReLU(inplace=True)  # ✅
        self.sigmoid = nn.Sigmoid()  # ✅
    
    def forward(self, x):
        scale = self.pool(x)
        scale = self.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale  # Element-wise Mul ✅
```

**NPU 호환성 검증:**
| 연산 | ONNX Operation | AImotive 지원 |
|-----|---------------|--------------|
| GlobalAvgPool | GlobalAveragePool | ✅ 지원 |
| Conv 1×1 | Conv (kernel=1) | ✅ 지원 |
| ReLU | Relu | ✅ 지원 |
| Sigmoid | Sigmoid | ✅ 지원 |
| Channel-wise Mul | Mul | ✅ 지원 |

**효과:**
- **채널 간 관계** 학습 (어떤 feature가 중요한지)
- 추가 파라미터: ~0.1% (negligible)
- FLOPs 증가: ~1%

---

### 2.3 CBAM (Convolutional Block Attention Module) - ✅ 부분 적용 권장

```
┌────────────────────────────────────────────────────────────────┐
│                         CBAM Module                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input: [B, C, H, W]                                          │
│           │                                                     │
│           ↓                                                     │
│   ┌───────────────────────────────────────────┐                │
│   │         Channel Attention (SE-like)        │                │
│   │                                            │                │
│   │  ┌─────────────┐    ┌─────────────┐       │                │
│   │  │ AvgPool(HW) │    │ MaxPool(HW) │       │                │
│   │  └──────┬──────┘    └──────┬──────┘       │                │
│   │         ↓                   ↓              │                │
│   │  ┌─────────────────────────────────┐      │                │
│   │  │     Shared MLP (FC→ReLU→FC)     │      │                │
│   │  └──────┬──────────────────┬───────┘      │                │
│   │         │                  │              │                │
│   │         └───────┬──────────┘              │                │
│   │                 ↓                          │                │
│   │            Add + Sigmoid                   │                │
│   │                 ↓                          │                │
│   │         Channel Attention Map              │                │
│   └─────────────────┬─────────────────────────┘                │
│                     ↓                                           │
│   ┌───────────────────────────────────────────┐                │
│   │         Spatial Attention                  │                │
│   │                                            │                │
│   │  ┌─────────────┐    ┌─────────────┐       │                │
│   │  │ AvgPool(C)  │    │ MaxPool(C)  │       │                │
│   │  │ [B,1,H,W]   │    │ [B,1,H,W]   │       │                │
│   │  └──────┬──────┘    └──────┬──────┘       │                │
│   │         └──────┬───────────┘              │                │
│   │                ↓ Concat                    │                │
│   │         [B, 2, H, W]                       │                │
│   │                ↓                           │                │
│   │         Conv 7×7 → Sigmoid                 │                │
│   │                ↓                           │                │
│   │         Spatial Attention Map              │                │
│   └─────────────────┬─────────────────────────┘                │
│                     ↓                                           │
│   Output: [B, C, H, W]                                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**NPU 호환성:**

| 부분 | 연산 | NPU 지원 | 비고 |
|-----|------|---------|------|
| Channel Attention | GlobalAvgPool, GlobalMaxPool, Conv1×1, Sigmoid | ✅ | 완전 호환 |
| Spatial Attention | ReduceMax/Mean on Channels | ⚠️ | Axis=1만 지원, 채널 256 제한 |
| Spatial Attention | Conv 7×7 | ✅ | kernel ≤ 17 |

**권장 구현:**
```python
class NPUCompatibleCBAM(nn.Module):
    """CBAM with NPU-friendly Spatial Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention (SE-like) ✅
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_max = nn.AdaptiveMaxPool2d(1)
        self.ca_fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.ca_fc2 = nn.Conv2d(channels // reduction, channels, 1)
        
        # Spatial Attention ✅ (채널 수 제한 준수)
        # 주의: 입력 채널이 256 초과 시 ReduceMax 미지원
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.ca_fc2(self.relu(self.ca_fc1(self.ca_avg(x))))
        max_out = self.ca_fc2(self.relu(self.ca_fc1(self.ca_max(x))))
        ca = self.sigmoid(avg_out + max_out)
        x = x * ca
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)  # ReduceMean ✅
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # ReduceMax ⚠️
        sa = self.sigmoid(self.sa_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * sa
```

**⚠️ 주의사항:**
- Spatial Attention의 `ReduceMax`는 채널 ≤ 256에서만 NPU 가속
- 512 채널 (feat4)에서는 **Channel Attention만 사용** 권장

---

### 2.4 Efficient Attention (Linear Attention) - ⚠️ 제한적 적용

```
Standard Attention: O(N²)
   Attention = softmax(Q × K^T) × V

Linear Attention: O(N)
   Attention = ϕ(Q) × (ϕ(K)^T × V)
   
   where ϕ is a kernel function (e.g., elu(x) + 1)
```

**문제점:**
- `MatMul`이 spatial dimension에서 필요 → AImotive 미지원
- Reshape 필요 → 미지원

**❌ 직접 적용 불가**

---

### 2.5 Axial Attention (1D Factorized) - ⚠️ 부분 적용 가능

```
┌─────────────────────────────────────────────────────────────┐
│                      Axial Attention                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   대신 1D Convolution으로 근사:                              │
│                                                              │
│   ┌────────────────┐                                        │
│   │ Input [B,C,H,W]│                                        │
│   └───────┬────────┘                                        │
│           │                                                  │
│           ├─────────────────────────────────┐               │
│           ↓                                 ↓                │
│   ┌───────────────┐                 ┌───────────────┐       │
│   │ Conv (1, k_h) │ Height-wise     │ Conv (k_w, 1) │ Width │
│   │ 큰 커널 사용   │                 │ 큰 커널 사용   │       │
│   └───────┬───────┘                 └───────┬───────┘       │
│           │                                 │                │
│           └─────────────┬───────────────────┘               │
│                         ↓                                    │
│                    Add or Concat                            │
│                         ↓                                    │
│   ┌────────────────────────────────────────┐                │
│   │        Output [B, C, H, W]              │                │
│   └────────────────────────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**NPU 호환 구현:**
```python
class AxialConvBlock(nn.Module):
    """Axial Attention approximation using 1D Convolutions"""
    def __init__(self, channels, kernel_size=17):
        super().__init__()
        # Height-wise: (1, k) kernel
        self.conv_h = nn.Conv2d(
            channels, channels, 
            kernel_size=(kernel_size, 1),  # ✅ max 17
            padding=(kernel_size // 2, 0),
            groups=channels  # Depthwise for efficiency
        )
        # Width-wise: (k, 1) kernel  
        self.conv_w = nn.Conv2d(
            channels, channels,
            kernel_size=(1, kernel_size),  # ✅ max 17
            padding=(0, kernel_size // 2),
            groups=channels
        )
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        h_attn = self.conv_h(x)
        w_attn = self.conv_w(x)
        return self.relu(self.norm(h_attn + w_attn + x))
```

**장점:**
- 큰 receptive field (17×17 → 33×33 효과)
- NPU 완전 호환
- 파라미터 증가 최소

---

### 2.6 ECA (Efficient Channel Attention) - ✅ 강력 권장

```
┌─────────────────────────────────────────────────────────────┐
│              ECA (Efficient Channel Attention)               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   SE Block의 개선 버전 - FC 대신 1D Conv 사용                │
│                                                              │
│   Input: [B, C, H, W]                                       │
│           │                                                  │
│           ↓                                                  │
│   ┌───────────────────┐                                     │
│   │ GlobalAvgPool     │  [B, C, 1, 1]                       │
│   └─────────┬─────────┘                                     │
│             ↓                                                │
│   ┌───────────────────┐                                     │
│   │ Squeeze [B, C]    │  Unsqueeze 후                       │
│   │ → Conv1D (k=3~5)  │  인접 채널 관계 학습                │
│   │ → Sigmoid         │                                     │
│   └─────────┬─────────┘                                     │
│             ↓                                                │
│   ┌───────────────────┐                                     │
│   │ Expand + Mul      │  원본과 곱                          │
│   └─────────┬─────────┘                                     │
│             ↓                                                │
│   Output: [B, C, H, W]                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**⚠️ NPU 제약:**
- Conv1D는 직접 지원되지 않음
- **대안**: Conv2D (kernel=1×k)로 구현

```python
class ECABlock(nn.Module):
    """ECA using Conv2D (NPU Compatible)"""
    def __init__(self, channels, k_size=5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Conv2d with kernel (1, k) simulates 1D conv
        self.conv = nn.Conv2d(
            1, 1, 
            kernel_size=(1, k_size), 
            padding=(0, k_size // 2),
            bias=False
        )  # ✅ NPU 지원
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        B, C, H, W = x.shape
        # [B, C, 1, 1] → [B, 1, 1, C]
        y = self.pool(x).view(B, 1, 1, C)
        # Conv2d acts as 1D conv on channel dimension
        y = self.conv(y)
        y = self.sigmoid(y).view(B, C, 1, 1)
        return x * y.expand_as(x)
```

**주의:** `view/reshape` 연산이 필요 → NPU에서 미지원될 수 있음
→ **SE Block이 더 안전한 선택**

---

## 3. 적용 위치별 권장 사항

### 3.1 모델 구조상 Attention 적용 위치

```
┌────────────────────────────────────────────────────────────────┐
│                    권장 Attention 적용 위치                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   RGB Input                                                    │
│       │                                                         │
│       ↓                                                         │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    ResNet-18 Encoder                     │  │
│   │                                                          │  │
│   │   Layer1 (64ch)  → [1️⃣ SE Block 선택적]                  │  │
│   │        │                                                 │  │
│   │   Layer2 (128ch) → [1️⃣ SE Block 선택적]                  │  │
│   │        │                                                 │  │
│   │   Layer3 (256ch) → [2️⃣ CBAM (Channel + Spatial)]        │  │
│   │        │              ← 가장 효과적인 위치                │  │
│   │   Layer4 (512ch) → [2️⃣ SE Block Only]                   │  │
│   │                       ← Spatial 제외 (채널 > 256)        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│              Skip Connections│                                  │
│              ┌───────────────┼───────────────┐                 │
│              │               │               │                  │
│              ↓               ↓               ↓                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Dual-Head Decoder                     │  │
│   │                                                          │  │
│   │   UpConv4 (256ch) ─┬─ [3️⃣ SE Block]                      │  │
│   │        │           │                                     │  │
│   │   UpConv3 (128ch) ─┴─ [3️⃣ SE Block]                      │  │
│   │        │              ← Skip fusion 직후                  │  │
│   │   UpConv2 (64ch)  → [3️⃣ Axial Conv 선택적]               │  │
│   │        │              ← 고해상도에서 큰 receptive field   │  │
│   │   UpConv1 (32ch)  → [4️⃣ Skip - 최종 출력 직전]           │  │
│   │        │                                                 │  │
│   │   UpConv0 (16ch)  → Integer + Fractional Heads          │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 우선순위별 적용 전략

| 우선순위 | 위치 | 적용 모듈 | 예상 효과 | 연산 증가 |
|---------|------|----------|----------|----------|
| **🥇 1순위** | Encoder Layer3 (256ch) | CBAM | 중간 레벨 feature 강화 | ~3% |
| **🥇 1순위** | Decoder Skip Fusion | SE Block | 중요 채널 선택 | ~1% |
| **🥈 2순위** | Encoder Layer4 (512ch) | SE Block | 고수준 semantic 강화 | ~0.5% |
| **🥈 2순위** | Decoder UpConv2 | Axial Conv | 경계 선명도 향상 | ~2% |
| **🥉 3순위** | Encoder Layer1-2 | SE Block | 저수준 feature 강화 | ~0.5% |

---

## 4. 구현 권장 코드

### 4.1 NPU-Compatible SE Block

```python
class NPUSEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block optimized for AImotive NPU
    
    All operations are NPU-compatible:
    - GlobalAveragePool ✅
    - Conv2d 1×1 ✅
    - ReLU ✅
    - Sigmoid ✅
    - Element-wise Mul ✅
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)  # 최소 8채널
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)  # [B, C, 1, 1]
        scale = self.fc(scale)  # [B, C, 1, 1]
        return x * scale  # Broadcast mul
```

### 4.2 NPU-Compatible CBAM (Channel Attention Only for 512ch)

```python
class NPUCBAMBlock(nn.Module):
    """
    CBAM with NPU constraints consideration
    - Spatial Attention disabled for channels > 256
    """
    def __init__(self, channels: int, reduction: int = 16, 
                 use_spatial: bool = True):
        super().__init__()
        self.channels = channels
        self.use_spatial = use_spatial and (channels <= 256)
        
        # Channel Attention (always enabled)
        reduced_channels = max(channels // reduction, 8)
        self.ca_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_pool_max = nn.AdaptiveMaxPool2d(1)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        
        # Spatial Attention (conditional)
        if self.use_spatial:
            self.sa_conv = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid()
            )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        ca_avg = self.ca_mlp(self.ca_pool_avg(x))
        ca_max = self.ca_mlp(self.ca_pool_max(x))
        ca = self.sigmoid(ca_avg + ca_max)
        x = x * ca
        
        # Spatial Attention (if channels <= 256)
        if self.use_spatial:
            sa_avg = torch.mean(x, dim=1, keepdim=True)
            sa_max, _ = torch.max(x, dim=1, keepdim=True)
            sa = self.sa_conv(torch.cat([sa_avg, sa_max], dim=1))
            x = x * sa
        
        return x
```

### 4.3 Skip Connection에 Attention 적용

```python
class AttentiveSkipFusion(nn.Module):
    """
    Attention-weighted skip connection fusion
    학습 가능한 attention으로 encoder/decoder 특징 선택적 결합
    """
    def __init__(self, enc_channels: int, dec_channels: int):
        super().__init__()
        total_channels = enc_channels + dec_channels
        
        # SE-style attention for fused features
        self.attention = NPUSEBlock(total_channels, reduction=8)
        
        # 1x1 conv to match channels
        self.conv = nn.Conv2d(total_channels, dec_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(dec_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        # Concatenate
        fused = torch.cat([enc_feat, dec_feat], dim=1)
        
        # Apply channel attention
        fused = self.attention(fused)
        
        # Reduce channels
        out = self.relu(self.bn(self.conv(fused)))
        return out
```

---

## 5. 수정된 Decoder 구조 제안

### 5.1 Attention-Enhanced Dual-Head Decoder

```python
class AttentionDualHeadDecoder(nn.Module):
    """
    Dual-Head Decoder with SE Attention at skip connections
    """
    def __init__(self, num_ch_enc, scales=range(4), max_depth=15.0):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.max_depth = max_depth
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        self.convs = OrderedDict()
        
        for i in range(4, -1, -1):
            # UpConv 0
            num_ch_in = num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # UpConv 1 with SE attention after skip connection
            num_ch_in = num_ch_out
            if i > 0:
                num_ch_in += num_ch_enc[i - 1]
            
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            
            # 🆕 SE Attention after fusion (i > 0에서만)
            if i > 0:
                self.convs[("se_block", i)] = NPUSEBlock(num_ch_out)
        
        # Dual Heads (unchanged)
        for s in scales:
            self.convs[("integer_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)
            self.convs[("fractional_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)
        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            
            if i > 0:
                x += [input_features[i - 1]]
            
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            
            # 🆕 Apply SE attention after skip fusion
            if i > 0:
                x = self.convs[("se_block", i)](x)
            
            if i in self.scales:
                outputs[("integer", i)] = self.sigmoid(
                    self.convs[("integer_conv", i)](x))
                outputs[("fractional", i)] = self.sigmoid(
                    self.convs[("fractional_conv", i)](x))
        
        return outputs
```

---

## 6. 실험 계획

### 6.1 단계별 적용 및 평가

| 단계 | 적용 내용 | 평가 지표 | 예상 결과 |
|-----|----------|----------|----------|
| **Baseline** | 현재 모델 | Abs Rel, RMSE, δ<1.25 | 기준 |
| **Stage 1** | Decoder SE Block | 위 지표 + Latency | Δ < 5ms, δ↑1-2% |
| **Stage 2** | Encoder L3 CBAM | 위 지표 + Latency | 경계 정확도 개선 |
| **Stage 3** | Axial Conv (선택) | 위 지표 + 메모리 | 원거리 정확도 개선 |

### 6.2 NPU 배포 검증 체크리스트

- [ ] ONNX 변환 성공 여부
- [ ] Reshape 연산 없음 확인
- [ ] 모든 Conv kernel ≤ 17
- [ ] 출력 채널 8의 배수 확인
- [ ] AImotive Compiler 통과

---

## 7. 결론 및 권장사항

### 7.1 즉시 적용 권장 (Low Risk, High Return)

1. **SE Block on Decoder Skip Connections**
   - 구현 난이도: ⭐ (쉬움)
   - NPU 호환성: ✅ 완전 호환
   - 예상 개선: 1-3% accuracy, <5% latency 증가

2. **CBAM on Encoder Layer3 (256ch)**
   - 구현 난이도: ⭐⭐ (보통)
   - NPU 호환성: ✅ 완전 호환
   - 예상 개선: 2-4% accuracy (특히 중거리)

### 7.2 추가 검토 필요 (Medium Risk)

3. **Axial Conv for Large Receptive Field**
   - 큰 커널 (17×1, 1×17)로 원거리 의존성 개선
   - 추가 실험 필요

### 7.3 적용 불가 (NPU 제약)

- ❌ Full Self-Attention (Reshape, MatMul 미지원)
- ❌ Transformer Block (위와 동일)
- ❌ Cross-Attention (위와 동일)
- ❌ Deformable Convolution (미지원)

---

## 8. 참고 문헌

1. Hu, J., et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
2. Woo, S., et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
3. Wang, Q., et al. "ECA-Net: Efficient Channel Attention" (CVPR 2020)
4. Ho, J., et al. "Axial Attention in Multidimensional Transformers" (2019)
5. Ranftl, R., et al. "Vision Transformers for Dense Prediction" (ICCV 2021)

---

**작성자**: AI Analysis System  
**검토 필요**: 실제 NPU 테스트 결과 반영  
**다음 단계**: SE Block 구현 및 학습 실험
