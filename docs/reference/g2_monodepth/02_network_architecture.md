# G2-MonoDepth 네트워크 아키텍처 상세

## 1. 전체 구조 개요

G2-MonoDepth는 **7-Layer UNet** 구조를 사용하며, **ReZero** 기법을 적용한 BottleNeck 블록으로 구성됩니다.

### 1.1 아키텍처 다이어그램

```
Input: [B, 5, H, W]
         │
         ↓
┌─────────────────────────────────────────────────────┐
│                    ENCODER                          │
├─────────────────────────────────────────────────────┤
│ Layer 0: FirstModule(5 → 64)                        │
│          ├─ Conv3×3 (stride=1)                      │
│          └─ LayerNorm + GELU                        │
│                    │                                │
│ Layer 1: UNetModule(64 → 128, stride=2)             │
│                    │                                │
│ Layer 2: UNetModule(128 → 256, stride=2)            │
│                    │                                │
│ Layer 3: UNetModule(256 → 512, stride=2)            │
│                    │                                │
│ Layer 4: UNetModule(512 → 512, stride=2)            │
│                    │                                │
│ Layer 5: UNetModule(512 → 512, stride=2)            │
│                    │                                │
│ Layer 6: UNetModule(512 → 512, stride=2) ← Bottleneck│
└─────────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────┐
│                    DECODER                          │
├─────────────────────────────────────────────────────┤
│ Layer 6: UNetModule(512 → 512, upsample=2)          │
│          + Skip Connection from Encoder Layer 5     │
│                    │                                │
│ Layer 5: UNetModule(512 → 512, upsample=2)          │
│          + Skip Connection from Encoder Layer 4     │
│                    │                                │
│ Layer 4: UNetModule(512 → 512, upsample=2)          │
│          + Skip Connection from Encoder Layer 3     │
│                    │                                │
│ Layer 3: UNetModule(512 → 256, upsample=2)          │
│          + Skip Connection from Encoder Layer 2     │
│                    │                                │
│ Layer 2: UNetModule(256 → 128, upsample=2)          │
│          + Skip Connection from Encoder Layer 1     │
│                    │                                │
│ Layer 1: UNetModule(128 → 64, upsample=2)           │
│          + Skip Connection from Encoder Layer 0     │
└─────────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────┐
│                  OUTPUT HEAD                        │
├─────────────────────────────────────────────────────┤
│ Conv3×3(64 → 1) + ReLU                              │
└─────────────────────────────────────────────────────┘
         │
         ↓
Output: [B, 1, H, W]
```

---

## 2. 핵심 모듈 상세

### 2.1 UNet 클래스 (networks.py)

```python
class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, layers=7, features=64):
        """
        Args:
            in_channels: 입력 채널 수 (기본 5: RGB + point + hole_point)
            out_channels: 출력 채널 수 (기본 1: depth)
            layers: UNet 레이어 수 (기본 7)
            features: 첫 번째 레이어의 feature 수 (기본 64)
        """
```

#### 채널 구성
| Layer | Encoder Channels | Decoder Channels |
|-------|------------------|------------------|
| 0 | 5 → 64 | 64 |
| 1 | 64 → 128 | 128 → 64 |
| 2 | 128 → 256 | 256 → 128 |
| 3 | 256 → 512 | 512 → 256 |
| 4 | 512 → 512 | 512 → 512 |
| 5 | 512 → 512 | 512 → 512 |
| 6 | 512 → 512 | 512 → 512 |

### 2.2 FirstModule (modules.py)

첫 번째 레이어는 단순 Conv + Norm + Activation:

```python
class FirstModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        return self.act(x)
```

### 2.3 UNetModule (modules.py)

Encoder/Decoder의 기본 빌딩 블록:

```python
class UNetModule(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 stride=1, upsample=1, num_bottle=3):
        """
        Args:
            stride: Encoder에서 downsampling 비율
            upsample: Decoder에서 upsampling 비율
            num_bottle: StackedBottleNeck 개수 (기본 3)
        """
```

#### 구조
```
Input
  │
  ├─[stride > 1]─→ Conv(stride=stride) ─→ StackedBottleNeck
  │
  ├─[upsample > 1]─→ Upsample ─→ Conv(1×1) ─→ StackedBottleNeck
  │
  └─[else]─→ Conv(3×3) ─→ StackedBottleNeck
  │
Output
```

### 2.4 StackedBottleNeck (modules.py)

여러 개의 BottleNeck을 직렬로 연결:

```python
class StackedBottleNeck(nn.Module):
    def __init__(self, in_channels, num_bottle=3):
        self.blocks = nn.ModuleList([
            BottleNeck(in_channels) for _ in range(num_bottle)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

---

## 3. BottleNeck with ReZero (custom_blocks.py)

### 3.1 ReZero 기법

**ReZero**는 residual connection의 초기 학습을 안정화하는 기법입니다.

#### 기존 Residual Block:
```python
output = x + F(x)  # F(x)는 랜덤 초기화
```

#### ReZero Residual Block:
```python
output = x + alpha * F(x)  # alpha는 0으로 초기화
```

- 학습 초기에 `alpha = 0`이므로 identity mapping (`output = x`)
- 학습이 진행됨에 따라 `alpha`가 점진적으로 증가
- 더 깊은 네트워크의 안정적인 학습 가능

### 3.2 BottleNeck 구현

```python
class BottleNeck(nn.Module):
    def __init__(self, in_channels, 
                 expansion_ratio=2, 
                 use_act=True, 
                 rezero=True):
        """
        Args:
            in_channels: 입력/출력 채널 수
            expansion_ratio: 중간 채널 확장 비율 (기본 2)
            use_act: 최종 activation 사용 여부
            rezero: ReZero 기법 사용 여부
        """
        mid_channels = in_channels * expansion_ratio
        
        # Main branch
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.norm1 = nn.LayerNorm(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels)
        self.norm2 = nn.LayerNorm(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, in_channels, 1)
        self.norm3 = nn.LayerNorm(in_channels)
        
        # ReZero parameter
        self.alpha = nn.Parameter(torch.zeros(1)) if rezero else 1.0
        
        self.act = nn.GELU() if use_act else nn.Identity()
```

#### Forward 연산:
```python
def forward(self, x):
    residual = x
    
    # Main branch
    out = self.conv1(x)
    out = self.norm1(out.permute(0,2,3,1)).permute(0,3,1,2)
    out = F.gelu(out)
    
    out = self.conv2(out)
    out = self.norm2(out.permute(0,2,3,1)).permute(0,3,1,2)
    out = F.gelu(out)
    
    out = self.conv3(out)
    out = self.norm3(out.permute(0,2,3,1)).permute(0,3,1,2)
    
    # ReZero: residual + alpha * F(x)
    out = residual + self.alpha * out
    
    return self.act(out)
```

### 3.3 BottleNeck 구조 다이어그램

```
Input: [B, C, H, W]
         │
         ├──────────────────────────────────┐ (skip connection)
         │                                  │
         ↓                                  │
    Conv 1×1 (C → 2C)                       │
         │                                  │
    LayerNorm + GELU                        │
         │                                  │
    DepthwiseConv 3×3 (2C → 2C)             │
         │                                  │
    LayerNorm + GELU                        │
         │                                  │
    Conv 1×1 (2C → C)                       │
         │                                  │
    LayerNorm                               │
         │                                  │
         ↓                                  │
       × alpha ←─ (learnable, init=0)       │
         │                                  │
         ↓                                  │
       + ←──────────────────────────────────┘
         │
         ↓
       GELU
         │
Output: [B, C, H, W]
```

---

## 4. 설계 선택의 이유

### 4.1 LayerNorm vs BatchNorm

| 특성 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 정규화 차원 | Batch | Channel |
| Batch Size 의존성 | 높음 | 없음 |
| 추론 시 동작 | 다름 | 동일 |
| 작은 Batch | 불안정 | 안정적 |

**LayerNorm 선택 이유**:
- DDP 학습에서 GPU별 batch size가 작을 수 있음
- 추론 시 training과 동일한 동작 보장

### 4.2 GELU vs ReLU

| 특성 | ReLU | GELU |
|------|------|------|
| 음수 영역 | 완전히 0 | 부드러운 감쇠 |
| 미분 불연속점 | 있음 (x=0) | 없음 |
| 학습 안정성 | 보통 | 더 안정적 |

### 4.3 Depthwise Separable Convolution

BottleNeck의 3×3 conv는 **depthwise conv**:
```python
nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels)
```

**장점**:
- 파라미터 수 감소: `C×C×3×3` → `C×1×3×3`
- 계산량 감소: 약 1/C 배
- 채널 간 정보 혼합은 1×1 conv가 담당

---

## 5. Skip Connection 처리

### 5.1 Decoder에서의 Skip Connection

```python
# Decoder forward
def forward_decoder(self, features, skip_connections):
    x = features
    for i, (decoder, skip) in enumerate(zip(self.decoders, skip_connections)):
        x = decoder(x)
        x = x + skip  # Element-wise addition
    return x
```

### 5.2 Skip Connection 의미

| Encoder Layer | 정보 | Decoder Layer |
|---------------|------|---------------|
| Layer 0 (64ch) | 저수준 특징 (edge, texture) | Layer 1 |
| Layer 1 (128ch) | | Layer 2 |
| Layer 2 (256ch) | 중수준 특징 (object parts) | Layer 3 |
| Layer 3 (512ch) | | Layer 4 |
| Layer 4-5 (512ch) | 고수준 특징 (semantic) | Layer 5-6 |

---

## 6. 모델 파라미터

### 6.1 파라미터 수 (layers=7, features=64)

| 부분 | 파라미터 수 (추정) |
|------|-------------------|
| Encoder | ~15M |
| Decoder | ~15M |
| Output Head | ~0.6K |
| **Total** | **~30M** |

### 6.2 메모리 사용량 (batch=8, 640×480)

| 단계 | GPU 메모리 |
|------|-----------|
| Forward | ~4GB |
| Backward | ~8GB |
| **Total** | **~12GB** |

---

## 7. 우리 프로젝트와의 비교

| 항목 | G2-MonoDepth | PackNet-SfM (ResNetSAN) |
|------|--------------|-------------------------|
| 아키텍처 | 7-Layer UNet | ResNet18 Encoder + Decoder |
| Normalization | LayerNorm | BatchNorm |
| Activation | GELU | ELU |
| Residual 기법 | ReZero | Standard |
| Skip Connection | Addition | Addition |
| 출력 Head | Conv + ReLU | Sigmoid × max_depth |

### 7.1 적용 가능한 요소

1. **ReZero 기법**:
   - ResNetSAN의 residual block에 적용 가능
   - 학습 안정성 향상 기대

2. **LayerNorm**:
   - 작은 batch size에서의 안정성
   - 단, ResNet 구조와의 호환성 검토 필요

3. **GELU Activation**:
   - 기존 ELU 대비 부드러운 특성
   - 직접 교체 가능

---

## 다음 문서

- [03_loss_functions.md](03_loss_functions.md) - Loss 함수 상세
