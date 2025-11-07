# 1. 전략 개요 및 코드베이스 분석

## 1.1. Phase 1 결과 분석

**Phase 1 (Advanced PTQ Calibration) 결과**:

| Metric | 100 samples | 300 samples | 목표 | 달성 여부 |
|--------|-------------|-------------|------|----------|
| **abs_rel** | 0.1133 | 0.1139 | < 0.09 | ❌ 실패 |
| **rmse** | 0.741m | 0.751m | - | ❌ 악화 |
| **δ<1.25** | 0.9239 | 0.9061 | - | ❌ 악화 |

**핵심 발견**:
- Calibration 이미지를 100 → 300으로 확장했으나 **성능 개선 없음**
- 오히려 일부 메트릭이 악화됨
- **결론**: 데이터셋 최적화만으로는 목표 달성 불가능 → **모델 구조 변경 필수**

---

## 1.2. 현재 코드베이스 구조 분석

**✅ 코드베이스의 설계 패턴 (이미 Parameter-driven)**:

```python
# packnet_sfm/networks/depth/ResNetSAN01.py (현재 구조)
class ResNetSAN01(nn.Module):
    def __init__(self, ..., use_film=False, use_enhanced_lidar=False, **kwargs):
        # Encoder는 공통
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=True)
        
        # Decoder는 단일 (확장 예정)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        
        # Optional features (조건부 활성화)
        if use_film:
            if use_enhanced_lidar:
                self.mconvs = EnhancedMinkowskiEncoder(...)  # Enhanced
            else:
                self.mconvs = MinkowskiEncoder(...)  # Standard
        else:
            self.mconvs = None  # Inference-only
```

**핵심 발견**:
1. ✅ **이미 Decoder 교체 패턴 존재**: `DepthDecoder`, `RaySurfaceDecoder`, `YOLOv8DepthDecoder`
2. ✅ **조건부 기능 활성화**: `use_film`, `use_enhanced_lidar` 등
3. ✅ **YAML 기반 설정**: `configs/train_resnet_san_ncdb_640x384.yaml`

**기존 유사 패턴**:
```python
# packnet_sfm/networks/depth/RaySurfaceResNet.py (참고 예시)
class RaySurfaceResNet(nn.Module):
    def __init__(self, ...):
        self.encoder = ResnetEncoder(...)
        self.decoder = DepthDecoder(...)        # Standard decoder
        self.ray_surf = RaySurfaceDecoder(...)  # Additional decoder
```

---

## 1.3. 설계 결정: 확장 vs 신규 생성

| 비교 항목 | ❌ 신규 모델 생성<br/>`ResNetSAN01_DualHead.py` | ✅ **기존 모델 확장**<br/>`use_dual_head` 파라미터 |
|-----------|------------------------------------------------|---------------------------------------------------|
| **코드 중복** | ~300줄 복사 (Encoder, FiLM, Minkowski 등) | 0줄 (Decoder만 교체) |
| **유지보수** | 버그 수정 시 2곳 수정 필요 | 1곳만 수정 |
| **기능 조합** | `use_film + dual_head` 조합 어려움 | 모든 조합 자유롭게 가능 |
| **Rollback** | 새 모델 삭제 필요 | YAML flag만 변경 |
| **Checkpoint 호환** | 복잡한 변환 로직 필요 | 투명하게 동작 |
| **테스트** | 모든 기능 재테스트 | Decoder만 테스트 |

**✅ 최종 결정: 기존 ResNetSAN01 확장**

이유:
1. 코드베이스가 이미 이 패턴을 따르고 있음 (best practice)
2. 최소 변경으로 최대 효과
3. 실험 실패 시 즉시 rollback 가능

---

## 2. 기술적 배경

### 2.1. INT8 양자화의 근본적 한계

**현재 모델의 근본적 문제**:

1. **Per-channel Quantization 미지원**:
   - NPU는 Per-tensor 양자화만 지원
   - 단일 Scale/Zero-point로 0.5m~15m 범위를 표현해야 함
   - INT8(256 levels)로 14.5m 범위 표현 → **양자화 오차 ±28mm**

2. **넓은 Depth 범위**:
   - 현재 모델: 단일 출력으로 0.5~15m 예측
   - FP32는 높은 정밀도로 모든 범위 표현 가능
   - INT8은 256 레벨만 사용 가능 → 정밀도 급격히 저하

3. **Calibration만으로 한계**:
   - NPU의 자동 Clipping/Bias Correction은 이미 최적화되어 있음
   - 더 많은 calibration 데이터를 제공해도 효과 없음
   - **구조적 해결책 필요**

**양자화 공식**:
```
x_quantized = round((x - zero_point) / scale)
scale = (max - min) / 255
```

**현재 Single-Head 모델**:
- 범위: [0.5, 15.0]m
- scale = (15.0 - 0.5) / 255 = 0.0569
- **양자화 간격**: 56.9mm (약 5.7cm)
- 실제 오차: ±28.4mm

**문제점**:
1. **거친 양자화**: 5.7cm 간격으로만 값 표현 가능
2. **모든 거리에 동일한 오차**: 1m 거리도, 15m 거리도 같은 ±28mm 오차
3. **Per-tensor 제약**: 채널별로 다른 scale을 사용할 수 없음

---

### 2.2. 왜 Integer-Fractional 분리가 효과적인가?

**핵심 아이디어**:

```
Original Single-Head:
  depth ∈ [0.5, 15.0]m  →  1 output  →  INT8 (256 levels)
  양자화 오차: ±28mm

Proposed Dual-Head:
  integer_part ∈ [0, 15]  →  Head 1 (INT8, 16 levels effective)
  fractional_part ∈ [0, 1]m  →  Head 2 (INT8, 256 levels)
  양자화 오차: ±2mm (14배 개선!)
```

**Dual-Head 접근**:

**Head 1: Integer Part (정수부 예측)**
```
범위: [0, 15] (16개 정수값)
출력: Sigmoid [0, 1] → 선형 변환 → [0, 15]
양자화: INT8(256 levels)로 16개 값 표현
효과적 정밀도: 16배 오버샘플링 (각 정수당 16개 레벨)
```

**Head 2: Fractional Part (소수부 예측)**
```
범위: [0.0, 1.0]m
출력: Sigmoid [0, 1] → 그대로 사용
양자화: INT8(256 levels)로 1m 범위 표현
scale = 1.0 / 255 = 0.00392
양자화 간격: 3.92mm
실제 오차: ±1.96mm (14배 개선!)
```

**최종 깊이 복원**:
```python
depth = integer_part + fractional_part
예: integer=5, fractional=0.347 → depth=5.347m
```

**정밀도 비교**:

| 방식 | 범위 | 양자화 간격 | 오차 | 개선율 |
|------|------|-------------|------|--------|
| Single-Head | [0.5, 15.0]m | 56.9mm | ±28.4mm | - |
| Dual-Head (Integer) | [0, 15] | 16 levels | ±0.5 | - |
| Dual-Head (Fractional) | [0, 1.0]m | 3.92mm | **±1.96mm** | **14.5배** |

**장점**:
- ✅ NPU의 Dual-Output 기능 활용 (추가 비용 없음)
- ✅ 양자화 정밀도 14배 향상
- ✅ Per-channel 없이도 높은 정밀도 확보
- ✅ 물리적 의미가 명확 (정수부 = 미터 단위, 소수부 = 서브미터 정밀도)

---

### 2.3. NPU Dual-Output 활용

**NPU 확인된 사항**:
- ✅ **Dual-Output 지원 확정**
- 두 개의 독립적인 출력 텐서 생성 가능
- 추가 연산 비용 없음 (동일한 feature map에서 분기)

**구현 방식**:
```
Encoder Features → Decoder → [Branch 1: Integer Head]
                           → [Branch 2: Fractional Head]
```

---

## 3. 아키텍처 설계 (코드베이스 통합)

### 3.1. 현재 ResNetSAN01 구조 분석

**파일 위치**: `packnet_sfm/networks/depth/ResNetSAN01.py`

```python
class ResNetSAN01(nn.Module):
    def __init__(self, dropout=None, version=None, use_film=False, 
                 film_scales=[0], use_enhanced_lidar=False,
                 min_depth=0.5, max_depth=80.0, **kwargs):
        super().__init__()
        
        # Depth range (YAML에서 전달됨)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        
        # Encoder (공통 - 모든 모드에서 동일)
        num_layers = int(version[:2]) if version else 18
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=True)
        
        # ⬇️ Decoder (여기를 확장할 예정)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        
        # Optional: FiLM modulation
        self.use_film = use_film
        if use_film:
            # Minkowski encoder 생성
            if use_enhanced_lidar:
                self.mconvs = EnhancedMinkowskiEncoder(...)
            else:
                self.mconvs = MinkowskiEncoder(...)
        else:
            self.mconvs = None
        
        # Learnable fusion weights (FiLM용)
        self.weight = nn.Parameter(torch.ones(5) * 0.5)
        self.bias = nn.Parameter(torch.zeros(5))
    
    def run_network(self, rgb, input_depth=None):
        # Encode RGB features
        skip_features = self.encoder(rgb)
        
        # Optional: FiLM modulation
        if input_depth is not None and self.use_film:
            # ... FiLM processing ...
            pass
        
        # Decode to sigmoid outputs
        outputs = self.decoder(skip_features)
        # outputs = {("disp", 0): sigmoid [0,1], ...}
        
        return outputs
```

**핵심 발견**:
- `self.decoder` 교체만으로 Single/Dual-Head 전환 가능
- 나머지 300줄 코드는 그대로 재사용
- `min_depth`, `max_depth`가 이미 YAML에서 전달됨

---

### 3.2. Decoder Factory Pattern 설계

**목표**: YAML 파라미터로 Decoder 선택

```python
# packnet_sfm/networks/depth/ResNetSAN01.py (수정 부분)
class ResNetSAN01(nn.Module):
    def __init__(self, ..., use_dual_head=False, **kwargs):
        super().__init__()
        
        # ... 기존 encoder 코드 유지 ...
        
        # 🆕 Decoder 선택 (Factory Pattern)
        if use_dual_head:
            from packnet_sfm.networks.layers.resnet.dual_head_depth_decoder import DualHeadDepthDecoder
            self.decoder = DualHeadDepthDecoder(
                num_ch_enc=self.encoder.num_ch_enc,
                max_depth=self.max_depth,
                scales=range(4)
            )
            self.is_dual_head = True
            print(f"✅ Using Dual-Head Decoder (max_depth={self.max_depth})")
        else:
            self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
            self.is_dual_head = False
            print(f"✅ Using Single-Head Decoder")
        
        # ... 기존 FiLM/Minkowski 코드 유지 ...
```

**변경량**: **10줄 추가** (기존 코드 0줄 수정)

---

### 3.3. 기존 기능과의 통합

**모든 조합 가능**:

```yaml
# 조합 1: Single-Head (기존)
depth_net:
    name: 'ResNetSAN01'
    use_dual_head: false
    use_film: false

# 조합 2: Dual-Head only
depth_net:
    name: 'ResNetSAN01'
    use_dual_head: true
    use_film: false

# 조합 3: Dual-Head + FiLM (하이브리드)
depth_net:
    name: 'ResNetSAN01'
    use_dual_head: true
    use_film: true
    film_scales: [0]

# 조합 4: Dual-Head + FiLM + Enhanced LiDAR (Full)
depth_net:
    name: 'ResNetSAN01'
    use_dual_head: true
    use_film: true
    use_enhanced_lidar: true
```

**Backward Compatibility 보장**:
- `use_dual_head` 파라미터 없으면 → Single-Head (기존 동작)
- 기존 checkpoint 로딩 → 정상 동작 (decoder만 다름)
