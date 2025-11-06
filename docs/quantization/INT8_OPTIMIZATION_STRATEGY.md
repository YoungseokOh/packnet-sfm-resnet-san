# INT8 양자화 성능 최적화 전략

**목표**: NPU INT8 성능 향상 (현재 abs_rel 0.1133 → 목표 0.05 이하)  
**제약사항**: Post-Training Quantization (PTQ) only, min/max calibration  
**날짜**: 2025-11-06

---

## 📊 현재 상태

### 성능 지표
```
FP32 (PyTorch):  abs_rel = 0.0304
INT8 (NPU PTQ):  abs_rel = 0.1133
Degradation:     +272% (3.7배 악화)
```

### 문제 분석
- **Output quantization**: ±28mm (이론적)
- **실제 RMSE 증가**: 351mm (이론의 12.5배!)
- **주요 원인**: Multi-layer feature map quantization 누적 효과

---

## 🎯 최적화 전략 (3가지 접근)

---

## 전략 1: Integer-Fractional Separation (정수부/소수부 분리)

### 🔍 핵심 아이디어

깊이 값을 **정수부와 소수부로 분리**하여 각각 독립적으로 양자화

```python
# Depth decomposition
depth = 7.5m
integer_part = 7    # INT8: 0-15 (16 levels for integer)
fractional_part = 0.5  # INT8: 0-255 (256 levels for fraction)

# Reconstruction
depth_reconstructed = integer_part + (fractional_part / 256)
```

### 📐 수학적 분석

#### 현재 방식 (Single INT8)
```
Range: [0.5, 15.0]m
Step: 14.5 / 255 = 0.0569m = 56.9mm
Error: ±28.4mm
```

#### 제안 방식 (Integer + Fractional)
```
Integer part (0-15):
  - 4 bits (16 levels)
  - Step: 1m
  - Error: ±0.5m

Fractional part (0-1):
  - 8 bits (256 levels)  
  - Step: 1/256 = 0.00391m = 3.9mm
  - Error: ±1.95mm

Total error: ±1.95mm (14.5배 개선!)
```

### 🏗️ 구현 방안

#### Option A: Dual-Head Architecture
```python
class DualHeadDepthDecoder(nn.Module):
    def __init__(self):
        # Shared encoder
        self.encoder = ResNetEncoder()
        
        # Separate heads
        self.integer_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()  # Output: [0, 1] → scale to [0, 15]
        )
        
        self.fractional_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()  # Output: [0, 1] → fractional part
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        # Integer part: [0, 15]m
        int_sigmoid = self.integer_head(features)
        integer_part = int_sigmoid * 15.0
        
        # Fractional part: [0, 1)
        frac_sigmoid = self.fractional_head(features)
        
        # Combine
        depth = integer_part + frac_sigmoid
        return depth
```

#### Option B: Single-Head with Post-Processing
```python
class DepthDecoderWithSeparation(nn.Module):
    def forward(self, x):
        # Standard depth prediction
        depth = self.depth_head(x)  # [0.5, 15.0]m
        
        # Training: No separation (standard loss)
        if self.training:
            return depth
        
        # Inference: Separate for INT8
        else:
            integer_part = torch.floor(depth)
            fractional_part = depth - integer_part
            return integer_part, fractional_part
```

### ✅ 장점
1. **정밀도 향상**: ±28mm → ±2mm (14배 개선)
2. **Uniform error**: 모든 깊이 범위에서 동일
3. **PTQ 호환**: Post-processing만으로 적용 가능

### ❌ 단점
1. **NPU 제약**: Dual output 지원 여부 확인 필요
2. **재학습 필요**: Dual-head는 처음부터 재학습
3. **복잡도 증가**: Inference pipeline 수정 필요

### 🎯 추천 구현 순서
1. **Phase 1**: Option B (Post-processing) - 즉시 테스트 가능
2. **Phase 2**: NPU dual-output 검증
3. **Phase 3**: Option A (Dual-head) - 재학습

---

## 전략 2: Knowledge Distillation (Teacher-Student)

### 🔍 핵심 아이디어

FP32 Teacher 모델의 **feature-level 지식**을 INT8 Student에 전달

```
Teacher (FP32) ──→ Feature Maps ──┐
                                  ├──→ Distillation Loss
Student (INT8) ──→ Feature Maps ──┘
                     ↓
                Output Depth
```

### 📐 수학적 정의

#### Standard Loss (현재)
```python
L_standard = MSE(pred_int8, gt_depth)
```

#### Distillation Loss (제안)
```python
L_distill = L_output + α·L_feature + β·L_hint

L_output = MSE(pred_int8, pred_fp32)  # Output matching
L_feature = Σ MSE(F_int8[i], F_fp32[i])  # Feature matching
L_hint = MSE(attention_int8, attention_fp32)  # Attention matching
```

### 🏗️ 구현 방안

#### Distillation Training Loop
```python
class DistillationTrainer:
    def __init__(self, teacher_fp32, student_int8):
        self.teacher = teacher_fp32.eval()  # Frozen
        self.student = student_int8
        
        # Loss weights
        self.alpha = 0.5  # Feature distillation
        self.beta = 0.3   # Hint distillation
    
    def forward(self, batch):
        # Teacher inference (no grad)
        with torch.no_grad():
            teacher_output = self.teacher(batch['rgb'])
            teacher_features = self.teacher.get_features()
        
        # Student training
        student_output = self.student(batch['rgb'])
        student_features = self.student.get_features()
        
        # Losses
        L_output = F.mse_loss(student_output, teacher_output)
        L_feature = sum([
            F.mse_loss(s_feat, t_feat.detach())
            for s_feat, t_feat in zip(student_features, teacher_features)
        ])
        
        total_loss = L_output + self.alpha * L_feature
        return total_loss
```

#### Quantization-Aware Feature Matching
```python
def feature_distillation_loss(student_feat, teacher_feat):
    """
    Match feature distributions instead of exact values
    → More robust to quantization noise
    """
    # Statistical matching
    loss_mean = F.mse_loss(student_feat.mean(), teacher_feat.mean())
    loss_std = F.mse_loss(student_feat.std(), teacher_feat.std())
    
    # Distribution matching (KL divergence)
    loss_kl = F.kl_div(
        F.log_softmax(student_feat.flatten(), dim=0),
        F.softmax(teacher_feat.flatten(), dim=0),
        reduction='batchmean'
    )
    
    return loss_mean + loss_std + 0.1 * loss_kl
```

### ✅ 장점
1. **Feature-level guidance**: 단순 output matching보다 효과적
2. **Quantization 대응**: INT8 특성에 맞게 학습
3. **검증된 방법**: CV 분야에서 널리 사용

### ❌ 단점
1. **재학습 필수**: FP32 Teacher 필요
2. **메모리 2배**: Teacher + Student 동시 로드
3. **학습 시간 증가**: ~1.5-2배

### 🎯 추천 구현 순서
1. **Phase 1**: Output distillation (L_output만)
2. **Phase 2**: Feature distillation 추가
3. **Phase 3**: Attention/Hint distillation

### 📊 예상 성능 개선
```
Baseline PTQ:        abs_rel = 0.1133
+ Output distill:    abs_rel = 0.08 (30% 개선)
+ Feature distill:   abs_rel = 0.06 (47% 개선)
+ Attention distill: abs_rel = 0.04 (65% 개선)
```

---

## 전략 3: Mixed Precision (NPU 지원 시)

### 🔍 핵심 아이디어

**Critical layers는 FP16**, Non-critical layers는 INT8

```
Input (INT8)
  ↓
Encoder Layers 1-3: INT8 (빠름, 정확도 덜 중요)
  ↓
Encoder Layer 4: FP16 (중요한 high-level features)
  ↓
Decoder: FP16 (정밀도 중요)
  ↓
Final Conv: FP16 (depth output, 최고 정밀도 필요)
  ↓
Output (FP32)
```

### 📐 성능 분석

#### Layer-wise Sensitivity Analysis (사전 분석 필요)
```python
def analyze_layer_sensitivity(model, val_loader):
    """
    각 layer를 INT8로 변환했을 때 성능 저하 측정
    """
    sensitivities = {}
    
    for layer_name in model.layers:
        # Quantize only this layer
        quantized_model = quantize_single_layer(model, layer_name)
        
        # Measure degradation
        metrics = evaluate(quantized_model, val_loader)
        sensitivity = metrics['abs_rel'] - baseline_abs_rel
        
        sensitivities[layer_name] = sensitivity
    
    return sensitivities

# Example output:
# {
#   'encoder.layer1': 0.002,  # Low sensitivity → INT8 OK
#   'encoder.layer4': 0.045,  # High sensitivity → FP16!
#   'decoder.conv5': 0.038,   # High sensitivity → FP16!
#   'final_conv': 0.052,      # Highest sensitivity → FP16!
# }
```

#### Precision Assignment Strategy
```python
class MixedPrecisionModel:
    def __init__(self, sensitivity_dict, threshold=0.02):
        self.precision_map = {}
        
        for layer, sensitivity in sensitivity_dict.items():
            if sensitivity > threshold:
                self.precision_map[layer] = 'FP16'
            else:
                self.precision_map[layer] = 'INT8'
    
    def get_precision_config(self):
        return self.precision_map
```

### 🏗️ NPU 제약사항 확인 필요

#### 확인 사항
1. **NPU FP16 지원 여부**
   - 일부 NPU는 INT8만 지원
   - FP16 지원 시 throughput 확인

2. **Per-layer precision 설정 가능 여부**
   - ONNX mixed precision export 지원
   - NPU runtime mixed precision 지원

3. **메모리 제약**
   - INT8: 14MB
   - FP16: 27MB
   - Mixed: ~18-22MB (예상)

### ✅ 장점
1. **정확도-속도 균형**: FP32와 INT8의 중간
2. **선택적 최적화**: Critical layers만 FP16
3. **PTQ 가능**: 재학습 없이 적용

### ❌ 단점
1. **NPU 의존성**: NPU가 FP16/Mixed precision 지원해야 함
2. **복잡한 최적화**: Layer sensitivity 분석 필요
3. **불확실성**: NPU에서 실제 동작 보장 안됨

### 🎯 확인 절차
```bash
# 1. NPU FP16 지원 확인
npu-info --supported-dtypes

# 2. ONNX mixed precision export 테스트
python scripts/export_onnx_mixed_precision.py

# 3. NPU에서 로드 테스트
python scripts/test_npu_mixed_precision.py
```

---

## 🎯 종합 전략 및 우선순위

### Phase 1: 즉시 적용 가능 (1-2일)
1. ✅ **Integer-Fractional Separation (Post-processing)**
   - 재학습 불필요
   - 즉시 테스트 가능
   - 예상 개선: abs_rel 0.1133 → 0.09

2. ✅ **NPU 제약사항 확인**
   - FP16 지원 여부
   - Mixed precision 가능성
   - Dual-output 지원 여부

### Phase 2: 중기 전략 (1-2주)
3. 🔄 **Knowledge Distillation (Output-level)**
   - Teacher: 현재 FP32 모델
   - Student: INT8-aware training
   - 예상 개선: abs_rel 0.1133 → 0.06-0.08

### Phase 3: 장기 전략 (2-4주)
4. 🔄 **Dual-Head Architecture (재학습)**
   - Integer + Fractional heads
   - 처음부터 분리 학습
   - 예상 개선: abs_rel 0.1133 → 0.04-0.05

5. 🔄 **Advanced Distillation**
   - Feature-level matching
   - Attention distillation
   - 예상 최종: abs_rel 0.03-0.04 (FP32 수준!)

### Phase 4: 조건부 (NPU 지원 시)
6. ⏸️ **Mixed Precision**
   - NPU FP16 지원 시만 가능
   - Layer sensitivity 분석
   - Critical layers FP16 할당

---

## 📊 예상 성능 로드맵

```
Current:                    abs_rel = 0.1133

Phase 1 (Post-processing):  abs_rel = 0.09   (20% 개선)
Phase 2 (Output distill):   abs_rel = 0.07   (38% 개선)
Phase 3 (Dual-head):        abs_rel = 0.05   (56% 개선)
Phase 4 (Feature distill):  abs_rel = 0.035  (69% 개선)

Target:                     abs_rel < 0.05   (✅ 달성 가능!)
```

---

## 🔧 실험 체크리스트

### Phase 1: 즉시 실험
- [ ] Integer-Fractional post-processing 구현
- [ ] NPU dual-output 테스트
- [ ] NPU FP16 지원 확인
- [ ] Baseline 성능 측정

### Phase 2: 재학습 실험
- [ ] Output distillation 구현
- [ ] Teacher 모델 로드 테스트
- [ ] Distillation training loop
- [ ] Validation 성능 비교

### Phase 3: 고급 실험
- [ ] Dual-head architecture 설계
- [ ] Feature distillation 구현
- [ ] Mixed precision layer analysis
- [ ] 최종 성능 검증

---

## 💡 핵심 권장사항

1. **Phase 1부터 순차 진행**
   - 빠른 검증 → 점진적 개선
   - 각 단계마다 성능 측정

2. **NPU 제약사항 최우선 확인**
   - FP16, Dual-output 지원 여부
   - 이에 따라 전략 조정

3. **Knowledge Distillation 우선 추천**
   - 검증된 방법
   - 재학습 필요하지만 효과 확실
   - Feature-level까지 확장 가능

4. **Integer-Fractional은 보조 전략**
   - 즉시 테스트 가능
   - NPU 제약 있을 수 있음
   - 하지만 시도할 가치 있음!

---

## 📝 다음 단계

1. **NPU 스펙 확인** (최우선!)
2. **Post-processing 테스트** (1일)
3. **Distillation 구현** (1주)
4. **성능 비교 및 최종 선택** (2주)

**목표: abs_rel 0.05 이하 달성!** 🎯
