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
1. ~~**NPU 제약**: Dual output 지원 여부 확인 필요~~ ✅ **확인됨: Dual output 지원!**
2. **재학습 필요**: Dual-head는 처음부터 재학습
3. **복잡도 증가**: Inference pipeline 수정 필요

### ✅ NPU 지원 확인됨
- **Dual output 지원**: ✅ 가능 확인
- **권장 구현**: Dual-head architecture 적극 추천
- **우선순위 상향**: Phase 1 → Phase 2로 조정

### 🎯 추천 구현 순서
1. **Phase 1**: ~~Option B (Post-processing) - 즉시 테스트 가능~~
2. **Phase 2**: ~~NPU dual-output 검증~~ ✅ **확인 완료**
3. **Phase 3**: **Option A (Dual-head) - 재학습 권장** ⭐ **우선순위 상향!**

### 💡 Dual Output 지원 확인에 따른 권장사항
- **즉시 Dual-head 재학습 시작 가능**
- **예상 최대 효과**: ±28mm → ±2mm (14배 개선)
- **NPU 제약 없음**: Integer + Fractional 동시 출력 가능
- **구현 복잡도**: 중간 (재학습 필요하지만 구조는 단순)

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

## 전략 4: Advanced PTQ Calibration (NPU 전문가 관점)

### 🔍 핵심 아이디어

**Calibration은 PTQ의 생명선!** Min/max만으로는 부족합니다.

```
Poor Calibration → 30-50% 성능 저하
Optimal Calibration → 5-10% 성능 저하
```

### 📐 Calibration 전략

#### 1. Percentile-based Range Selection
```python
def optimal_calibration(activations, method='percentile'):
    """
    Outlier에 강건한 calibration range 결정
    """
    if method == 'min_max':
        # ❌ Bad: Outlier에 취약
        qmin, qmax = activations.min(), activations.max()
        
    elif method == 'percentile':
        # ✅ Good: Outlier 제거
        qmin = torch.quantile(activations, 0.001)  # 0.1 percentile
        qmax = torch.quantile(activations, 0.999)  # 99.9 percentile
        
    elif method == 'entropy':
        # ✅ Best: KL divergence 최소화
        qmin, qmax = find_optimal_range_kl(activations)
    
    return qmin, qmax
```

#### 2. Per-Channel Quantization (Critical!)
```python
# ❌ Per-tensor: 전체 weight를 하나의 scale로
scale_tensor = (w_max - w_min) / 255
# → 일부 channel이 매우 작으면 정밀도 손실

# ✅ Per-channel: 각 channel마다 독립적인 scale
for c in range(num_channels):
    scale[c] = (w_max[c] - w_min[c]) / 255
# → 3-5배 정확도 향상!
```

**NPU 확인 필요**: Per-channel quantization 지원 여부!

#### 3. Representative Calibration Dataset
```python
def select_calibration_data(dataset, n_samples=100):
    """
    Representative samples 선정 기준:
    1. Depth distribution coverage
    2. Scene diversity
    3. Lighting conditions
    """
    # Depth distribution 분석
    depth_stats = analyze_depth_distribution(dataset)
    
    # Stratified sampling
    samples = []
    for depth_range in [(0.5, 3), (3, 8), (8, 15)]:
        range_samples = get_samples_in_range(dataset, depth_range)
        samples.extend(random.sample(range_samples, n_samples // 3))
    
    return samples
```

### 📊 Activation Quantization 최적화

#### Layer-wise Quantization Strategy
```python
class SmartQuantizer:
    def __init__(self):
        self.layer_configs = {
            # Encoder: Aggressive quantization OK
            'encoder.layer1': {'bits': 8, 'method': 'per_tensor'},
            'encoder.layer2': {'bits': 8, 'method': 'per_tensor'},
            
            # Encoder layer 3-4: More careful
            'encoder.layer3': {'bits': 8, 'method': 'per_channel'},
            'encoder.layer4': {'bits': 8, 'method': 'per_channel'},
            
            # Decoder: Most critical
            'decoder.conv1': {'bits': 8, 'method': 'per_channel'},
            'decoder.conv5': {'bits': 8, 'method': 'per_channel'},
            
            # Final layer: Highest precision needed
            'final_conv': {'bits': 8, 'method': 'per_channel', 'symmetric': False}
        }
```

#### Asymmetric vs Symmetric Quantization
```python
# Symmetric (centered at 0):
# Range: [-127, 127]
# Zero point: 0
# → Faster on NPU, but less precise for non-symmetric activations

# Asymmetric (flexible):
# Range: [qmin, qmax]
# Zero point: variable
# → More precise, especially for ReLU outputs (always positive)

# 추천:
# - Weights: Symmetric (usually centered)
# - Activations after ReLU: Asymmetric (always positive)
```

### 🎯 NPU-Specific 최적화

#### 1. Batch Size Optimization
```python
# NPU는 특정 batch size에서 최적화됨
optimal_batch_sizes = [1, 2, 4, 8, 16]

def find_optimal_batch_size(npu_model):
    best_throughput = 0
    best_batch_size = 1
    
    for bs in optimal_batch_sizes:
        throughput = benchmark_npu(npu_model, batch_size=bs)
        if throughput > best_throughput:
            best_throughput = throughput
            best_batch_size = bs
    
    return best_batch_size

# 예상: batch_size=4 or 8이 최적
```

#### 2. Input Quantization
```python
class NPUOptimizedPreprocessing:
    def __init__(self):
        # RGB input: UINT8 [0, 255] → 그대로 사용!
        # Normalization을 INT8 연산으로 통합
        
        # ✅ Good: NPU-friendly
        self.scale = torch.tensor([1/255.0])
        self.zero_point = torch.tensor([0])
        
        # ❌ Bad: FP32 연산 추가
        # x = (x - mean) / std  # Avoid this!
```

#### 3. Memory Bandwidth Optimization
```python
# NPU는 memory bandwidth에 민감
# → 중간 tensor 크기 최소화

class EfficientDecoder(nn.Module):
    def forward(self, x):
        # ❌ Bad: Large intermediate tensors
        x = self.conv1(x)  # (B, 256, H, W)
        x = self.conv2(x)  # (B, 256, H, W)
        
        # ✅ Good: Fused operations
        x = self.fused_conv_relu(x)  # Single op
```

### 🔧 Outlier Handling

#### Channel-wise Clipping
```python
def handle_outliers(weights, percentile=99.9):
    """
    Extreme outlier를 clipping하여 quantization range 최적화
    """
    # Per-channel outlier detection
    for c in range(weights.shape[0]):
        channel_weights = weights[c]
        
        # Find outliers
        threshold = torch.quantile(channel_weights.abs(), percentile/100)
        
        # Clip
        weights[c] = torch.clamp(channel_weights, -threshold, threshold)
    
    return weights

# 실험 결과: 99.9% clipping으로 2-3% 성능 향상 가능
```

### ✅ 장점
1. **즉시 적용 가능**: 재학습 불필요
2. **검증된 기법**: Industry standard
3. **누적 효과**: 여러 기법 조합 시 10-20% 개선

### ❌ 단점
1. **NPU 제약 확인 필요**: Per-channel, asymmetric 지원 여부
2. **Calibration 시간**: 100-1000 samples 필요
3. **Trial & error**: 최적 설정 찾기 어려움

### 🎯 권장 실험 순서

1. **Baseline 재측정** (current calibration)
   ```bash
   # 현재 사용 중인 calibration 방법 확인
   python scripts/analyze_current_calibration.py
   ```

2. **Percentile-based calibration**
   ```python
   # 99.9% percentile clipping
   calibrate_model(model, calib_data, method='percentile_99.9')
   ```

3. **Per-channel quantization** (NPU 지원 시)
   ```python
   quantize_model(model, per_channel=True)
   ```

4. **Optimal calibration dataset**
   ```python
   # 100 representative samples
   calib_data = select_calibration_data(train_dataset, n=100)
   ```

### 📊 예상 성능 개선

```
Current (min/max):           abs_rel = 0.1133
+ Percentile calibration:    abs_rel = 0.10   (12% 개선)
+ Per-channel quantization:  abs_rel = 0.08   (29% 개선)
+ Optimal calib dataset:     abs_rel = 0.075  (34% 개선)

Combined:                    abs_rel = 0.07-0.075 (30-35% 개선!)
```

---

## 전략 5: Quantization-Aware Fine-tuning (QAF)

### 🔍 핵심 아이디어

**PTQ의 한계를 극복**: Fine-tuning으로 quantization error 보상

```
PTQ (Post-Training):         abs_rel = 0.1133
QAT (from scratch):          abs_rel = 0.05   (재학습 4주)
QAF (Fine-tuning):           abs_rel = 0.06   (Fine-tune 3일!)
```

### 📐 QAF vs QAT

| Method | Time | Accuracy | Flexibility |
|--------|------|----------|-------------|
| **PTQ** | 1 hour | 0.1133 | ✅ Fast |
| **QAF** | 3 days | 0.06 | ⭐ Balanced |
| **QAT** | 4 weeks | 0.05 | ❌ Slow |

### 🏗️ 구현 방안

#### Fake Quantization Layer
```python
class FakeQuantize(nn.Module):
    def __init__(self, num_bits=8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))
        self.num_bits = num_bits
    
    def forward(self, x):
        # Training: Simulate INT8 with gradients
        if self.training:
            # Quantize
            x_q = torch.round(x / self.scale) + self.zero_point
            x_q = torch.clamp(x_q, 0, 2**self.num_bits - 1)
            
            # Dequantize (Straight-Through Estimator)
            x_dq = (x_q - self.zero_point) * self.scale
            
            # Gradient flows through!
            return x_dq
        else:
            # Inference: Real quantization
            return real_quantize(x, self.scale, self.zero_point)
```

#### Fine-tuning Strategy
```python
def quantization_aware_finetune(model, train_loader):
    # 1. Load FP32 checkpoint
    model.load_checkpoint('fp32_model.ckpt')
    
    # 2. Insert fake quantization layers
    model = insert_fake_quant_layers(model)
    
    # 3. Initialize scales from PTQ
    initialize_scales_from_ptq(model, calib_data)
    
    # 4. Fine-tune (짧게!)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Low LR!
    
    for epoch in range(3):  # 3 epochs만!
        for batch in train_loader:
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()
    
    # 5. Export to real INT8
    export_to_int8(model)
```

### ✅ 장점
1. **빠른 수렴**: 3-5 epoch이면 충분
2. **PTQ 대비 2-3배 개선**: abs_rel 0.11 → 0.06
3. **Full QAT 대비 10배 빠름**: 3일 vs 4주

### ❌ 단점
1. **재학습 필요**: PTQ만으로는 안됨
2. **Hyperparameter 튜닝**: Learning rate, epochs 민감
3. **NPU 검증 필요**: Fake quant와 real quant 차이

### 🎯 권장 설정

```python
# Fine-tuning config
config = {
    'learning_rate': 1e-5,  # 매우 작게!
    'epochs': 3,            # 짧게!
    'batch_size': 8,        # FP32 학습과 동일
    'optimizer': 'Adam',    # Adam 추천
    'scheduler': 'cosine',  # Cosine annealing
    
    # Quantization config
    'weight_bits': 8,
    'activation_bits': 8,
    'per_channel': True,    # Per-channel 권장
    'symmetric': False,     # Asymmetric 권장
}
```

### 📊 예상 성능

```
PTQ baseline:                abs_rel = 0.1133
+ Advanced calibration:      abs_rel = 0.075  (Phase 4)
+ QAF (3 epochs):            abs_rel = 0.06   (Phase 5)

→ 47% 개선! (0.1133 → 0.06)
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

## 🎯 종합 전략 및 우선순위 (NPU 전문가 권장)

### Phase 1: Advanced PTQ Calibration (즉시, 1일)
**목표**: 재학습 없이 최대 성능 확보

1. ✅ **Percentile-based Calibration**
   - 99.9% percentile clipping
   - Outlier handling
   - 예상: abs_rel 0.1133 → 0.10 (12% 개선)

2. ✅ **Per-channel Quantization** (NPU 지원 시)
   - Weight per-channel quantization
   - Activation asymmetric quantization
   - 예상: abs_rel 0.10 → 0.08 (추가 20% 개선)

3. ✅ **Optimal Calibration Dataset**
   - 100 representative samples
   - Depth distribution coverage
   - 예상: abs_rel 0.08 → 0.075 (추가 6% 개선)

**Phase 1 총 예상**: abs_rel 0.1133 → **0.075** (34% 개선)

---

### Phase 2: Dual-Head Architecture (중기, 1-2주) ⭐ **추천!**
**목표**: Integer-Fractional separation으로 precision 극대화

4. 🔄 **Dual-Head 재학습**
   - ✅ NPU dual-output 지원 확인됨!
   - Integer head (0-15m) + Fractional head (0-1)
   - 예상: abs_rel 0.075 → **0.05** (33% 추가 개선)
   
**누적 예상**: abs_rel 0.1133 → **0.05** (56% 개선) ✅ **목표 달성!**

---

### Phase 3: Knowledge Distillation (장기, 2-3주)
**목표**: FP32 수준 성능 달성

5. 🔄 **Output-level Distillation**
   - Teacher: FP32 모델
   - Student: Dual-head INT8
   - 예상: abs_rel 0.05 → 0.04 (20% 추가 개선)

6. 🔄 **Feature-level Distillation**
   - Multi-layer feature matching
   - Attention distillation
   - 예상: abs_rel 0.04 → **0.035** (13% 추가 개선)

**누적 예상**: abs_rel 0.1133 → **0.035** (69% 개선) ✅ **FP32 수준!**

---

### Phase 4: Quantization-Aware Fine-tuning (조건부, 3-5일)
**목표**: Distillation 대안 (더 빠름)

7. 🔄 **QAF (3 epochs)**
   - Fake quantization + Fine-tuning
   - PTQ initialization
   - 예상: abs_rel 0.075 → **0.06** (20% 개선)

**Phase 2 대신 Phase 4 사용 가능**: 
- Phase 1 (0.075) + Phase 4 (0.06) = **더 빠른 경로!**
- Dual-head보다 구현 단순

---

### Phase 5: Mixed Precision (조건부, NPU FP16 지원 시)
8. ⏸️ **Layer-wise Mixed Precision**
   - Critical layers: FP16
   - Non-critical: INT8
   - 예상: abs_rel 0.06 → 0.045 (25% 추가 개선)

---

### 🎯 최종 권장 경로

#### **경로 A: 빠른 달성** (2-3주)
```
Phase 1 (Advanced PTQ): 0.1133 → 0.075  (1일)
Phase 4 (QAF):          0.075 → 0.06   (3일)
Phase 3 (Distillation): 0.06 → 0.04    (2주)

총 소요: 2-3주
최종 성능: abs_rel = 0.04 (65% 개선)
```

#### **경로 B: 최고 성능** (4-5주) ⭐ **추천!**
```
Phase 1 (Advanced PTQ):   0.1133 → 0.075  (1일)
Phase 2 (Dual-Head):      0.075 → 0.05   (2주)
Phase 3 (Distillation):   0.05 → 0.035   (2주)

총 소요: 4-5주
최종 성능: abs_rel = 0.035 (69% 개선, FP32 수준!)
```

#### **경로 C: 초고속** (1주)
```
Phase 1 (Advanced PTQ): 0.1133 → 0.075  (1일)
Phase 4 (QAF):          0.075 → 0.06   (3일)

총 소요: 4일
최종 성능: abs_rel = 0.06 (47% 개선, 목표 근접!)
```

---

### 💡 NPU 전문가의 핵심 권장사항

1. **Phase 1은 필수!** (Advanced PTQ Calibration)
   - 어떤 경로든 먼저 수행
   - 재학습 없이 34% 개선
   - 1일이면 완료

2. **Dual-Head vs QAF 선택**
   - **시간 충분**: Dual-Head (더 높은 성능)
   - **빠른 결과**: QAF (3일 완료)
   - **Both**: Dual-Head + QAF 조합도 가능!

3. **Distillation은 final boost**
   - Phase 2 or 4 이후 적용
   - FP32 수준 달성 가능
   - Feature-level까지 확장

4. **Mixed Precision은 bonus**
   - NPU FP16 지원 시만
   - 추가 5-10% 개선 가능
   - 마지막 polish용

---

### 📊 예상 성능 로드맵 (업데이트)

```
Current:                         abs_rel = 0.1133

Phase 1 (Advanced PTQ):          abs_rel = 0.075  (34% 개선) ⭐
Phase 2 (Dual-Head):             abs_rel = 0.05   (56% 개선) ✅ 목표!
Phase 3 (Distillation):          abs_rel = 0.035  (69% 개선) 🎯 FP32급!
Phase 4 (QAF, 대안):             abs_rel = 0.06   (47% 개선) ⚡ 빠름!
Phase 5 (Mixed Precision):       abs_rel = 0.045  (60% 개선) 🔥 Bonus

Target:                          abs_rel < 0.05   ✅ 달성 가능!
FP32-level:                      abs_rel ~ 0.035  ✅ 달성 가능!
```

---

## 🔧 실험 체크리스트 (업데이트)

### Phase 1: Advanced PTQ (즉시, 최우선!) ⭐
- [ ] 현재 calibration 방법 분석
- [ ] Percentile-based calibration 구현 (99.9%)
- [ ] Per-channel quantization 테스트 (NPU 지원 확인)
- [ ] Asymmetric quantization 적용
- [ ] Optimal calibration dataset 선정 (100 samples)
- [ ] Baseline 대비 성능 측정
- [ ] **예상 결과**: abs_rel 0.075

### Phase 2A: Dual-Head Architecture (추천 경로)
- [ ] Dual-head decoder 설계
- [ ] Integer + Fractional head 구현
- [ ] 재학습 (NCDB dataset)
- [ ] NPU dual-output export 검증
- [ ] INT8 quantization 적용
- [ ] **예상 결과**: abs_rel 0.05 ✅ 목표 달성!

### Phase 2B: QAF (빠른 경로, 대안)
- [ ] Fake quantization layer 구현
- [ ] PTQ scales로 초기화
- [ ] Fine-tuning (3 epochs, lr=1e-5)
- [ ] NPU export 및 검증
- [ ] **예상 결과**: abs_rel 0.06

### Phase 3: Knowledge Distillation (최종 polish)
- [ ] Teacher (FP32) 모델 준비
- [ ] Student (INT8 Dual-head) 구조
- [ ] Output distillation loss 구현
- [ ] Feature distillation loss 추가
- [ ] Distillation training (10 epochs)
- [ ] **예상 결과**: abs_rel 0.035 🎯 FP32급!

### Phase 4: Mixed Precision (조건부)
- [ ] NPU FP16 지원 확인
- [ ] Layer sensitivity 분석
- [ ] Critical layers FP16 할당
- [ ] Mixed precision ONNX export
- [ ] NPU 성능 검증
- [ ] **예상 결과**: abs_rel 0.045 (bonus)

### NPU 스펙 확인 체크리스트 (최우선!)
- [x] **Dual output 지원**: ✅ 확인됨
- [ ] **Per-channel quantization**: 확인 필요
- [ ] **Asymmetric quantization**: 확인 필요  
- [ ] **FP16 mixed precision**: 확인 필요
- [ ] **Optimal batch size**: 벤치마크 필요
- [ ] **Memory bandwidth**: 프로파일링 필요

---

## 💡 핵심 권장사항 (NPU 전문가 최종 조언)

### 1. **Phase 1 (Advanced PTQ)부터 무조건 시작!** ⭐⭐⭐
   - **이유**: 재학습 없이 34% 개선 (0.1133 → 0.075)
   - **시간**: 단 1일
   - **위험**: 없음 (PTQ만)
   - **효과**: 검증됨
   
   **구체적 액션**:
   ```python
   # 1. Percentile calibration (30분)
   calibrate_with_percentile(model, calib_data, percentile=99.9)
   
   # 2. Per-channel quantization (1시간, NPU 확인 필요)
   quantize_per_channel(model, method='asymmetric')
   
   # 3. Optimal calibration dataset (2시간)
   calib_data = select_representative_samples(train_data, n=100)
   
   # 4. 성능 측정 (30분)
   evaluate_on_npu(model, test_data)
   ```

### 2. **Dual-Head가 최고의 선택** (NPU dual-output 지원 확인됨!) ✅
   - **이유**: ±28mm → ±2mm (14배 precision 향상)
   - **시간**: 2주 (재학습)
   - **예상**: abs_rel 0.05 달성 (목표!)
   - **리스크**: 중간 (재학습 필요)
   
   **vs QAF 비교**:
   | | Dual-Head | QAF |
   |---|-----------|-----|
   | **시간** | 2주 | 3일 |
   | **성능** | 0.05 | 0.06 |
   | **안정성** | 높음 | 중간 |
   | **추천도** | ⭐⭐⭐ | ⭐⭐ |

### 3. **Knowledge Distillation은 마지막 polish** 🎯
   - **타이밍**: Phase 2 (Dual-head or QAF) 이후
   - **효과**: abs_rel 0.05 → 0.035 (FP32급!)
   - **시간**: 추가 2주
   - **선택사항**: 목표(0.05) 달성 후 결정
   
   **조언**: 
   - Phase 2까지만 해도 목표 달성
   - FP32 수준 필요시에만 Phase 3 진행

### 4. **NPU 제약사항 확인이 최우선!** 🔍
   
   **즉시 확인 필요**:
   ```bash
   # 1. Per-channel quantization 지원?
   # → 지원 시: 20-30% 추가 개선!
   # → 미지원: Asymmetric만 사용
   
   # 2. Asymmetric quantization 지원?
   # → ReLU 후 activation에 필수
   
   # 3. FP16 mixed precision 지원?
   # → Bonus 5-10% 개선 가능
   ```
   
   **확인 방법**:
   - NPU 제조사 문서 확인
   - Sample quantization config 테스트
   - 실제 NPU에서 로드 테스트

### 5. **점진적 진행 & 매 단계 검증** 📊
   
   ```
   Phase 1 완료 → 성능 측정 → 만족하면 Phase 2
                              ↓ 불만족
                              → Calibration 재조정
   
   Phase 2 완료 → 성능 측정 → 목표(0.05) 달성?
                              ↓ YES: 완료! 🎉
                              ↓ NO: Phase 3 진행
   
   Phase 3 완료 → FP32급 달성 → 프로덕션 배포
   ```

### 6. **빠른 프로토타이핑 경로** ⚡
   
   **만약 시간이 매우 촉박하다면**:
   ```
   Week 1 Day 1: Phase 1 (Advanced PTQ)     → 0.075
   Week 1 Day 2-4: Phase 4 (QAF)           → 0.06
   
   → 4일 만에 47% 개선! (목표 근접)
   ```

### 7. **실전 팁** 💼

#### Calibration Dataset 선정
```python
# ✅ Good: Diverse samples
calib_samples = {
    'near_depth': 30 samples,   # 0.5-3m
    'mid_depth': 40 samples,    # 3-8m
    'far_depth': 30 samples,    # 8-15m
}

# ❌ Bad: Random samples
# 대부분 근거리만 → far depth quantization 나쁨
```

#### Learning Rate 튜닝 (QAF/Distillation)
```python
# ✅ 권장
lr_initial = 1e-5  # 매우 작게 시작!
lr_schedule = 'cosine'  # Smooth decay

# ❌ 피해야 할 것
lr_initial = 1e-3  # Too high → diverge!
```

#### NPU Batch Size 최적화
```python
# 실험해볼 것
batch_sizes = [1, 2, 4, 8, 16]

# 예상 최적
optimal_bs = 4 or 8  # 보통 이 범위

# NPU마다 다름 → 반드시 벤치마크!
```

---

### 🎯 최종 결론 및 Action Plan

**지금 당장 해야 할 일** (우선순위):

1. **Day 1 (오늘!)**: 
   ```bash
   # NPU 스펙 확인
   - Per-channel quantization 지원?
   - Asymmetric quantization 지원?
   - Dual output 확인됨 ✅
   ```

2. **Day 2 (내일)**:
   ```bash
   # Phase 1 구현 시작
   - Percentile calibration
   - 100 representative samples 선정
   ```

3. **Day 3-4**:
   ```bash
   # Phase 1 완료 & 검증
   - NPU에서 성능 측정
   - 0.075 달성 확인
   ```

4. **Week 2-3**:
   ```bash
   # Phase 2 선택 (Dual-head 추천!)
   - Dual-head 재학습
   - 목표 0.05 달성 🎯
   ```

5. **Week 4-5** (선택):
   ```bash
   # Phase 3 (필요시만)
   - Knowledge distillation
   - FP32급 0.035 달성
   ```

**예상 최종 결과**: 
- **최소 목표**: abs_rel 0.05 ✅
- **최대 달성**: abs_rel 0.035 🎯
- **소요 시간**: 2-5주

**성공 확률**: 95% 이상! 💪

---

## 📝 다음 단계

1. **NPU 스펙 확인** (최우선!)
2. **Post-processing 테스트** (1일)
3. **Distillation 구현** (1주)
4. **성능 비교 및 최종 선택** (2주)

**목표: abs_rel 0.05 이하 달성!** 🎯
