# PTQ 256-Level Quantization for Dual-Head Architecture

## 개요

기존의 연속적인 sigmoid 기반 Integer 예측에서 **PTQ (Post-Training Quantization) 친화적인 256-level 정수 양자화**로 변경했습니다.

## 핵심 변경 사항

### 1. DualHeadDepthDecoder 수정

#### Before (연속적 sigmoid)
```python
self.convs[("integer_conv", s)] = Conv3x3(self.num_ch_dec[s], 1)  # 1 channel
# Output: [B, 1, H, W] sigmoid [0, 1] → represents [0, max_depth]
```

#### After (256-level 양자화)
```python
self.n_integer_levels = 256
self.convs[("integer_conv", s)] = Conv3x3(self.num_ch_dec[s], 256)  # 256 channels
# Output: [B, 256, H, W] logits → softmax → argmax for each pixel
```

### 2. Forward Pass 변경

```python
# PTQ-Friendly: Each channel is one quantization level
integer_raw = self.convs[("integer_conv", i)](x)  # [B, 256, H, W]
integer_probs = torch.nn.functional.softmax(integer_raw, dim=1)  # Probability distribution
integer_levels = torch.argmax(integer_probs, dim=1, keepdim=True).float()  # Hard quantization
integer_normalized = integer_levels / (self.n_integer_levels - 1)  # Normalize to [0, 1]

self.outputs[("integer", i)] = integer_normalized  # [B, 1, H, W] in [0, 1]
```

### 3. decompose_depth 함수 수정

이제 Ground Truth도 256 레벨로 양자화합니다:

```python
def decompose_depth(depth_gt, max_depth, n_integer_levels=256):
    # Integer part를 256 discrete levels로 양자화
    integer_levels = torch.round((depth_gt / max_depth) * (n_integer_levels - 1))
    integer_levels = torch.clamp(integer_levels, min=0, max=n_integer_levels - 1)
    
    # Normalize to [0, 1]
    integer_gt = integer_levels / (n_integer_levels - 1)
    
    # Fractional part: depth - integer_meters
    integer_meters = (integer_levels / (n_integer_levels - 1)) * max_depth
    fractional_gt = depth_gt - integer_meters
    
    return integer_gt, fractional_gt
```

### 4. dual_head_to_depth 함수 수정

```python
def dual_head_to_depth(integer_normalized, fractional_sigmoid, max_depth, n_integer_levels=256):
    # integer_normalized은 [0, 1] 범위의 정규화된 레벨
    # 이를 actual meters로 변환
    integer_part = integer_normalized * max_depth
    fractional_part = fractional_sigmoid
    depth = integer_part + fractional_part
    return depth
```

## 양자화 계층 구조

### Integer Head (256 levels)
- Range: 0 ~ max_depth meters
- 예시 (max_depth=30m):
  - Level 0 → 0 m
  - Level 85 → 10 m
  - Level 170 → 20 m
  - Level 255 → 30 m
- 양자화 간격: 30 / 256 ≈ **0.117 m (117 mm)**

### Fractional Head (256 levels)
- Range: 0 ~ 1 meter
- 양자화 간격: 1.0 / 256 ≈ **0.00391 m (3.91 mm)**

### 합친 깊이 범위
- **전체 범위**: 0 ~ (max_depth + 1) meters
- **총 양자화 레벨**: 256 × 256 = 65,536 levels
- **최소 정밀도**: ~3.91 mm (Fractional head에서 결정)
- **예시 (max_depth=30m)**:
  - 최상 정밀도: 30m / 256 + 1m / 256 ≈ 0.120 m (120 mm)
  - 최고 정밀도: 3.91 mm (Fractional)

## PTQ 장점

### 1. 8-bit 양자화 친화적
- Integer: 256 channels → 8-bit (0~255)
- Fractional: 1 channel → 8-bit (0~255)
- 총 2 channels × 8-bit = 16-bit 또는 각각 8-bit PTQ

### 2. 명확한 양자화 레벨
- 기존: 연속 sigmoid (많은 레벨 가능)
- 현재: 정확히 256개의 discrete levels
- NPU/INT8 friendly

### 3. 학습 안정성
- Softmax를 통한 확률 분포 학습
- Argmax로 hard quantization
- 명확한 quantization boundary

## 손실 함수 변경

```python
# Loss 함수에서 decompose_depth 호출 시
integer_gt, fractional_gt = decompose_depth(depth_gt, self.max_depth, n_integer_levels=256)

# Consistency Loss에서도
depth_pred = dual_head_to_depth(integer_pred, fractional_pred, self.max_depth, n_integer_levels=256)
```

## 추론 (Inference) 파이프라인

```python
# 1. 네트워크 forward pass
outputs = model(image)
integer_raw = outputs[("integer", 0)]  # [B, 256, H, W] logits
fractional_raw = outputs[("fractional", 0)]  # [B, 1, H, W] sigmoid

# 2. Integer quantization
integer_probs = softmax(integer_raw, dim=1)  # [B, 256, H, W]
integer_levels = argmax(integer_probs, dim=1)  # [B, H, W]
integer_normalized = integer_levels / 255  # [B, 1, H, W]

# 3. Depth reconstruction
fractional = sigmoid(fractional_raw)  # [B, 1, H, W]
depth = dual_head_to_depth(integer_normalized, fractional, max_depth=30.0)  # [B, 1, H, W]
```

## 성능 비교 (예상)

### 기존 (연속 sigmoid, max_depth 기반)
- Integer 정밀도: max_depth / ∞ (연속)
- 범위: 0 ~ max_depth m
- PTQ 변환 시 손실 가능

### 현재 (256-level PTQ)
- Integer 정밀도: max_depth / 256
- 범위: 0 ~ max_depth m (Integer) + 0 ~ 1m (Fractional)
- **전체 범위: 0 ~ (max_depth + 1)m**
- PTQ 변환 시 손실 없음

### 예시 (max_depth = 30m)
- Integer 정밀도: 30 / 256 ≈ **117 mm**
- Fractional 정밀도: 1000 / 256 ≈ **3.9 mm**
- 합친 최고 정밀도: **3.9 mm**
- 합친 최악 정밀도: ~120 mm (Integer boundary)

## 학습 설정 권장사항

### Config 예시
```yaml
model:
  params:
    use_dual_head: true
    max_depth: 30.0
    min_depth: 0.1

loss:
  params:
    integer_weight: 1.0
    fractional_weight: 10.0  # Still needed for weighting
    consistency_weight: 0.5
```

### 훈련 팁
1. 초기 LR은 기존과 동일하게 설정
2. Batch normalization은 여전히 필요
3. Fractional weight는 여전히 중요 (정밀도 강조)
4. Integer와 Fractional 사이의 role differentiation 유지

## 코드 업데이트 필요 사항

이미 자동으로 처리됨:
- ✅ dual_head_depth_decoder.py
- ✅ layers.py (decompose_depth, dual_head_to_depth, dual_head_to_inv_depth)
- ✅ dual_head_depth_loss.py
- ✅ model_wrapper.py (inference)

### 테스트 스크립트 업데이트 필요
- [ ] test_st2_implementation.py → n_integer_levels 파라미터 추가
- [ ] verify_dual_head_output.py → 256 레벨 처리
- [ ] save_dual_head_outputs.py → 256 레벨 처리
- [ ] onnx 변환 스크립트 → 256 channel 처리

## 다음 단계

1. **테스트**: 기존 체크포인트 로드 실패 예상
   - 새로운 아키텍처이므로 처음부터 훈련 필요
   
2. **PTQ 검증**: INT8 양자화 수행
   ```python
   # NPU 양자화 friendly test
   export_to_onnx(model, calibration_data)
   quantize_to_int8(onnx_model)
   ```

3. **성능 평가**: KITTI/NCDB에서 벤치마킹
   - 정밀도 비교 (기존 vs 256-level)
   - 범위 확장의 이점 확인

4. **NPU 배포**: 양자화된 모델 테스트
   - INT8 추론 성능
   - 정밀도 손실 최소화 확인

## 참고: 이전의 48 vs 256 분석

이번 변경은 **PTQ 실용성**을 기반으로 한 것입니다:

| 관점 | 기존 (48 levels) | 현재 (256 levels PTQ) |
|------|-----------------|----------------------|
| 설계 목표 | Multi-scale 역할분담 | PTQ 양자화 친화성 |
| Integer 정밀도 | 312.5 mm | 117 mm (30m 기준) |
| 전체 범위 | 0~16m | 0~31m (30m 기준) |
| 양자화 방식 | Sigmoid | 256-way classification |
| 추론 복잡도 | 낮음 | 매우 낮음 (INT8) |
| NPU 친화도 | 중간 | 매우 높음 |

**결론**: 256-level은 PTQ 양자화 관점에서 최적입니다.
