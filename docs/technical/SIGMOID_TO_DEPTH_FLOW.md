# 🔧 Sigmoid to Depth Conversion Flow

## 문제 인식

모델이 출력하는 **sigmoid [0, 1]**을 **어떻게 depth로 변환하느냐**가 학습과 평가에서 일관되어야 합니다.

## 기존 방식 (Original PackNet-SFM)

### 모델 출력
- DepthNet → **inverse depth** (이미 물리적 의미를 가진 값)
- Range: [1/max_depth, 1/min_depth] 정도

### 평가
```python
inv_depths = model(batch)['inv_depths']  # Already inverse depth
depth = inv2depth(inv_depths[0])  # 1 / inv_depth
compute_depth_metrics(gt=depth_gt, pred=depth)  # Depth space
```

## 현재 구현 (ResNetSAN01 with Sigmoid)

### 모델 출력
- DepthNet → **sigmoid [0, 1]** (raw sigmoid, 물리적 의미 없음)
- 이것을 **bounded inverse depth로 해석**해야 함

### 핵심: Bounded Inverse Depth 변환

```python
def sigmoid_to_inv_depth(sigmoid, min_depth, max_depth):
    """
    Convert sigmoid [0, 1] to bounded inverse depth.
    
    Formula:
        min_inv = 1 / max_depth  # far
        max_inv = 1 / min_depth  # near
        inv_depth = min_inv + (max_inv - min_inv) × sigmoid
    
    Example (min=0.05, max=80):
        sigmoid=0.0 → inv_depth=0.0125 → depth=80m (far)
        sigmoid=0.5 → inv_depth=10.0   → depth=0.1m (mid)
        sigmoid=1.0 → inv_depth=20.0   → depth=0.05m (near)
    """
    min_inv = 1.0 / max_depth
    max_inv = 1.0 / min_depth
    return min_inv + (max_inv - min_inv) * sigmoid
```

## 올바른 데이터 흐름

### 학습 시 (Training)

```python
# Model output
sigmoid = model(batch)['inv_depths']  # [0, 1]

# ✅ Convert to bounded inverse depth
from packnet_sfm.utils.post_process_depth import sigmoid_to_inv_depth
bounded_inv = sigmoid_to_inv_depth(sigmoid, min_depth=0.05, max_depth=80.0)
# bounded_inv: [0.0125, 20.0]

# Convert GT depth to inverse depth
gt_inv = depth2inv(batch['depth'])  # 1 / depth

# Compute loss in inverse depth space
loss = ssi_silog_loss(pred_inv=bounded_inv, gt_inv=gt_inv)
```

### 평가 시 (Evaluation)

```python
# Model output
sigmoid = model(batch)['inv_depths']  # [0, 1]

# ✅ Convert to bounded inverse depth (SAME as training!)
bounded_inv = sigmoid_to_inv_depth(sigmoid, min_depth=0.05, max_depth=80.0)

# Convert to depth for metrics
depth_pred = inv2depth(bounded_inv)  # 1 / bounded_inv

# Compute metrics in depth space (traditional)
metrics = compute_depth_metrics(gt=depth_gt, pred=depth_pred)
```

## 왜 Bounded Inverse Depth인가?

### 1. Monodepth Convention
- 원본 Monodepth2 논문: disparity (inverse depth) 예측
- Network outputs disparity, not depth directly
- Better for self-supervised learning

### 2. Range Control
- **Unbounded**: sigmoid [0, 1] → inverse depth [0, ∞] → depth [∞, 0]
  - 문제: sigmoid=0일 때 depth=∞ (발산)
- **Bounded**: sigmoid [0, 1] → inverse depth [1/80, 1/0.05] → depth [0.05, 80]
  - 해결: 항상 유효한 depth 범위 보장

### 3. 선형성 (Linearity)
```
Depth space (비선형):
  0.05m → 1m: 0.95m 차이
  10m → 80m: 70m 차이 (불균등!)

Inverse depth space (더 선형적):
  20 → 1.0: 19 차이
  0.1 → 0.0125: ~0.0875 차이 (더 균등)
```

### 4. Gradient Flow
- Inverse depth: 먼 물체(10m~80m)의 gradient가 더 안정적
- Depth: 먼 물체의 gradient 소실 문제

## 주요 함수

### sigmoid_to_inv_depth (Training & Evaluation)
```python
# Used in both training and evaluation
bounded_inv = sigmoid_to_inv_depth(sigmoid, min_depth, max_depth)
```

### sigmoid_to_depth_linear (Comparison only)
```python
# Direct conversion for comparison
# Same as: inv2depth(sigmoid_to_inv_depth(...))
depth = sigmoid_to_depth_linear(sigmoid, min_depth, max_depth)
```

### sigmoid_to_depth_log (Comparison only)
```python
# Log-space conversion for INT8 quantization study
depth = sigmoid_to_depth_log(sigmoid, min_depth, max_depth)
```

## 평가 Metrics

### MAIN (Primary)
- **depth**: sigmoid → bounded_inv → depth (no GT scale)
- **depth_gt**: sigmoid → bounded_inv → depth (with GT median scale)

### LINEAR (Comparison)
- **depth_linear**: Direct linear conversion
- **depth_linear_gt**: With GT median scale

### LOG (Comparison)
- **depth_log**: Log-space conversion
- **depth_log_gt**: With GT median scale

## 일관성 검증

### 학습 시:
```python
sigmoid [0, 1]
  ↓ sigmoid_to_inv_depth()
bounded_inv [1/80, 1/0.05]
  ↓ SSI Loss (inverse depth space)
Loss
```

### 평가 시:
```python
sigmoid [0, 1]
  ↓ sigmoid_to_inv_depth() ← SAME FUNCTION!
bounded_inv [1/80, 1/0.05]
  ↓ inv2depth()
depth [0.05, 80]
  ↓ compute_depth_metrics()
Metrics
```

**핵심**: `sigmoid_to_inv_depth()` 함수가 학습과 평가에서 **동일하게** 사용됨!

## 수정 파일 목록

1. **packnet_sfm/utils/post_process_depth.py**
   - 추가: `sigmoid_to_inv_depth()` 함수

2. **packnet_sfm/models/SemiSupCompletionModel.py**
   - 수정: sigmoid → bounded_inv 변환 추가 (학습 시)

3. **packnet_sfm/models/model_wrapper.py**
   - 수정: sigmoid → bounded_inv → depth 변환 (평가 시)
   - 추가: Main metrics (depth, depth_gt)

4. **packnet_sfm/losses/ssi_silog_loss.py**
   - 수정: Silog Loss 공식 오류 수정 (× ratio 제거)

## 기대 효과

- ✅ 학습과 평가의 일관성 확보
- ✅ Loss가 정상 범위로 감소 (0.15~0.25 예상)
- ✅ Metrics가 의미있는 값 산출
- ✅ Bounded range로 안정적 학습
