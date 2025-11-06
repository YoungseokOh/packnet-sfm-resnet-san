# 🚨 CRITICAL BUG FOUND - Sigmoid to Depth Mapping은 REVERSED!

## 문제 발견

### 현재 동작 (잘못됨):
```python
min_inv = 1.0 / max_depth  # 1/80 = 0.0125
max_inv = 1.0 / min_depth  # 1/0.05 = 20

inv_depth = min_inv + (max_inv - min_inv) × sigmoid
# sigmoid=0 → inv_depth=0.0125 → depth=80m (멀리!)
# sigmoid=0.5 → inv_depth=10 → depth=0.1m (가까이!)
# sigmoid=1 → inv_depth=20 → depth=0.05m (매우 가까이!)
```

### 문제:
- **sigmoid가 증가하면 depth가 감소** (직관에 반대!)
- Network는 먼 물체를 예측하려면 sigmoid=0을 출력해야 함
- 하지만 대부분의 depth network는 sigmoid=1이 먼 거리를 의미함

### 실제 학습 결과:
```
GT depth range: [0.05, 58.72]m
Pred depth range: [0.05, 2.24]m  ← 너무 작음!

이는 sigmoid가 0.8~1.0 범위에 몰려있다는 뜻
→ Network가 가까운 거리만 예측하고 있음
```

## 해결책

### Option 1: Sigmoid 매핑 반전 (권장)

```python
def sigmoid_to_depth_linear(sigmoid_output, min_depth=0.05, max_depth=80.0):
    # sigmoid=0 → min_depth (가까이)
    # sigmoid=1 → max_depth (멀리)
    depth = min_depth + (max_depth - min_depth) * sigmoid_output
    return depth
```

**장점**:
- 직관적: sigmoid 증가 = depth 증가
- Network 학습이 쉬움
- 기존 많은 depth network와 동일한 방식

**단점**:
- Depth space에서 linear interpolation (inv_depth space가 아님)
- Quantization 시 non-uniform error (하지만 이건 원래도 그랬음)

### Option 2: Inv-depth mapping 유지하되 순서 반전

```python
def sigmoid_to_depth_linear(sigmoid_output, min_depth=0.05, max_depth=80.0):
    # sigmoid를 반전
    inv_sigmoid = 1.0 - sigmoid_output
    
    min_inv = 1.0 / max_depth
    max_inv = 1.0 / min_depth
    
    inv_depth = min_inv + (max_inv - min_inv) * inv_sigmoid
    depth = 1.0 / (inv_depth + 1e-8)
    return depth
```

**장점**:
- Inv-depth space에서 균일한 sampling 유지
- 기존 이론과 일치

**단점**:
- Sigmoid를 반전하는 것이 비직관적

### Option 3: 원본 코드 구조로 복원

기존에 `disp_to_inv` 함수가 있었는데, 그 코드를 확인해봅시다.

## 권장 사항

**즉시 Option 1로 수정해야 합니다!**

현재 network가 학습이 안되는 이유:
1. Sigmoid가 0.8~1.0 범위에서만 작동 (가까운 거리만)
2. Loss가 제대로 계산되지 않음
3. Gradient가 올바르게 전파되지 않음

이것은 **training을 멈추고 즉시 수정해야 하는 critical bug**입니다!
