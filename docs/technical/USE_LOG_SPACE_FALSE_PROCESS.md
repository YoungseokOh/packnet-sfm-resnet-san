# use_log_space=False일 때 정확한 처리 흐름

> **상태**: `use_log_space: false` (현재 설정)
> 
> **영향**: LINEAR SPACE 변환 사용

---

## 📊 전체 흐름 다이어그램

```
입력 이미지 (RGB)
    ↓
[모델] ResNetSAN01
    ↓
Sigmoid 출력 (0.0 ~ 1.0)  ← 모델의 출력값
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[선형 변환] (use_log_space=False일 때)  ← ⭐ 지금 여기!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
역-깊이 (0.0125 ~ 20.0)
    ↓
깊이 (0.05m ~ 80m)
    ↓
최종 깊이 맵 (결과)
```

---

## 🔧 단계별 정확한 처리

### **단계 1: Sigmoid 출력 (모델)**

```python
# 모델에서 출력되는 값
sigmoid_output = model(rgb_image)  # 형태: (B, 1, H, W)
# 값 범위: 0.0 ~ 1.0
```

**근거**: 
- ResNetSAN01은 마지막 계층에서 항상 Sigmoid를 적용
- 값 범위가 정해진 [0, 1] 범위

---

### **단계 2: Sigmoid → 역-깊이 (선형 변환)**

```python
# 파일: packnet_sfm/utils/post_process_depth.py
def sigmoid_to_inv_depth(sigmoid_output, min_depth=0.05, max_depth=80.0, 
                         use_log_space=False):  # ← FALSE!
    
    # 상수 계산
    min_inv = 1.0 / max_depth  # 1 / 80.0 = 0.0125
    max_inv = 1.0 / min_depth  # 1 / 0.05 = 20.0
    
    # use_log_space=False이므로 선형 변환 실행
    inv_depth = min_inv + (max_inv - min_inv) * sigmoid_output
    #          =  0.0125 + (20.0 - 0.0125) * sigmoid_output
    #          =  0.0125 + 19.9875 * sigmoid_output
    
    return inv_depth
```

**수식 (선형)**:
$$\text{inv\_depth} = 0.0125 + 19.9875 \times \sigma$$

| Sigmoid 값 | 계산 | 역-깊이 | 깊이 |
|-----------|------|---------|------|
| 0.0 | 0.0125 + 0 | 0.0125 | 80.0m (먼 거리) |
| 0.5 | 0.0125 + 9.9937 | 10.0062 | 0.1m (가까움) |
| 1.0 | 0.0125 + 19.9875 | 20.0 | 0.05m (최가까움) |

**근거**:
- `use_log_space=False` 조건 → `if use_log_space:` 블록을 건너뜀
- 직선 보간법 (linear interpolation) 사용
- 최소 깊이(0.05m)에서 최대 깊이(80m) 범위로 매핑

---

### **단계 3: 역-깊이 → 깊이 (역변환)**

```python
# 파일: packnet_sfm/utils/depth.py
def inv2depth(inv_depth):
    depth = 1.0 / inv_depth
    return depth
```

**계산**:
```
깊이 = 1.0 / 역-깊이
```

| 역-깊이 | 계산 | 깊이 |
|---------|------|------|
| 0.0125 | 1.0 / 0.0125 | 80.0m |
| 10.0062 | 1.0 / 10.0062 | 0.0999m ≈ 0.1m |
| 20.0 | 1.0 / 20.0 | 0.05m |

---

## 🎯 학습 중 적용 위치

```python
# 파일: packnet_sfm/models/SemiSupCompletionModel.py (line 460~470)
def forward(self, batch):
    # ... 모델 연산 ...
    sigmoid_outputs = self_sup_output['inv_depths']  # Sigmoid [0, 1]
    
    # ⭐ 선형 변환 적용 (use_log_space=False)
    bounded_inv_depths = [
        sigmoid_to_inv_depth(sig, 
                           min_depth=0.05, 
                           max_depth=80.0, 
                           use_log_space=self.use_log_space)  # ← False!
        for sig in sigmoid_outputs
    ]
    
    # Loss 계산 (역-깊이 도메인에서)
    sup_output = self.supervised_loss(
        bounded_inv_depths, 
        depth2inv(batch['depth'])  # GT도 역-깊이로 변환
    )
```

**근거**:
- `self.use_log_space`는 config에서 읽어온 값 (False)
- 모든 배치 데이터에 동일한 변환 적용

---

## 🔍 평가 중 적용 위치

```python
# 파일: packnet_sfm/models/model_wrapper.py (line 625~635)
def evaluate_depth(self, batch):
    sigmoid_outputs = self.model(batch)['inv_depths']
    sigmoid0 = sigmoid_outputs[0]  # 첫 번째 스케일
    
    # ⭐ 모델에서 저장된 use_log_space 값 읽기
    use_log_space = getattr(self.model, 'use_log_space', False)
    
    # ⭐ 동일한 선형 변환 적용
    inv_depth = sigmoid_to_inv_depth(
        sigmoid0, 
        min_depth=0.05, 
        max_depth=80.0, 
        use_log_space=use_log_space  # ← False!
    )
    
    # 깊이로 변환
    depth_pred = inv2depth(inv_depth)
    
    # 메트릭 계산
    metrics = compute_metrics(depth_pred, batch['gt_depth'])
```

**근거**:
- `getattr(self.model, 'use_log_space', False)` = 모델에서 저장된 값 읽기
- **학습과 평가가 반드시 같은 변환 사용**

---

## ⚠️ 중요: LINEAR vs LOG 차이

### **Linear (use_log_space=False) - 현재**

```
직선 보간:  inv_depth = 0.0125 + 19.9875 × sigmoid

특징:
  ✓ 간단한 선형 관계
  ✓ Sigmoid=0 → 80m (먼 거리)
  ✓ Sigmoid=1 → 0.05m (가까움)
  ✗ 중간값(Sigmoid=0.5) → 0.1m (선형이라 깊이 편차 큼)
  ✗ INT8 양자화 시 오류 큼 (~39%)
```

### **Log (use_log_space=True) - 미사용**

```
로그 보간:  inv_depth = exp(log(0.0125) + 3.178 × sigmoid)

특징:
  ✓ 기하학적 평균
  ✓ Sigmoid=0.5 → 2.0m (균형잡힌 중간값)
  ✓ INT8 양자화 시 오류 작음 (~3%)
  ✗ 계산이 복잡함
  ✗ 원거리 데이터 많은 경우에만 유리
```

---

## 📌 결론

**`use_log_space=false` 선택 시**:

1. **선형 변환 사용**: Sigmoid → 역-깊이 (직선 공식)
2. **계산**: `inv_depth = 0.0125 + 19.9875 × sigmoid`
3. **깊이 범위**: 0.05m ~ 80m
4. **근거**: 
   - NCDB 데이터 98% 픽셀이 0~5m 범위 (근거리 중심)
   - 근거리 데이터에는 LINEAR 모드가 최적
   - 코드 간결성

---

## 🔗 관련 파일

| 파일 | 역할 |
|------|------|
| `packnet_sfm/utils/post_process_depth.py` | `sigmoid_to_inv_depth()` 구현 |
| `packnet_sfm/models/SemiSupCompletionModel.py` | 학습 중 변환 적용 |
| `packnet_sfm/models/model_wrapper.py` | 평가 중 변환 적용 |
| `packnet_sfm/utils/depth.py` | `inv2depth()` 역변환 |
| `configs/train_resnet_san_ncdb_640x384.yaml` | 파라미터 설정 위치 |

