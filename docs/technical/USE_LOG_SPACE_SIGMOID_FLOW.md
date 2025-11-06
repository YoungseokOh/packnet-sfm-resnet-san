# USE_LOG_SPACE=True일 때 Sigmoid 다음 코드 흐름

## 🎯 개요

`use_log_space=True`로 설정하면, Sigmoid 출력값 이후에 **로그 공간에서 깊이 변환**이 발생합니다.

---

## 📊 전체 파이프라인

```
Model Output: Sigmoid [0, 1]
    ↓
sigmoid_to_inv_depth() 
  (use_log_space=True)
    ↓
Inverse Depth (Log Space)
    ↓
inv2depth()
    ↓
Depth (m)
```

---

## 🔍 단계별 상세 분석

### **1단계: 모델 출력 → Sigmoid [0, 1]**

```python
# 파일: packnet_sfm/networks/depth/ResNetSAN01.py
# 모델이 디코더의 마지막에서 Sigmoid 활성화
sigmoid_output = model(rgb)  # shape: [B, 1, H, W], 값 범위: [0, 1]

# 예시
# sigmoid=0.0  → 원거리
# sigmoid=0.5  → 중거리
# sigmoid=1.0  → 근거리
```

---

### **2단계: sigmoid_to_inv_depth() - 핵심 변환**

**파일 위치**: `packnet_sfm/utils/post_process_depth.py` (라인 12-71)

#### **2-1) Linear 모드 (use_log_space=False) - 기본값**

```python
def sigmoid_to_inv_depth(sigmoid_output, min_depth=0.05, max_depth=80.0, use_log_space=False):
    """
    선형 공간 변환:
        inv_depth = min_inv + (max_inv - min_inv) × sigmoid
    """
    # 설정값
    min_depth = 0.05      # 5cm (매우 근거리)
    max_depth = 80.0      # 80m (원거리)
    
    # 역깊이 범위 계산
    min_inv = 1.0 / 80.0  # = 0.0125
    max_inv = 1.0 / 0.05  # = 20.0
    
    if not use_log_space:  # ← LINEAR 모드
        # 직선 보간
        inv_depth = 0.0125 + (20.0 - 0.0125) × sigmoid
        
        # 예시 계산
        sigmoid=0.0  → inv_depth = 0.0125     (1/80m = 80m 거리)
        sigmoid=0.5  → inv_depth = 10.00625   (1/10 = 0.1m 근거리!)
        sigmoid=1.0  → inv_depth = 20.0       (1/20 = 0.05m 극근거리)
```

**특징**:
- ✅ 단순한 선형 보간
- ❌ Sigmoid 극소 범위(0-0.1)에 집중 (수치 불안정)
- ❌ INT8 양자화에서 39% 오차

---

#### **2-2) Log 모드 (use_log_space=True) - 새로운 기능**

```python
if use_log_space:  # ← LOG 모드
    # 로그 공간에서 보간
    log_min_inv = log(0.0125) = -4.605
    log_max_inv = log(20.0) = 2.996
    
    log_inv_depth = -4.605 + (2.996 - (-4.605)) × sigmoid
                  = -4.605 + 7.601 × sigmoid
    
    inv_depth = exp(log_inv_depth)
    
    # 예시 계산
    sigmoid=0.0   → log_inv = -4.605         → inv_depth = 0.0125  (80m)
    sigmoid=0.5   → log_inv = -4.605 + 3.801 = -0.804
                  → inv_depth = exp(-0.804) = 0.447  (depth = 2.24m)
    sigmoid=1.0   → log_inv = 2.996          → inv_depth = 20.0   (0.05m)
```

**특징**:
- ✅ 기하학적 균등 분포 (로그 스케일)
- ✅ Sigmoid 정상 범위(0.3-0.7)에서 작동 (수치 안정)
- ✅ INT8 양자화에서 3% 오차 (13배 개선!)

---

### **3단계: inv2depth() - 역깊이 → 깊이 변환**

**파일 위치**: `packnet_sfm/utils/depth.py` (라인 123-140)

```python
def inv2depth(inv_depth):
    """역깊이를 깊이로 변환"""
    depth = 1.0 / inv_depth  # 간단하게 역수 취함
    return depth

# 예시 (Log 모드)
# inv_depth = 0.447 → depth = 1/0.447 = 2.24m
# inv_depth = 20.0  → depth = 1/20.0 = 0.05m
```

---

## 📈 Linear vs Log 비교표

| 항목 | Linear (use_log_space=False) | Log (use_log_space=True) |
|------|------|------|
| **Sigmoid 범위** | 0.0095-0.0995 (극소) | 0.3941-0.6971 (정상) |
| **sigmoid=0** | 80m | 80m |
| **sigmoid=0.5** | **0.1m** (근거리) | **2.24m** (기하평균) |
| **sigmoid=1.0** | 0.05m | 0.05m |
| **INT8 오차** | ❌ 39% | ✅ 3% |
| **NCDB 적합성** | ✅ 우수 | ❌ 불안정 |

---

## 🔄 학습 vs 평가 일관성

### **학습 시간 (Forward Pass)**

```python
# 파일: packnet_sfm/models/SemiSupCompletionModel.py (라인 459-479)
def forward(self, batch):
    # 모델 출력: sigmoid [0, 1]
    sigmoid_outputs = self_sup_output['inv_depths']
    
    # ★ CRITICAL: sigmoid → inverse depth 변환
    from packnet_sfm.utils.post_process_depth import sigmoid_to_inv_depth
    
    bounded_inv_depths = [
        sigmoid_to_inv_depth(
            sig, 
            self.min_depth,      # 0.05 (m)
            self.max_depth,      # 80.0 (m)
            use_log_space=self.use_log_space  # ← 여기서 적용!
        )
        for sig in sigmoid_outputs
    ]
    
    # 손실 함수에 전달
    sup_output = self.supervised_loss(
        bounded_inv_depths,
        depth2inv(batch['depth']),  # GT를 inverse depth로 변환
        ...
    )
```

### **평가 시간 (Evaluation)**

```python
# 파일: packnet_sfm/models/model_wrapper.py (라인 631-645)
def evaluate_depth(self, batch):
    # 모델 출력: sigmoid [0, 1]
    sigmoid0 = self.model(batch)['inv_depths'][0]
    
    # ★ CRITICAL: 학습과 동일한 변환 적용
    use_log_space = getattr(self.model, 'use_log_space', False)
    from packnet_sfm.utils.post_process_depth import sigmoid_to_inv_depth
    
    inv_depth = sigmoid_to_inv_depth(
        sigmoid0,
        min_depth,
        max_depth,
        use_log_space=use_log_space  # ← 학습과 같은 설정!
    )
    
    # 깊이로 변환
    from packnet_sfm.utils.depth import inv2depth
    depth_pred = inv2depth(inv_depth)
    
    # 메트릭 계산
    ...
```

---

## ⚠️ 중요: 일관성 필요

**학습과 평가에서 `use_log_space` 설정이 반드시 같아야 합니다!**

### ❌ 잘못된 예시 (일관성 없음)

```python
# 학습: LINEAR 모드
# sigmoid → inv_depth (선형)
# Loss 계산

# 평가: LOG 모드
# sigmoid → inv_depth (로그)
# 메트릭 계산
# → 완전히 다른 결과! (abs_rel = 40.101 같은 이상값)
```

### ✅ 올바른 예시 (일관성 있음)

```python
# 학습: LINEAR 모드
# sigmoid → inv_depth (선형)
# Loss 계산

# 평가: LINEAR 모드
# sigmoid → inv_depth (선형)
# 메트릭 계산
# → 일관된 결과! (abs_rel = 0.039)
```

---

## 🎯 NCDB 권장 설정

```yaml
# configs/train_resnet_san_ncdb_640x384.yaml
model:
  params:
    min_depth: 0.05      # 5cm
    max_depth: 80.0      # 80m
    use_log_space: False # ← LINEAR 모드 권장!
    
    # 이유:
    # - NCDB는 98% 픽셀이 0-5m (극근거리)
    # - LINEAR가 이 범위에 최적화
    # - E29 성능: abs_rel=0.039 (우수)
```

---

## 💡 요약

1. **`use_log_space=False` (Linear)**: 기본값, NCDB에 최적화
2. **`use_log_space=True` (Log)**: 원거리 데이터 많은 데이터셋용
3. **핵심**: 학습-평가에서 같은 설정 필수!
4. **미래**: INT8 양자화 시 Log 모드가 13배 더 정확
