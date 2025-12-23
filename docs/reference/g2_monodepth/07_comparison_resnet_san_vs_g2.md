# ResNet-SAN vs G2-MonoDepth 비교 분석

## 1. 데이터 처리 비교

### 1.1 현재 ResNet-SAN 데이터 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                  ResNet-SAN Data Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📥 Input                                                       │
│  └── RGB 이미지 (3 channels)                                    │
│                                                                 │
│  🔄 Augmentation (transforms.py)                                │
│  ├── crop_sample()          : 이미지 crop                       │
│  ├── resize_sample()        : 리사이즈                          │
│  ├── duplicate_sample()     : original 복사                     │
│  ├── colorjitter_sample()   : brightness, contrast, saturation, hue │
│  └── to_tensor_sample()     : 텐서 변환                         │
│                                                                 │
│  📤 Output                                                      │
│  └── sample['rgb'], sample['depth'], sample['intrinsics']       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 G2-MonoDepth 데이터 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                  G2-MonoDepth Data Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📥 Input                                                       │
│  └── RGB + Sparse Depth + Hole Mask (5 channels)                │
│                                                                 │
│  🔄 Augmentation (data_tools.py)                                │
│  ├── horizontal_flip()      : 좌우 반전 (50%)                   │
│  ├── color_jitter()         : brightness, contrast, sat, hue    │
│  ├── random_sparsity()      : 0~100% sparsity 적용 ⭐            │
│  ├── point_hole()           : depth에 구멍 생성 (50%) ⭐         │
│  ├── point_noise()          : depth에 가우시안 노이즈 (50%) ⭐   │
│  └── point_blur()           : depth에 블러 적용 (50%) ⭐         │
│                                                                 │
│  📤 Output                                                      │
│  └── rgb, depth_gt, point, hole_point                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Augmentation 비교표

| Augmentation | ResNet-SAN | G2-MonoDepth | 비고 |
|--------------|------------|--------------|------|
| **기본** |
| Horizontal Flip | ❌ 없음 | ✅ 50% | 좌우 대칭 |
| Vertical Flip | ❌ | ❌ | 둘 다 미사용 |
| Rotation | ❌ | ❌ | 둘 다 미사용 |
| **Color** |
| Brightness | ✅ | ✅ | 유사 |
| Contrast | ✅ | ✅ | 유사 |
| Saturation | ✅ | ✅ | 유사 |
| Hue | ✅ | ✅ | 유사 |
| Color Matrix | ✅ | ❌ | ResNet-SAN만 |
| **Spatial** |
| Crop | ✅ | ❌ | ResNet-SAN만 |
| Resize | ✅ | ❌ (고정 크기) | |
| **Depth 관련** |
| Random Sparsity | ❌ | ✅ 0~100% | G2만 (RGB+X용) |
| Point Hole | ❌ | ✅ 50% | G2만 |
| Point Noise | ❌ | ✅ 50% | G2만 |
| Point Blur | ❌ | ✅ 50% | G2만 |
| **Advanced** |
| RandAugment | ⚠️ 구현됨 (미사용) | ❌ | |
| Random Erasing | ⚠️ 구현됨 (미사용) | ❌ | |
| MixUp | ⚠️ 구현됨 (미사용) | ❌ | |
| CutMix | ⚠️ 구현됨 (미사용) | ❌ | |

### 1.4 ResNet-SAN 데이터 처리의 부족한 점

#### ❌ 1. Horizontal Flip 미적용
```python
# 현재: colorjitter만 적용
def train_transforms(sample, image_shape, jittering, crop_train_borders):
    ...
    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering)  # Color만!
    ...
```

**문제점**: 
- 데이터 다양성 부족
- 좌우 대칭 학습 기회 손실

**제안**: Horizontal flip (50% 확률) 추가

#### ❌ 2. Depth Augmentation 전무
```
G2-MonoDepth는 depth에도 augmentation 적용:
- Noise 추가 → 센서 오차 시뮬레이션
- Blur 적용 → Edge bleeding 시뮬레이션  
- Hole 생성 → 센서 실패 시뮬레이션

ResNet-SAN은 RGB augmentation만 존재
```

**참고**: RGB-only이므로 sparse depth augmentation은 해당 없음.
하지만 GT depth에 noise를 추가하는 것은 robustness 향상에 도움될 수 있음.

#### ❌ 3. Advanced Augmentation 미활용
```python
# augmentations_kitti_compatible.py에 구현되어 있으나 미사용
class KITTIAdvancedTrainTransform:
    """RandAugment, RandomErasing, MixUp, CutMix"""
    # 구현됨 but 실제 학습에서 사용되지 않음
```

#### ❌ 4. Normalization 방식
| 항목 | ResNet-SAN | G2-MonoDepth |
|------|------------|--------------|
| RGB Normalize | ImageNet mean/std | 없음 (0-1 범위) |
| Depth Normalize | 없음 | Robust Standardization (MAD) |

---

## 2. Loss 함수 비교

### 2.1 현재 ResNet-SAN Loss 구조

```python
# SSISilogLoss (ssi_silog_loss.py)
total_loss = ssi_weight * SSI_Loss + silog_weight * Silog_Loss
```

#### SSI Loss (Scale-Shift Invariant)
```python
def compute_ssi_loss(self, pred_depth, gt_depth, mask):
    diff = (pred_depth[mask] - gt_depth[mask])
    diff2 = diff ** 2
    mean = diff.mean()
    var = diff2.mean() - mean ** 2
    ssi_loss = var + self.alpha * mean ** 2  # alpha=0.85
    return ssi_loss
```

수식:
$$\mathcal{L}_{SSI} = \text{Var}(d - \hat{d}) + \alpha \cdot \text{Mean}(d - \hat{d})^2$$

#### Silog Loss
```python
def compute_silog_loss(self, pred_depth, gt_depth, mask):
    log_diff = torch.log(pred_depth) - torch.log(gt_depth)
    silog1 = torch.mean(log_diff ** 2)
    silog2 = self.silog_ratio2 * (log_diff.mean() ** 2)  # ratio2=0.85
    silog_loss = torch.sqrt(silog1 - silog2 + 1e-8)
    return silog_loss
```

수식:
$$\mathcal{L}_{Silog} = \sqrt{E[(\log d - \log \hat{d})^2] - \lambda \cdot E[\log d - \log \hat{d}]^2}$$

### 2.2 G2-MonoDepth Loss 구조

```python
# 3-Term Loss
total_loss = loss_adepth + loss_rdepth + 0.5 * loss_rgrad
```

#### Absolute Depth Loss (L1)
```python
loss_adepth = L1(pred_depth, gt_depth, mask)
```

수식:
$$\mathcal{L}_{adepth} = \frac{1}{N}\sum_{i \in M} |d_i - \hat{d}_i|$$

#### Relative Depth Loss (Standardized L1)
```python
# Robust Standardization
sta_depth = (pred - mean_pred) / mad_pred
sta_gt = (gt - mean_gt) / mad_gt
loss_rdepth = L1(sta_depth, sta_gt, mask)
```

수식:
$$z_d = \frac{d - \mu_d}{\sigma_{MAD}}, \quad z_{\hat{d}} = \frac{\hat{d} - \mu_{\hat{d}}}{\sigma_{MAD}}$$
$$\mathcal{L}_{rdepth} = \frac{1}{N}\sum_{i \in M} |z_{d,i} - z_{\hat{d},i}|$$

#### Gradient Loss (Multi-Scale Sobel)
```python
def forward(self, depth, gt, mask):
    total_loss = 0
    for scale in [1, 2, 4, 8]:
        grad_pred = sobel_gradient(downsample(depth, scale))
        grad_gt = sobel_gradient(downsample(gt, scale))
        total_loss += L1(grad_pred, grad_gt, mask)
    return total_loss / 4
```

수식:
$$\mathcal{L}_{rgrad} = \frac{1}{4}\sum_{s \in \{1,2,4,8\}} \left( |G_x^s(d) - G_x^s(\hat{d})| + |G_y^s(d) - G_y^s(\hat{d})| \right)$$

### 2.3 Loss 비교표

| 항목 | ResNet-SAN (SSI-Silog) | G2-MonoDepth |
|------|------------------------|--------------|
| **구성** |
| Absolute Loss | Silog (log domain) | L1 (linear domain) |
| Relative Loss | SSI (variance based) | Standardized L1 |
| Gradient Loss | ❌ 없음 | ✅ Multi-scale Sobel |
| **특성** |
| Scale Invariance | SSI로 달성 | Standardization으로 달성 |
| Edge 보존 | ❌ 명시적 없음 | ✅ Gradient Loss |
| Outlier Robustness | Silog (log space) | MAD Standardization |
| **Weight** |
| Default | SSI:0.5 + Silog:0.5 | A:1.0 + R:1.0 + G:0.5 |

### 2.4 핵심 차이점 분석

#### 1️⃣ Gradient Loss의 유무

**ResNet-SAN**: Gradient Loss 없음
```
문제점:
- Edge 영역에서 depth가 blur될 수 있음
- 전체 맵의 구조적 일관성 부족 가능
- 객체 경계가 불분명해질 수 있음
```

**G2-MonoDepth**: Multi-Scale Gradient Loss
```
장점:
- Edge 선명도 유지
- 다양한 스케일의 구조 보존
- 전체 depth map 일관성 향상
```

#### 2️⃣ Relative Loss 계산 방식

**ResNet-SAN (SSI)**:
```python
# Variance + scaled mean²
diff = pred - gt
var = E[diff²] - E[diff]²
loss = var + alpha * mean²
```
- Variance 기반 (2차 통계량)
- alpha로 bias 페널티 조절
- 단일 연산으로 계산

**G2-MonoDepth (Standardized L1)**:
```python
# MAD 기반 정규화 후 L1
z_pred = (pred - mean) / mad
z_gt = (gt - mean) / mad  
loss = L1(z_pred, z_gt)
```
- MAD (Mean Absolute Deviation) 사용
- Outlier에 더 강건
- 명시적 standardization

#### 3️⃣ Absolute Loss 계산 방식

**ResNet-SAN (Silog)**:
```python
# Log space에서 계산
log_diff = log(pred) - log(gt)
loss = sqrt(E[log_diff²] - λ·E[log_diff]²)
```
- Log space → 상대적 오차에 집중
- 멀리 있는 객체 오차 완화
- Scale-invariant 특성

**G2-MonoDepth (L1)**:
```python
# Linear space에서 계산
loss = mean(|pred - gt|)
```
- Linear space → 절대 오차
- 단순하고 직관적
- 근거리 정확도 중시

---

## 3. 상세 분석

### 3.1 SSI Loss vs Relative (Standardized) Loss

| 측면 | SSI | Standardized L1 |
|------|-----|-----------------|
| **수학적 기반** | Variance 최소화 | Distribution 정합 |
| **Scale 보정** | Implicit (variance) | Explicit (÷std) |
| **Shift 보정** | alpha 파라미터 | mean 제거 |
| **Outlier 처리** | 제곱으로 민감 | MAD로 강건 |
| **계산 복잡도** | 낮음 | 중간 (2회 통계 계산) |

#### SSI Loss 특성:
```
장점:
- 단일 수식으로 scale-shift invariance
- 계산 효율적
- 잘 연구된 방법

단점:
- 제곱 연산으로 outlier에 민감
- alpha 튜닝 필요
```

#### Standardized L1 특성:
```
장점:
- MAD로 outlier에 강건
- 직관적인 해석 (z-score 비교)
- 분포 정합 관점

단점:
- 추가 통계 계산 필요
- pred와 gt 각각 정규화 필요
```

### 3.2 Silog Loss vs L1 Loss

| 측면 | Silog | L1 |
|------|-------|-----|
| **Domain** | Log | Linear |
| **원거리 객체** | 오차 완화 | 오차 그대로 |
| **근거리 객체** | 오차 증폭 | 오차 그대로 |
| **수치 안정성** | log(0) 위험 | 안전 |

#### Silog 특성:
```
log(pred) - log(gt) = log(pred/gt)

pred=10, gt=9  → log(10/9) ≈ 0.105  (11% 상대 오차)
pred=100, gt=90 → log(100/90) ≈ 0.105  (동일!)

→ 상대적 오차에 집중, 거리에 관계없이 공평한 학습
```

#### L1 특성:
```
|pred - gt|

pred=10, gt=9   → |10-9| = 1
pred=100, gt=90 → |100-90| = 10

→ 절대 오차 기준, 근거리 정확도에 유리
```

### 3.3 Gradient Loss 부재의 영향

```
현재 ResNet-SAN:
┌────────────────────┐
│  SSI Loss          │  → 전체 분포 정합
│  +                 │
│  Silog Loss        │  → 스케일 정확도
│                    │
│  = Total Loss      │  → Edge 정보 없음!
└────────────────────┘

결과:
- 객체 경계에서 depth 불연속성 blur
- 전체 맵의 구조적 일관성 부족
- "전체 맵이 안 좋다"는 문제와 연관 가능
```

```
G2-MonoDepth:
┌────────────────────┐
│  Absolute Loss     │  → 절대 정확도
│  +                 │
│  Relative Loss     │  → 상대 분포
│  +                 │
│  Gradient Loss     │  → Edge 보존!
│                    │
│  = Total Loss      │
└────────────────────┘

결과:
- 객체 경계 선명도 유지
- Multi-scale 구조 보존
- 전체 맵 일관성 향상
```

---

## 4. 개선 제안

### 4.1 데이터 처리 개선

#### Priority 1: Horizontal Flip 추가
```python
# transforms.py 수정
def train_transforms(sample, image_shape, jittering, crop_train_borders):
    ...
    # 추가: Horizontal flip (50%)
    if random.random() < 0.5:
        sample = horizontal_flip_sample(sample)
    ...
```

#### Priority 2: Advanced Augmentation 활성화
```yaml
# YAML config
augmentation:
  use_advanced: true
  rand_augment_n: 2
  rand_augment_m: 9
```

### 4.2 Loss 개선

#### Priority 1: Gradient Loss 추가 (강력 권장)
```python
# 새 파일: gradient_loss.py
class MultiScaleGradientLoss(nn.Module):
    def __init__(self, scales=[1, 2, 4, 8]):
        ...
    
    def forward(self, pred, gt, mask):
        total = 0
        for s in self.scales:
            grad_p = self.sobel(F.avg_pool2d(pred, s))
            grad_g = self.sobel(F.avg_pool2d(gt, s))
            total += F.l1_loss(grad_p * mask, grad_g * mask)
        return total / len(self.scales)
```

```yaml
# YAML config
loss:
  supervised_method: ssi-silog
  ssi_weight: 0.4
  silog_weight: 0.4
  gradient_weight: 0.2  # 새로 추가
```

#### Priority 2: Robust Standardization 검토
현재 SSI Loss도 효과적이나, outlier가 많은 환경에서는 MAD 기반 standardization 검토

---

## 5. 요약

### 5.1 데이터 처리 Gap

| 항목 | 현재 상태 | 제안 |
|------|----------|------|
| Horizontal Flip | ❌ 없음 | ✅ 추가 (쉬움) |
| Advanced Aug | 구현됨/미사용 | ✅ 활성화 검토 |
| Depth Aug | N/A (RGB-only) | - |

### 5.2 Loss Gap

| 항목 | 현재 상태 | 제안 |
|------|----------|------|
| Gradient Loss | ❌ 없음 | ✅ 추가 (권장) |
| Robust Normalization | SSI 사용 | 현재로 충분 |
| Absolute Loss | Silog | 현재로 충분 |

### 5.3 최종 권장사항

```
🔴 즉시 적용:
   1. Gradient Loss 추가 → 전체 맵 일관성 개선

🟡 검토 후 적용:
   2. Horizontal Flip 추가 → 데이터 다양성
   3. Advanced Augmentation 활성화 → Robustness

🟢 현재 유지:
   - SSI Loss (효과적인 scale-invariance)
   - Silog Loss (log-domain accuracy)
```

---

## 6. 참고: 코드 위치

| 파일 | 역할 |
|------|------|
| `packnet_sfm/datasets/transforms.py` | Transform 정의 |
| `packnet_sfm/datasets/augmentations.py` | Augmentation 함수 |
| `packnet_sfm/datasets/augmentations_kitti_compatible.py` | Advanced Aug (미사용) |
| `packnet_sfm/losses/supervised_loss.py` | Loss 함수 팩토리 |
| `packnet_sfm/losses/ssi_silog_loss.py` | SSI-Silog Loss 구현 |
