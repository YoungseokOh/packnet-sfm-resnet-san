# 깊이 추정을 위한 Scale-Adaptive Loss

## 📚 개요

**Scale-Adaptive Loss**는 주로 **자기지도 단안 깊이 추정(self-supervised monocular depth estimation)**에서 사용되는 손실 함수로, 근본적인 scale ambiguity 문제를 해결하기 위해 고안되었습니다. 절대 scale 정보 없이도 깊이 추정 모델을 학습할 수 있게 해줍니다.

## 🎯 문제: Scale Ambiguity

단안(single camera) 깊이 추정에서는 **절대 scale을 알 수 없는 근본적인 문제**가 있습니다:

- 단일 카메라만으로는 추가 정보 없이 실제 거리를 결정할 수 없음
- 예측된 깊이와 ground truth 깊이가 scale에서 차이날 수 있음
- 전통적인 L1/L2 손실은 이러한 scale 차이에 매우 민감함

### 수학적 정식화

단안 비전에서 투영 방정식은 다음과 같습니다:

```
p = K [R|t] P

where:
  p: 2D 이미지 포인트 (u, v, 1)ᵀ
  P: 3D 월드 포인트 (X, Y, Z, 1)ᵀ
  K: 내부 파라미터 행렬 (3×3)
  [R|t]: 외부 파라미터 행렬 (3×4)
```

**모호성:**
```
p = K [R|t] (λ·P)  여기서 λ는 임의의 scale

→ 같은 이미지 p이지만, 깊이 Z는 λ배만큼 차이남
```

---

## 💡 해결책: Scale-Adaptive Loss

Scale-Adaptive Loss는 예측과 ground truth 사이의 **scale 차이를 자동으로 보정**합니다.

### 핵심 공식

```
L_scale_adaptive = min_s (1/N) Σ |s·d_pred - d_gt|²

where s* = argmin_s Σ (s·d_pred - d_gt)²
```

**의미:** 최적의 scale factor `s*`를 찾아 예측을 ground truth에 정렬합니다.

---

## 🔬 최적 Scale Factor 계산

### 방법 1: Median Scaling (가장 일반적)

```
s* = median(d_gt) / median(d_pred)
```

**장점:**
- ✅ 이상치(outlier)에 강건함
- ✅ Scale invariance 보장
- ✅ 계산 효율적

**왜 평균이 아닌 중앙값?**
- 극단적인 깊이 값에 더 강건
- 희소하거나 노이즈가 있는 깊이 맵을 더 잘 처리
- 깊이 추정 벤치마크(KITTI, NYU-Depth)에서 표준

### 방법 2: Least Squares Scaling

**목적 함수:**
```
s* = argmin_s Σ (s·d_pred - d_gt)²
```

**유도:**
s에 대해 미분하고 0으로 설정:
```
∂/∂s Σ (s·d_pred - d_gt)² = 0

2 Σ d_pred(s·d_pred - d_gt) = 0

s Σ d_pred² = Σ d_pred·d_gt

s* = (Σ d_pred·d_gt) / (Σ d_pred²)
```

행렬 형태:
```
s* = (d_pred^T · d_gt) / (d_pred^T · d_pred)
```

**장점:**
- ✅ 수학적으로 최적 (L2 오차 최소화)
- ✅ 밀집 예측에 적합

**단점:**
- ⚠️ 이상치에 더 민감
- ⚠️ 계산 비용 증가

### 방법 3: Mean Scaling

```
s* = mean(d_gt) / mean(d_pred) = (Σ d_gt) / (Σ d_pred)
```

**장점:**
- ✅ 구현 간단
- ✅ 빠른 계산

**단점:**
- ⚠️ 이상치에 매우 민감
- ⚠️ 폐색이 있는 깊이 맵에는 비추천

---

## 📐 Scale-Invariant Loss (Eigen et al., 2014)

로그 공간에서 작동하는 더 고급 공식:

### 공식

```
L_si = (1/N) Σ (log(d_pred) - log(d_gt) + α)²

where α = (1/N) Σ (log(d_gt) - log(d_pred))
```

**전개 형태:**
```
L_si = (1/N) Σ (log(d_pred) - log(d_gt))² 
     - (1/N²)(Σ (log(d_pred) - log(d_gt)))²
```

### 해석

- **첫 번째 항:** 픽셀별 로그 차이의 제곱 (픽셀 단위 오차)
- **두 번째 항:** 전역 scale 차이 제거 (scale-invariant 항)

**로그 공간에서:**
```
δᵢ = log(d_pred_i) - log(d_gt_i)

δ_mean = (1/N) Σ δᵢ

L_si = (1/N) Σ δᵢ² - δ_mean²
     = Var(log(d_pred/d_gt))
```

**의미:** 로그 깊이 비율의 분산을 최소화하여 손실을 진정으로 scale-invariant하게 만듭니다.

---

## 🔧 PyTorch 구현

### 기본 Scale-Adaptive Loss

```python
import torch
import torch.nn as nn

def scale_adaptive_loss(pred, gt, loss_type='l1', scaling='median'):
    """
    Scale-Adaptive Loss 구현
    
    Args:
        pred: [B, 1, H, W] 예측 깊이
        gt: [B, 1, H, W] ground truth 깊이
        loss_type: 'l1', 'l2', 'berhu', 'si' (scale-invariant)
        scaling: 'median', 'mean', 'least_squares'
    
    Returns:
        loss: 스칼라
        scale: 최적 scale factor
    """
    # 유효 마스크 (depth > 0)
    valid = (gt > 0) & (pred > 0)
    
    if scaling == 'median':
        # s* = median(d_gt) / median(d_pred)
        scale = torch.median(gt[valid]) / torch.median(pred[valid])
        
    elif scaling == 'mean':
        # s* = mean(d_gt) / mean(d_pred)
        scale = gt[valid].mean() / pred[valid].mean()
        
    elif scaling == 'least_squares':
        # s* = (pred^T · gt) / (pred^T · pred)
        pred_v = pred[valid]
        gt_v = gt[valid]
        scale = (pred_v * gt_v).sum() / (pred_v ** 2).sum()
    
    # Scale 보정
    pred_scaled = pred * scale
    
    # 손실 계산
    if loss_type == 'l1':
        loss = torch.abs(pred_scaled[valid] - gt[valid]).mean()
        
    elif loss_type == 'l2':
        loss = ((pred_scaled[valid] - gt[valid]) ** 2).mean()
        
    elif loss_type == 'berhu':
        diff = torch.abs(pred_scaled[valid] - gt[valid])
        c = 0.2 * diff.max()
        loss = torch.where(
            diff <= c,
            diff,  # L1
            (diff ** 2 + c ** 2) / (2 * c)  # L2
        ).mean()
        
    elif loss_type == 'si':
        # Scale-Invariant Loss
        log_diff = torch.log(pred_scaled[valid]) - torch.log(gt[valid])
        loss = (log_diff ** 2).mean() - (log_diff.mean() ** 2)
    
    return loss, scale
```

### Scale-Invariant Gradient Loss

```python
def scale_invariant_gradient_loss(pred, gt):
    """
    Scale-Invariant Gradient Matching Loss
    
    L = Σ |∇log(d_pred) - ∇log(d_gt)|
    """
    log_pred = torch.log(pred.clamp(min=1e-3))
    log_gt = torch.log(gt.clamp(min=1e-3))
    
    # x 방향 그래디언트
    grad_pred_x = log_pred[:, :, :, 1:] - log_pred[:, :, :, :-1]
    grad_gt_x = log_gt[:, :, :, 1:] - log_gt[:, :, :, :-1]
    
    # y 방향 그래디언트
    grad_pred_y = log_pred[:, :, 1:, :] - log_pred[:, :, :-1, :]
    grad_gt_y = log_gt[:, :, 1:, :] - log_gt[:, :, :-1, :]
    
    loss_x = torch.abs(grad_pred_x - grad_gt_x).mean()
    loss_y = torch.abs(grad_pred_y - grad_gt_y).mean()
    
    return loss_x + loss_y
```

---

## 🎓 G2-MonoDepth Loss: 고급 구현

G2-MonoDepth는 성능 향상을 위해 scale-adaptive와 gradient 손실을 결합합니다.

### 전체 구조

```
L_G2 = L_sa + λ · L_sg

where:
    L_sa: Scale-Adaptive Loss (상대 + 절대)
    L_sg: Scale-Invariant Gradient Loss (멀티스케일)
    λ: gradient 손실 가중치 (기본값: 0.5)
```

### 1. Scale-Adaptive Loss (L_sa)

```
L_sa = L_relative + L_absolute

L_relative = (1/M) Σ |d_norm - z_norm|

L_absolute = (1/M_V) Σ_V |d_i - z_i|
```

#### A. 상대 관계 항

**목적:** Scale-invariant한 상대적 깊이 관계 학습

**정규화:**
```
mean(x) = (1/M) Σ x_i

σ(x) = (1/M) Σ |x_i - mean(x)|  (평균 절대 편차, MAD)

x_norm = (x - mean(x)) / (σ(x) + ε)
```

**손실:**
```
d_norm = normalize(pred_depth)
z_norm = normalize(gt_depth)

L_relative = (1/M) Σ |d_norm - z_norm|
```

**특징:**
- ✅ Scale ambiguity 해결
- ✅ 전역 깊이 분포 정렬
- ✅ 모든 픽셀 사용

#### B. 절대 관계 항

**목적:** 유효한 GT가 있는 영역에서 절대 깊이 정확도 향상

```
L_absolute = (1/M_V) Σ_V |d_i - z_i| · mask_i

where:
    M_V: 유효 픽셀 수
    mask_i: 유효 픽셀 마스크 (0 또는 1)
```

**사용 사례:**
- 희소 LiDAR 깊이 완성
- 부분 ground truth 시나리오
- 유효 마스크가 없으면 이 항은 0

### 2. Scale-Invariant Gradient Loss (L_sg)

```
L_sg = (1/M) Σ_{k=1}^{K} Σ_i (|∇_h R_i^k| + |∇_w R_i^k|)

where:
    R_i^k = ρ_k(d_norm - z_norm)  (스케일 k에서의 정규화 잔차)
    ∇_h: 수평 그래디언트 (Sobel-x)
    ∇_w: 수직 그래디언트 (Sobel-y)
    K: 멀티스케일 레벨 수 (기본값: 4)
```

#### 과정

**1단계: 잔차 계산**
```
R = d_norm - z_norm
```

**2단계: 멀티스케일 다운샘플링**
```
R¹ = R                    (원본)
R² = downsample(R, 1/2)   (1/2 해상도)
R³ = downsample(R, 1/4)   (1/4 해상도)
R⁴ = downsample(R, 1/8)   (1/8 해상도)
```

**3단계: Sobel 그래디언트 계산**

Sobel-x (수평 에지):
```
       [-1  0  1]
K_x =  [-2  0  2]
       [-1  0  1]
```

Sobel-y (수직 에지):
```
       [-1 -2 -1]
K_y =  [ 0  0  0]
       [ 1  2  1]
```

그래디언트:
```
∇_h R^k = K_x * R^k
∇_w R^k = K_y * R^k
```

**4단계: 멀티스케일 손실 집계**
```
L_sg = Σ_{k=1}^{4} mean(|∇_h R^k| + |∇_w R^k|)
```

**이점:**
- 멀티스케일이 다양한 구조 크기를 포착
- 에지를 보존하면서 부드러움 유도
- Scale-invariant한 구조적 유사성

### 완전한 G2-MonoDepth 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class G2MonoDepthLoss(nn.Module):
    """
    G2-MonoDepth Loss Function
    L_G2 = L_sa + λ * L_sg
    """
    
    def __init__(self, lambda_sg=0.5, epsilon=1e-8, num_scales=4):
        super(G2MonoDepthLoss, self).__init__()
        self.lambda_sg = lambda_sg
        self.epsilon = epsilon
        self.num_scales = num_scales
        
        # Sobel 커널
        self.register_buffer('sobel_x', self._get_sobel_kernel('x'))
        self.register_buffer('sobel_y', self._get_sobel_kernel('y'))
    
    def _get_sobel_kernel(self, direction):
        if direction == 'x':
            kernel = torch.tensor([[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]])
        else:  # 'y'
            kernel = torch.tensor([[-1., -2., -1.],
                                   [ 0.,  0.,  0.],
                                   [ 1.,  2.,  1.]])
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def normalize_depth(self, depth):
        """MAD를 사용한 깊이 맵 정규화"""
        mean = torch.mean(depth, dim=[1, 2, 3], keepdim=True)
        std = torch.mean(torch.abs(depth - mean), dim=[1, 2, 3], keepdim=True)
        normalized = (depth - mean) / (std + self.epsilon)
        return normalized, mean, std
    
    def scale_adaptive_loss(self, pred_depth, gt_depth, valid_mask=None):
        """Scale-Adaptive Loss"""
        # 상대 항
        pred_norm, _, _ = self.normalize_depth(pred_depth)
        gt_norm, _, _ = self.normalize_depth(gt_depth)
        relative_loss = torch.mean(torch.abs(pred_norm - gt_norm))
        
        # 절대 항 (선택적)
        if valid_mask is not None:
            valid_pred = pred_depth * valid_mask
            valid_gt = gt_depth * valid_mask
            num_valid = torch.sum(valid_mask, dim=[1, 2, 3], keepdim=True)
            num_valid = torch.clamp(num_valid, min=1.0)
            
            absolute_error = torch.abs(valid_pred - valid_gt) * valid_mask
            absolute_loss = torch.sum(absolute_error) / (torch.sum(num_valid) + self.epsilon)
        else:
            absolute_loss = 0.0
        
        return relative_loss + absolute_loss
    
    def apply_sobel(self, x, kernel):
        """Sobel 연산자 적용"""
        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        gradient = F.conv2d(x_padded, kernel, padding=0)
        return gradient
    
    def scale_invariant_gradient_loss(self, pred_depth, gt_depth):
        """멀티스케일 그래디언트 손실"""
        pred_norm, _, _ = self.normalize_depth(pred_depth)
        gt_norm, _, _ = self.normalize_depth(gt_depth)
        residual = pred_norm - gt_norm
        
        total_gradient_loss = 0.0
        for k in range(1, self.num_scales + 1):
            if k > 1:
                scale_factor = 1.0 / (2 ** (k - 1))
                residual_k = F.interpolate(residual, scale_factor=scale_factor,
                                          mode='bilinear', align_corners=False)
            else:
                residual_k = residual
            
            grad_x = self.apply_sobel(residual_k, self.sobel_x)
            grad_y = self.apply_sobel(residual_k, self.sobel_y)
            gradient_loss = torch.mean(torch.abs(grad_x) + torch.abs(grad_y))
            total_gradient_loss += gradient_loss
        
        return total_gradient_loss
    
    def forward(self, pred_depth, gt_depth, valid_mask=None):
        """Forward pass"""
        loss_sa = self.scale_adaptive_loss(pred_depth, gt_depth, valid_mask)
        loss_sg = self.scale_invariant_gradient_loss(pred_depth, gt_depth)
        total_loss = loss_sa + self.lambda_sg * loss_sg
        
        loss_dict = {
            'total': total_loss.item(),
            'scale_adaptive': loss_sa.item() if isinstance(loss_sa, torch.Tensor) else loss_sa,
            'gradient': loss_sg.item(),
        }
        
        return total_loss, loss_dict
```

---

## 📊 방법 비교

| 방법 | 공식 | 장점 | 단점 |
|--------|---------|------|------|
| **Median** | `s* = median(gt)/median(pred)` | 강건함, 빠름 | 분포 무시 |
| **Mean** | `s* = mean(gt)/mean(pred)` | 간단함 | 이상치에 민감 |
| **Least Squares** | `s* = (pred·gt)/(pred·pred)` | 최적 | 계산량 증가 |
| **Scale-Invariant** | `L = Σδ² - (Σδ)²/N` | 진정한 불변성 | 더 복잡 |

---

## 📈 Scaling을 적용한 평가 메트릭

### 임계값 정확도

```
δ_t = % of pixels where max(d_pred/d_gt, d_gt/d_pred) < 1.25^t

for t ∈ {1, 2, 3}
```

**Scale 적용 후:**
```
d_pred_scaled = s* · d_pred

δ_t = % of pixels where max(d_pred_scaled/d_gt, d_gt/d_pred_scaled) < 1.25^t
```

### 오차 메트릭

```
AbsRel = (1/N) Σ |d_pred - d_gt| / d_gt

SqRel = (1/N) Σ (d_pred - d_gt)² / d_gt

RMSE = sqrt((1/N) Σ (d_pred - d_gt)²)

RMSE_log = sqrt((1/N) Σ (log(d_pred) - log(d_gt))²)
```

**Scaling 적용 시:**
모든 메트릭은 `d_pred → s* · d_pred` 후 계산

---

## 🎯 사용 사례

### 사례 1: 단안 깊이 추정
```python
# 유효 마스크 없음
loss, loss_dict = criterion(pred_depth, gt_depth, valid_mask=None)
```
- 모든 픽셀 사용
- Scale-invariant 학습
- 구조 보존 강조

### 사례 2: 희소 깊이 완성
```python
# 10% 유효 픽셀
valid_mask = (torch.rand(B, 1, H, W) > 0.9).float()
loss, loss_dict = criterion(pred_depth, gt_depth, valid_mask=valid_mask)
```
- 희소 GT 활용
- 유효 픽셀에서 절대 정확도
- 나머지 영역은 상대 관계

### 사례 3: 밀집 깊이 향상
```python
# 100% 유효 픽셀
valid_mask = torch.ones(B, 1, H, W)
loss, loss_dict = criterion(pred_depth, gt_depth, valid_mask=valid_mask)
```
- 밀집 지도
- 절대 깊이 정확도 최대화
- Scale과 절대값 모두 최적화

---

## 💡 하이퍼파라미터 튜닝

| 파라미터 | 기본값 | 의미 | 권장 범위 |
|-----------|---------|---------|-------------------|
| `lambda_sg` | 0.5 | 그래디언트 손실 가중치 | 0.1 ~ 1.0 |
| `epsilon` | 1e-8 | 0으로 나누기 방지 | 1e-10 ~ 1e-6 |
| `num_scales` | 4 | 멀티스케일 레벨 | 3 ~ 5 |

**튜닝 가이드:**
- `lambda_sg` ↑ → 에지 선명도 ↑, 평탄 영역 노이즈 ↑
- `lambda_sg` ↓ → 부드러운 예측, 에지 흐림
- `num_scales` ↑ → 다양한 구조 학습, 계산량 ↑

---

## 🔬 수학적 특성

### 왜 평균 절대 편차(MAD)?

```
σ_MAD(x) = (1/M) Σ |x_i - mean(x)|
```

**표준편차와 비교:**
```
σ_std(x) = sqrt((1/M) Σ (x_i - mean(x))²)
```

**장점:**
- ✅ 이상치에 더 강건 (제곱 없음)
- ✅ 계산이 안정적 (제곱근 불필요)
- ✅ 불규칙한 깊이 분포에 적합
- ✅ 희소 깊이 맵에 적합

### 정규화 효과

```
x_norm = (x - μ) / σ

→ E[x_norm] = 0
→ Var[x_norm] ≈ 1
```

**깊이에 적용:**
```
d_norm ~ N(0, 1)
z_norm ~ N(0, 1)

→ Scale-free 비교
```

### 로그 공간의 그래디언트 매칭

```
∇log(d) = ∇d / d

→ 상대적 변화율
```

**이점:**
- 깊이 불연속(에지) 보존
- 평탄한 영역의 부드러움 유지
- 멀티스케일로 다양한 구조 크기 커버

---

## 📚 관련 연구

### 주요 논문

1. **Eigen et al. (2014)** - "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
   - Scale-invariant loss 도입
   - 로그 공간 공식화 제안

2. **Godard et al. (2019)** - "Digging Into Self-Supervised Monocular Depth Estimation" (Monodepth2)
   - Scale-adaptive 학습
   - Median scaling 평가

3. **Guizilini et al. (2020)** - "3D Packing for Self-Supervised Monocular Depth Estimation" (PackNet-SFM)
   - Scale 일관성을 위한 3D packing
   - 평가에서 GT scale 옵션 제공

4. **Bhat et al. (2021)** - "AdaBins: Depth Estimation using Adaptive Bins"
   - Scale-invariant loss와 함께 적응형 binning

---

## ✅ 장점

1. **Scale-Invariant**: 절대 scale 없이 학습 가능
2. **통합 프레임워크**: 다양한 시나리오(희소/밀집)에 단일 손실
3. **구조 보존**: 그래디언트 매칭으로 에지 선명도 유지
4. **강건함**: MAD 정규화로 이상치 처리
5. **멀티스케일**: 다양한 스케일의 구조 포착

## ⚠️ 한계

1. **계산량**: 멀티스케일 + Sobel 연산으로 느림
2. **메모리**: 중간 결과로 메모리 사용량 증가
3. **하이퍼파라미터**: `lambda_sg` 튜닝 필요
4. **절대 항**: 유효 마스크 필요 (선택적 한계)

---

## 🚀 고급 확장

### 1. 적응형 Lambda (학습 중 동적 조정)

```python
class AdaptiveG2Loss(G2MonoDepthLoss):
    def forward(self, pred_depth, gt_depth, valid_mask=None, epoch=0):
        # 초기: 상대 관계 중심, 후기: 그래디언트 중심
        lambda_adaptive = self.lambda_sg * min(epoch / 50, 1.0)
        
        loss_sa = self.scale_adaptive_loss(pred_depth, gt_depth, valid_mask)
        loss_sg = self.scale_invariant_gradient_loss(pred_depth, gt_depth)
        
        return loss_sa + lambda_adaptive * loss_sg
```

### 2. 가중 멀티스케일 (특정 스케일에 우선순위)

```python
def scale_invariant_gradient_loss_weighted(self, pred_depth, gt_depth):
    weights = [1.0, 0.5, 0.25, 0.125]  # 원본 해상도에 더 큰 가중치
    
    total_loss = 0.0
    for k, w in enumerate(weights, start=1):
        # ... 그래디언트 계산 ...
        total_loss += w * gradient_loss
    
    return total_loss / sum(weights)
```

### 3. 에지 인식 가중치 (에지에서 더 큰 가중치)

```python
def edge_aware_gradient_loss(self, pred_depth, gt_depth):
    # GT 에지 강도 계산
    gt_grad_x = self.apply_sobel(gt_depth, self.sobel_x)
    gt_grad_y = self.apply_sobel(gt_depth, self.sobel_y)
    gt_edge_weight = torch.sqrt(gt_grad_x**2 + gt_grad_y**2)
    gt_edge_weight = gt_edge_weight / (gt_edge_weight.mean() + 1e-8)
    
    # ... 잔차 그래디언트 계산 ...
    
    # 에지 가중 손실
    weighted_loss = (torch.abs(grad_x) + torch.abs(grad_y)) * gt_edge_weight
    return weighted_loss.mean()
```

---

## 🎓 요약

Scale-Adaptive Loss는 **scale-invariance**와 **구조 보존**을 동시에 달성하는 우아한 솔루션입니다. 특히 다음에 효과적입니다:

- 자기지도 단안 깊이 추정
- 희소 깊이 완성 (LiDAR + 카메라 융합)
- 밀집 깊이 개선
- Scale ambiguity가 있는 모든 시나리오

G2-MonoDepth 공식은 멀티스케일 그래디언트 매칭으로 이를 확장하여 다양한 깊이 추정 작업에서 최고 수준의 성능을 제공합니다.

---

## 📖 참고문헌

1. Eigen, D., & Fergus, R. (2014). "Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture"
2. Godard, C., et al. (2019). "Digging Into Self-Supervised Monocular Depth Estimation"
3. Guizilini, V., et al. (2020). "3D Packing for Self-Supervised Monocular Depth Estimation"
4. Bhat, S. F., et al. (2021). "AdaBins: Depth Estimation using Adaptive Bins"

---

**문서 버전:** 1.0  
**최종 업데이트:** 2025년 10월 17일  
**작성자:** PackNet-SFM ResNet-SAN 팀
