# G2-MonoDepth Loss 함수 상세

## 1. Loss 구성 개요

G2-MonoDepth는 **3개의 Loss를 조합**하여 사용합니다:

```python
total_loss = loss_adepth + loss_rdepth + 0.5 * loss_rgrad
```

| Loss | 역할 | Weight |
|------|------|--------|
| `loss_adepth` | Absolute Depth Loss | 1.0 |
| `loss_rdepth` | Relative Depth Loss | 1.0 |
| `loss_rgrad` | Gradient Loss | 0.5 |

---

## 2. Absolute Depth Loss (loss_adepth)

### 2.1 정의

원본 depth 공간에서의 L1 Loss:

$$\mathcal{L}_{adepth} = \frac{1}{N} \sum_{i \in M} |d_i - \hat{d}_i|$$

여기서:
- $d_i$: 예측 depth
- $\hat{d}_i$: GT depth
- $M$: 유효한 GT가 있는 픽셀 마스크
- $N$: 유효 픽셀 수

### 2.2 구현 (WeightedDataLoss)

```python
class WeightedDataLoss(nn.Module):
    """
    Weighted L1 Loss for depth estimation
    """
    def __init__(self):
        super(WeightedDataLoss, self).__init__()
        self.loss_fn = nn.L1Loss(reduction="none")

    def forward(self, depth, gt, mask):
        """
        Args:
            depth: 예측 depth [B, 1, H, W]
            gt: GT depth [B, 1, H, W]
            mask: 유효 마스크 [B, 1, H, W]
        
        Returns:
            loss: weighted mean L1 loss
        """
        loss = self.loss_fn(depth, gt)  # [B, 1, H, W]
        
        # 유효 픽셀 수 계산
        mask_sum = torch.sum(mask, dim=(1, 2, 3))  # [B]
        mask_sum[mask_sum == 0] = 1e-6  # 0으로 나누기 방지
        
        # Masked mean
        loss = torch.sum(loss * mask, dim=(1, 2, 3)) / mask_sum  # [B]
        
        return loss.mean()  # scalar
```

### 2.3 특징

- **L1 Loss 사용**: L2보다 outlier에 강건
- **Mask 처리**: GT가 없는 영역 (sky, invalid) 제외
- **Batch-wise Mean**: 각 이미지별 평균 후 batch 평균

---

## 3. Relative Depth Loss (loss_rdepth)

### 3.1 동기

절대 depth 값은 스케일에 민감합니다. 상대적인 depth 분포를 학습하면:
- 스케일 변화에 강건
- 전체적인 depth 순서(ordering) 학습
- 우리 SSI Loss와 유사한 개념

### 3.2 Robust Standardization

일반 standardization 대신 **Robust Standardization** 사용:

$$z = \frac{d - \mu}{\sigma_{MAD} + \epsilon}$$

여기서:
- $\mu = \frac{1}{N}\sum_{i \in M} d_i$ (masked mean)
- $\sigma_{MAD} = \frac{1}{N}\sum_{i \in M} |d_i - \mu|$ (Mean Absolute Deviation)

### 3.3 구현 (StandardizeData)

```python
class StandardizeData(nn.Module):
    def __init__(self):
        super(StandardizeData, self).__init__()
    
    @staticmethod
    def __masked_mean_robust_standardization__(depth, mask, eps=1e-6):
        """
        Masked Mean + MAD standardization
        """
        # 유효 픽셀 수
        mask_num = torch.sum(mask, dim=(1, 2, 3))  # [B]
        mask_num[mask_num == 0] = eps
        
        # Masked Mean
        depth_mean = torch.sum(depth * mask, dim=(1, 2, 3)) / mask_num
        depth_mean = depth_mean.view(depth.shape[0], 1, 1, 1)  # [B, 1, 1, 1]
        
        # Mean Absolute Deviation (MAD)
        depth_std = torch.sum(
            torch.abs((depth - depth_mean) * mask), 
            dim=(1, 2, 3)
        ) / mask_num
        depth_std = depth_std.view(depth.shape[0], 1, 1, 1) + eps  # [B, 1, 1, 1]
        
        return depth_mean, depth_std
    
    def forward(self, depth, gt, mask):
        """
        Standardize both depth and gt to relative domain
        """
        mean_d, std_d = self.__masked_mean_robust_standardization__(depth, mask)
        mean_g, std_g = self.__masked_mean_robust_standardization__(gt, mask)
        
        sta_depth = (depth - mean_d) / std_d
        sta_gt = (gt - mean_g) / std_g
        
        return sta_depth, sta_gt
```

### 3.4 Relative Loss 계산

```python
# In training loop
standardize_fn = StandardizeData()

# 1. Standardize to relative domain
sta_depth, sta_gt = standardize_fn(pred_depth, gt_depth, mask)

# 2. L1 loss in relative domain
loss_rdepth = weighted_data_loss(sta_depth, sta_gt, mask)
```

### 3.5 왜 Robust Standardization인가?

| 방법 | Standard Deviation | MAD |
|------|-------------------|-----|
| 수식 | $\sqrt{\frac{1}{N}\sum(x-\mu)^2}$ | $\frac{1}{N}\sum|x-\mu|$ |
| Outlier 영향 | 큼 (제곱) | 작음 (절댓값) |
| 계산 안정성 | sqrt 필요 | 불필요 |

**Depth 데이터 특성**:
- 멀리 있는 객체 → 매우 큰 depth 값
- 이런 outlier가 std를 크게 왜곡할 수 있음
- MAD는 이런 outlier에 더 강건

---

## 4. Gradient Loss (loss_rgrad)

### 4.1 동기

Depth map의 **edge와 구조**를 보존하기 위한 loss입니다.

- Depth 경계에서의 선명도 유지
- 전체적인 depth 변화 패턴 학습
- 우리 프로젝트의 "전체 맵 일관성"과 직접적으로 관련!

### 4.2 Multi-Scale Gradient

단일 스케일 gradient는 특정 크기의 구조만 포착합니다. 
Multi-scale approach로 다양한 크기의 구조를 포착:

| Scale | Kernel Size | 포착하는 구조 |
|-------|-------------|--------------|
| 1 | 3×3 | 작은 edge, texture |
| 2 | 5×5 (effective) | 중간 크기 구조 |
| 4 | 9×9 (effective) | 큰 구조, 객체 경계 |
| 8 | 17×17 (effective) | 전체적인 depth 변화 |

### 4.3 Sobel Filter

이미지 gradient 계산에 **Sobel filter** 사용:

**X 방향 (가로 edge):**
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

**Y 방향 (세로 edge):**
$$G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

### 4.4 구현 (WeightedMSGradLoss)

```python
class WeightedMSGradLoss(nn.Module):
    """
    Multi-Scale Gradient Loss using Sobel filters
    """
    def __init__(self, scales=[1, 2, 4, 8]):
        super(WeightedMSGradLoss, self).__init__()
        self.scales = scales
        self.eps = 1e-6
        
        # Sobel filters (not learnable)
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def get_gradient(self, img):
        """
        Compute gradient using Sobel filters
        """
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        
        return grad_x, grad_y
    
    def forward(self, depth, gt, mask):
        """
        Multi-scale gradient loss
        
        Args:
            depth: predicted depth [B, 1, H, W]
            gt: ground truth depth [B, 1, H, W]
            mask: valid mask [B, 1, H, W]
        """
        total_loss = 0.0
        
        for scale in self.scales:
            # Downsample for multi-scale
            if scale > 1:
                depth_s = F.avg_pool2d(depth, scale, scale)
                gt_s = F.avg_pool2d(gt, scale, scale)
                mask_s = F.avg_pool2d(mask.float(), scale, scale)
                mask_s = (mask_s > 0.5).float()  # Binary mask
            else:
                depth_s, gt_s, mask_s = depth, gt, mask
            
            # Compute gradients
            grad_x_d, grad_y_d = self.get_gradient(depth_s)
            grad_x_g, grad_y_g = self.get_gradient(gt_s)
            
            # L1 loss on gradients
            loss_x = torch.abs(grad_x_d - grad_x_g) * mask_s
            loss_y = torch.abs(grad_y_d - grad_y_g) * mask_s
            
            # Masked mean
            mask_sum = torch.sum(mask_s) + self.eps
            scale_loss = (torch.sum(loss_x) + torch.sum(loss_y)) / mask_sum
            
            total_loss += scale_loss
        
        return total_loss / len(self.scales)
```

### 4.5 Gradient Loss 효과

```
Original Depth     Predicted (w/o grad loss)     Predicted (w/ grad loss)
┌──────────┐       ┌──────────┐                  ┌──────────┐
│██████    │       │██████▓▓  │                  │██████    │
│██████    │  →    │██████▓▓  │   vs             │██████    │
│      ░░░░│       │  ▓▓▓░░░░░│                  │      ░░░░│
│      ░░░░│       │  ▓▓▓░░░░░│                  │      ░░░░│
└──────────┘       └──────────┘                  └──────────┘
                   (blurry edges)                (sharp edges)
```

---

## 5. 전체 Loss 계산 흐름

### 5.1 코드 흐름

```python
class Trainer:
    def __init__(self):
        self.weighted_data_loss = WeightedDataLoss()
        self.weighted_grad_loss = WeightedMSGradLoss(scales=[1, 2, 4, 8])
        self.standardize = StandardizeData()
    
    def compute_loss(self, pred_depth, gt_depth, mask):
        """
        Total loss computation
        """
        # 1. Absolute Depth Loss
        loss_adepth = self.weighted_data_loss(pred_depth, gt_depth, mask)
        
        # 2. Relative Depth Loss
        sta_depth, sta_gt = self.standardize(pred_depth, gt_depth, mask)
        loss_rdepth = self.weighted_data_loss(sta_depth, sta_gt, mask)
        
        # 3. Gradient Loss (on relative domain for scale invariance)
        loss_rgrad = self.weighted_grad_loss(sta_depth, sta_gt, mask)
        
        # 4. Total Loss
        total_loss = loss_adepth + loss_rdepth + 0.5 * loss_rgrad
        
        return total_loss, {
            'loss_adepth': loss_adepth.item(),
            'loss_rdepth': loss_rdepth.item(),
            'loss_rgrad': loss_rgrad.item(),
        }
```

### 5.2 각 Loss의 역할

```
┌─────────────────────────────────────────────────────────────────┐
│                        Total Loss                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  loss_adepth (1.0)     loss_rdepth (1.0)     loss_rgrad (0.5)   │
│       │                     │                     │              │
│       ▼                     ▼                     ▼              │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐        │
│  │ 절대    │           │ 상대    │           │ 구조    │        │
│  │ 정확도  │           │ 분포    │           │ 보존    │        │
│  └─────────┘           └─────────┘           └─────────┘        │
│       │                     │                     │              │
│       ▼                     ▼                     ▼              │
│  metric 정확도         scale 일관성          edge 선명도         │
│  (RMSE, MAE)          (SI-RMSE)              (경계 보존)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 우리 SSI-Silog Loss와의 비교

### 6.1 개념 비교

| G2-MonoDepth | PackNet-SfM (우리) |
|--------------|-------------------|
| loss_adepth (L1) | Silog Loss의 일부 |
| loss_rdepth (standardized L1) | SSI Loss |
| loss_rgrad (Sobel gradient) | ❌ 없음 |

### 6.2 SSI vs Relative Depth Loss

**SSI Loss (Scale-Shift Invariant):**
$$d^* = \arg\min_{s,t} \|sd + t - \hat{d}\|^2$$

**Relative Loss (Standardization):**
$$z_d = \frac{d - \mu_d}{\sigma_d}, \quad z_{\hat{d}} = \frac{\hat{d} - \mu_{\hat{d}}}{\sigma_{\hat{d}}}$$

| 측면 | SSI | Relative (Standardization) |
|------|-----|---------------------------|
| Scale 보정 | 최적 scale 찾기 | std로 나누기 |
| Shift 보정 | 최적 shift 찾기 | mean 빼기 |
| 계산 복잡도 | 높음 (closed-form) | 낮음 (단순 연산) |
| 이론적 근거 | affine invariance | 통계적 normalization |

### 6.3 Gradient Loss 적용 가능성

우리 프로젝트에 **Gradient Loss 추가**를 고려할 수 있습니다:

```python
# 제안: 우리 loss에 gradient term 추가
total_loss = ssi_weight * ssi_loss + silog_weight * silog_loss + grad_weight * grad_loss
```

**기대 효과:**
- Edge 보존 향상
- 전체 맵 일관성 개선
- depth 경계에서의 선명도 증가

---

## 7. 핵심 인사이트

### 7.1 Loss 설계 원칙

1. **다중 도메인 학습**: 절대값 + 상대값 동시 최적화
2. **구조 보존**: Gradient loss로 edge 선명도 유지
3. **Scale Invariance**: Standardization으로 스케일 변화에 강건

### 7.2 우리 프로젝트에 적용 시

| 항목 | 현재 | 제안 |
|------|------|------|
| Loss | SSI + Silog | SSI + Silog + Gradient |
| Gradient weight | - | 0.3~0.5 |
| Multi-scale | - | [1, 2, 4, 8] |

### 7.3 구현 난이도

- **Gradient Loss 추가**: ⭐⭐ (중간)
  - Sobel filter 구현 필요
  - 기존 loss 구조에 통합

- **Relative Loss 변경**: ⭐ (쉬움)
  - SSI가 이미 유사한 역할
  - 대체보다는 보완으로 사용

---

## 다음 문서

- [04_data_processing.md](04_data_processing.md) - 데이터 처리 상세
