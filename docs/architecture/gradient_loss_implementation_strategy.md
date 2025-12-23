# Gradient Loss 구현 전략

## 1. 개요

### 1.1 목표
현재 SSI-Silog Loss에 **Multi-Scale Gradient Loss**를 추가하여 depth map의 edge 보존 및 전체 구조적 일관성을 향상시킨다.

### 1.2 기대 효과
- 객체 경계(edge)에서의 depth 선명도 향상
- 전체 depth map의 구조적 일관성 개선
- "전체 맵이 안 좋다"는 문제 해결

### 1.3 현재 Loss 구조
```
현재: total_loss = ssi_weight × SSI_Loss + silog_weight × Silog_Loss
목표: total_loss = ssi_weight × SSI_Loss + silog_weight × Silog_Loss + gradient_weight × Gradient_Loss
```

---

## 2. 기존 코드 분석

### 2.1 현재 Loss 구조 (`ssi_silog_loss.py`)

```python
class SSISilogLoss(LossBase):
    def __init__(self, alpha=0.85, silog_ratio=10, silog_ratio2=0.85, 
                 ssi_weight=0.7, silog_weight=0.3,
                 min_depth=None, max_depth=None):
        # ...
        self.ssi_weight = ssi_weight
        self.silog_weight = silog_weight
    
    def forward(self, pred_inv_depth, gt_inv_depth, mask=None):
        # SSI Loss (inverse depth domain)
        ssi_loss = self.compute_ssi_loss_inv(pred_inv_depth, gt_inv_depth, mask)
        
        # Silog Loss (depth domain)
        pred_depth = inv2depth(pred_inv_depth)
        gt_depth = inv2depth(gt_inv_depth)
        silog_loss = self.compute_silog_loss(pred_depth, gt_depth, mask)
        
        # Combined loss
        total_loss = self.ssi_weight * ssi_loss + self.silog_weight * silog_loss
        return total_loss
```

### 2.2 G2-MonoDepth의 Gradient Loss 구현 (`losses.py`)

```python
class Gradient2D(Module):
    """Sobel gradient 계산"""
    def __init__(self):
        kernel_x = torch.FloatTensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        kernel_y = torch.FloatTensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.weight_x = Parameter(data=kernel_x.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.weight_y = Parameter(data=kernel_y.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        grad_x = conv2d(x, self.weight_x)
        grad_y = conv2d(x, self.weight_y)
        return grad_x, grad_y


class WeightedMSGradLoss(Module):
    """Multi-Scale Gradient Loss"""
    def __init__(self, k=4, sobel=True):
        self.grad_fun = Gradient2D().cuda()
        self.k = k  # number of scales

    def forward(self, output, target, hole_target):
        residual = hole_target * output + (1.0 - hole_target) * target - target
        loss = 0.
        for i in range(self.k):
            scale_factor = 1.0 / (2 ** i)
            k_residual = interpolate(residual, scale_factor=scale_factor) if i > 0 else residual
            loss += self.__gradient_loss__(k_residual)
        return loss / number_valid
```

### 2.3 YAML Config 구조

```yaml
model:
    loss:
        supervised_method: 'sparse-ssi-silog'
        ssi_weight: 0.5
        silog_weight: 0.5
        alpha: 0.85
        silog_ratio2: 0.85
        # 추가 예정
        gradient_weight: 0.0  # 또는 0.2
```

---

## 3. 구현 전략

### 3.1 구현 옵션 비교

| 옵션 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **A. SSISilogLoss 내부 추가** | 기존 클래스에 gradient 계산 추가 | 기존 구조 유지, 호환성 좋음 | 클래스가 복잡해짐 |
| **B. 별도 GradientLoss 클래스** | 독립적인 Loss 클래스 생성 | 모듈화, 재사용 가능 | 통합 로직 필요 |
| **C. SSISilogGradientLoss 새 클래스** | SSI+Silog+Gradient 통합 클래스 | 깔끔한 인터페이스 | 코드 중복 |

### 3.2 선택: **옵션 A - SSISilogLoss 확장**

**이유**:
1. 기존 YAML config와 호환성 유지
2. `supervised_loss.py`의 `get_loss_func()` 수정 최소화
3. gradient_weight=0이면 기존과 동일하게 동작 (하위 호환)

### 3.3 구현 계획

```
파일 수정 목록:
1. packnet_sfm/losses/ssi_silog_loss.py  ← Gradient Loss 추가
2. packnet_sfm/losses/supervised_loss.py  ← gradient_weight 파라미터 전달
3. configs/*.yaml  ← gradient_weight 설정 추가
```

---

## 4. 상세 구현 설계

### 4.1 Gradient2D 클래스 (새로 추가)

```python
class Gradient2D(nn.Module):
    """
    Sobel filter를 사용한 2D gradient 계산
    
    Sobel X:          Sobel Y:
    [-1, 0, 1]       [-1, -2, -1]
    [-2, 0, 2]       [ 0,  0,  0]
    [-1, 0, 1]       [ 1,  2,  1]
    """
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3)
        
        # Non-learnable parameters
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 1, H, W] depth map
        Returns:
            grad_x: [B, 1, H-2, W-2] horizontal gradient
            grad_y: [B, 1, H-2, W-2] vertical gradient
        """
        grad_x = F.conv2d(x, self.weight_x, padding=0)
        grad_y = F.conv2d(x, self.weight_y, padding=0)
        return grad_x, grad_y
```

### 4.2 SSISilogLoss 확장

```python
class SSISilogLoss(LossBase):
    def __init__(self, 
                 alpha=0.85, silog_ratio=10, silog_ratio2=0.85, 
                 ssi_weight=0.7, silog_weight=0.3,
                 gradient_weight=0.0,  # 🆕 추가
                 gradient_scales=4,     # 🆕 추가
                 min_depth=None, max_depth=None):
        super().__init__()
        # 기존 파라미터
        self.ssi_weight = ssi_weight
        self.silog_weight = silog_weight
        
        # 🆕 Gradient Loss 파라미터
        self.gradient_weight = gradient_weight
        self.gradient_scales = gradient_scales
        
        # 🆕 Gradient 계산기 초기화 (weight > 0일 때만)
        if gradient_weight > 0:
            self.gradient_fn = Gradient2D()
            print(f"   Gradient weight: {gradient_weight}")
            print(f"   Gradient scales: {gradient_scales}")
    
    def compute_gradient_loss(self, pred_depth, gt_depth, mask):
        """
        Multi-scale gradient loss 계산
        
        Args:
            pred_depth: [B, 1, H, W] 예측 depth
            gt_depth: [B, 1, H, W] GT depth
            mask: [B, 1, H, W] 유효 픽셀 마스크
        
        Returns:
            loss: scalar gradient loss
        """
        if self.gradient_weight <= 0:
            return torch.tensor(0.0, device=pred_depth.device)
        
        total_loss = 0.0
        
        for scale_idx in range(self.gradient_scales):
            scale_factor = 1.0 / (2 ** scale_idx)
            
            if scale_idx == 0:
                pred_s = pred_depth
                gt_s = gt_depth
                mask_s = mask
            else:
                pred_s = F.interpolate(pred_depth, scale_factor=scale_factor, 
                                       mode='bilinear', align_corners=False)
                gt_s = F.interpolate(gt_depth, scale_factor=scale_factor,
                                     mode='bilinear', align_corners=False)
                mask_s = F.interpolate(mask.float(), scale_factor=scale_factor,
                                       mode='nearest') > 0.5
            
            # Gradient 계산
            grad_pred_x, grad_pred_y = self.gradient_fn(pred_s)
            grad_gt_x, grad_gt_y = self.gradient_fn(gt_s)
            
            # Mask resize (gradient output is H-2, W-2)
            mask_grad = mask_s[:, :, 1:-1, 1:-1]
            
            # L1 loss on gradients
            if mask_grad.sum() > 0:
                loss_x = torch.abs(grad_pred_x - grad_gt_x)[mask_grad].mean()
                loss_y = torch.abs(grad_pred_y - grad_gt_y)[mask_grad].mean()
                total_loss += (loss_x + loss_y)
        
        return total_loss / self.gradient_scales
    
    def forward(self, pred_inv_depth, gt_inv_depth, mask=None):
        # 기존 SSI Loss
        ssi_loss = self.compute_ssi_loss_inv(pred_inv_depth, gt_inv_depth, mask)
        
        # 기존 Silog Loss
        pred_depth = inv2depth(pred_inv_depth)
        gt_depth = inv2depth(gt_inv_depth)
        
        if mask is None:
            mask = (gt_depth > 0)
        
        silog_loss = self.compute_silog_loss(pred_depth, gt_depth, mask)
        
        # 🆕 Gradient Loss
        gradient_loss = self.compute_gradient_loss(pred_depth, gt_depth, mask)
        
        # 결합
        total_loss = (self.ssi_weight * ssi_loss + 
                      self.silog_weight * silog_loss + 
                      self.gradient_weight * gradient_loss)
        
        # 메트릭 저장
        self.add_metric('ssi_component', ssi_loss)
        self.add_metric('silog_component', silog_loss)
        self.add_metric('gradient_component', gradient_loss)  # 🆕
        
        return total_loss
```

### 4.3 supervised_loss.py 수정

```python
def get_loss_func(supervised_method, **kwargs):
    # ...
    elif supervised_method.endswith('ssi-silog'):
        return SSISilogLoss(
            min_depth=kwargs.get('min_depth', None),
            max_depth=kwargs.get('max_depth', None),
            ssi_weight=kwargs.get('ssi_weight', 0.7),
            silog_weight=kwargs.get('silog_weight', 0.3),
            alpha=kwargs.get('alpha', 0.85),
            silog_ratio=kwargs.get('silog_ratio', 10),
            silog_ratio2=kwargs.get('silog_ratio2', 0.85),
            gradient_weight=kwargs.get('gradient_weight', 0.0),  # 🆕
            gradient_scales=kwargs.get('gradient_scales', 4),     # 🆕
        )
```

### 4.4 YAML Config 업데이트

```yaml
model:
    loss:
        supervised_method: 'sparse-ssi-silog'
        ssi_weight: 0.4           # 조정
        silog_weight: 0.4         # 조정
        gradient_weight: 0.2      # 🆕 추가
        gradient_scales: 4        # 🆕 추가 (1, 2, 4, 8배 downsample)
        alpha: 0.85
        silog_ratio2: 0.85
```

---

## 5. Weight 설정 가이드

### 5.1 권장 초기 설정

| 설정명 | ssi | silog | gradient | 설명 |
|--------|-----|-------|----------|------|
| **Conservative** | 0.45 | 0.45 | 0.1 | Gradient 영향 최소화 |
| **Balanced** | 0.4 | 0.4 | 0.2 | 균형 잡힌 설정 (권장) |
| **Edge-Focused** | 0.35 | 0.35 | 0.3 | Edge 보존 강조 |

### 5.2 실험 계획

```
Phase 1: gradient_weight=0.1 (보수적)
  - 기존 성능 유지 확인
  - Edge 영역 개선 여부 확인

Phase 2: gradient_weight=0.2 (균형)
  - 전체 성능 비교
  - RMSE, MAE, δ1 변화 분석

Phase 3: gradient_weight=0.3 (edge 강조)
  - Edge 선명도 vs 전체 정확도 trade-off 분석
```

---

## 6. 구현 순서

### 6.1 단계별 구현

```
Step 1: Gradient2D 클래스 구현
        - ssi_silog_loss.py에 추가
        - 단위 테스트

Step 2: SSISilogLoss 확장
        - gradient_weight, gradient_scales 파라미터 추가
        - compute_gradient_loss() 메서드 구현
        - forward() 수정

Step 3: supervised_loss.py 수정
        - get_loss_func()에 gradient 파라미터 전달

Step 4: YAML Config 업데이트
        - gradient_weight, gradient_scales 추가
        - 기본값 0.0 (하위 호환)

Step 5: 테스트
        - Dry-run 학습 테스트
        - Loss 값 로깅 확인
        - 메트릭 출력 확인
```

### 6.2 파일 수정 순서

```
1. packnet_sfm/losses/ssi_silog_loss.py
   - Gradient2D 클래스 추가
   - SSISilogLoss.__init__() 수정
   - compute_gradient_loss() 추가
   - forward() 수정

2. packnet_sfm/losses/supervised_loss.py
   - get_loss_func() 수정 (파라미터 전달)

3. configs/train_resnet_san_ncdb_distance_dual_head_640x384.yaml
   - gradient_weight, gradient_scales 추가
```

---

## 7. 테스트 계획

### 7.1 단위 테스트

```python
# Gradient2D 테스트
def test_gradient_2d():
    grad_fn = Gradient2D()
    x = torch.randn(2, 1, 64, 64)
    grad_x, grad_y = grad_fn(x)
    assert grad_x.shape == (2, 1, 62, 62)  # H-2, W-2
    assert grad_y.shape == (2, 1, 62, 62)
```

### 7.2 통합 테스트

```bash
# Dry-run 학습 (1 epoch)
python scripts/train.py configs/train_resnet_san_ncdb_distance_dual_head_640x384.yaml \
    --max_epochs 1 \
    --checkpoint.filepath checkpoints/test_gradient_loss/
```

### 7.3 검증 항목

- [ ] Loss 값이 NaN이 아님
- [ ] gradient_component 메트릭이 정상 출력
- [ ] gradient_weight=0일 때 기존과 동일한 결과
- [ ] Multi-scale gradient가 정상 동작

---

## 8. 예상 이슈 및 해결책

### 8.1 메모리 이슈

**문제**: Multi-scale gradient 계산으로 메모리 증가
**해결**: 
- gradient_scales=4 기본값 (1, 1/2, 1/4, 1/8)
- 필요시 gradient_scales=2로 감소

### 8.2 수치 안정성

**문제**: Gradient 계산 시 edge에서 큰 값
**해결**:
- L1 loss 사용 (outlier에 강건)
- Mask 적용으로 invalid 영역 제외

### 8.3 학습 불안정

**문제**: Gradient loss가 너무 커서 학습 불안정
**해결**:
- gradient_weight 작게 시작 (0.1)
- Gradient clipping 사용 (clip_grad: 1.0)

---

## 9. 성능 모니터링

### 9.1 로깅할 메트릭

```
loss/total             : 전체 loss
loss/ssi_component     : SSI loss 기여분
loss/silog_component   : Silog loss 기여분
loss/gradient_component: Gradient loss 기여분 (🆕)
```

### 9.2 TensorBoard 시각화

```
- Loss curves (total, ssi, silog, gradient)
- Depth map 시각화 (edge 영역 비교)
- Gradient map 시각화 (optional)
```

---

## 10. 요약

### 10.1 구현 요약

| 항목 | 내용 |
|------|------|
| **접근법** | SSISilogLoss 클래스 확장 |
| **새 파라미터** | gradient_weight, gradient_scales |
| **기본값** | gradient_weight=0.0 (하위 호환) |
| **권장값** | gradient_weight=0.2, gradient_scales=4 |

### 10.2 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `ssi_silog_loss.py` | Gradient2D 추가, compute_gradient_loss 추가 |
| `supervised_loss.py` | get_loss_func 파라미터 전달 |
| `*.yaml` | gradient_weight, gradient_scales 추가 |

### 10.3 다음 단계

1. **구현**: 위 설계대로 코드 수정
2. **테스트**: Dry-run 및 단위 테스트
3. **실험**: Baseline vs +Gradient 비교
4. **튜닝**: 최적 weight 조합 탐색

---

## 11. PM 관점 코드 리뷰 (2024-12-18)

### 11.1 구현 가능성 평가: ✅ **한 번에 구현 가능**

전체 코드베이스를 검토한 결과, 제안된 구현 전략은 **정확하고 실현 가능**합니다.

### 11.2 코드 흐름 검증

```
YAML Config
    ↓
configs/default_config.py (기본값 정의)
    ↓
model_wrapper.py:setup_model()
    → model_args = {**config.loss}  # loss config를 dict로 변환
    → model = SemiSupCompletionModel(**model_args)
    ↓
SemiSupCompletionModel.__init__(**kwargs)
    → SupervisedLoss(min_depth=..., max_depth=..., **kwargs)
    ↓
SupervisedLoss.__init__(**kwargs)
    → get_loss_func(supervised_method, **kwargs)
    ↓
get_loss_func() 내부
    → SSISilogLoss(
        ssi_weight=kwargs.get('ssi_weight', 0.7),
        silog_weight=kwargs.get('silog_weight', 0.3),
        gradient_weight=kwargs.get('gradient_weight', 0.0),  # 🆕 추가할 부분
        ...
      )
```

**결론**: YAML → Model → Loss까지 `**kwargs` 체인이 정확히 연결되어 있어, YAML에 `gradient_weight` 추가만으로 자동 전달됩니다.

### 11.3 수정 파일 최종 확정

| 파일 | 수정 내용 | 난이도 | 리스크 |
|------|----------|--------|--------|
| `ssi_silog_loss.py` | Gradient2D 클래스 추가, compute_gradient_loss 메서드 추가, forward() 수정 | 중 | 낮음 |
| `supervised_loss.py` | get_loss_func()에 gradient_weight, gradient_scales 파라미터 추가 | 하 | 매우 낮음 |
| `default_config.py` | gradient_weight, gradient_scales 기본값 추가 | 하 | 매우 낮음 |
| `train_*.yaml` | gradient_weight, gradient_scales 설정 추가 | 하 | 없음 |

### 11.4 발견된 고려사항

#### ✅ 확인됨: import 추가 필요
```python
# ssi_silog_loss.py 상단에 추가 필요
import torch.nn.functional as F  # F.conv2d, F.interpolate 사용
from typing import Tuple  # Gradient2D 반환 타입
```

#### ✅ 확인됨: register_buffer 사용
Gradient2D 클래스에서 `register_buffer` 사용 시 자동으로 device 이동이 처리됨.
`.cuda()` 호출 불필요 (G2-MonoDepth와 다른 점)

#### ✅ 확인됨: 하위 호환성
- `gradient_weight=0.0` 기본값으로 기존 동작과 100% 동일
- 기존 YAML config 수정 없이도 동작

#### ⚠️ 주의: Multi-scale 시 mask 크기
```python
# gradient output은 H-2, W-2이므로 mask 조정 필요
mask_grad = mask_s[:, :, 1:-1, 1:-1]  # 전략 문서에 이미 포함됨 ✓
```

### 11.5 테스트 전략

#### Phase 1: 단위 테스트 (구현 직후)
```python
# 터미널에서 직접 실행
python -c "
import torch
from packnet_sfm.losses.ssi_silog_loss import Gradient2D, SSISilogLoss

# Gradient2D 테스트
grad_fn = Gradient2D()
x = torch.randn(2, 1, 64, 64)
gx, gy = grad_fn(x)
print(f'Gradient2D output: gx={gx.shape}, gy={gy.shape}')
assert gx.shape == (2, 1, 62, 62), 'Gradient2D shape mismatch'

# SSISilogLoss with gradient 테스트
loss_fn = SSISilogLoss(gradient_weight=0.2, gradient_scales=4)
pred = torch.rand(2, 1, 64, 64) * 0.1 + 0.01
gt = torch.rand(2, 1, 64, 64) * 0.1 + 0.01
mask = torch.ones(2, 1, 64, 64, dtype=torch.bool)
loss = loss_fn(pred, gt, mask)
print(f'SSISilogLoss with gradient: {loss.item():.6f}')
print('✅ All tests passed!')
"
```

#### Phase 2: 통합 테스트 (1 epoch)
```bash
python scripts/core/train.py \
    configs/train_resnet_san_ncdb_distance_dual_head_640x384.yaml \
    --arch.max_epochs 1
```

### 11.6 예상 소요 시간

| 단계 | 예상 시간 |
|------|----------|
| ssi_silog_loss.py 수정 | 15분 |
| supervised_loss.py 수정 | 5분 |
| default_config.py 수정 | 3분 |
| YAML config 수정 | 2분 |
| 단위 테스트 | 5분 |
| 통합 테스트 | 10분 |
| **총계** | **~40분** |

### 11.7 최종 결론

```
✅ 구현 가능성: 100% 확인
✅ 하위 호환성: 보장됨 (기본값 0.0)
✅ 코드 흐름: 검증됨 (kwargs 체인 정상)
✅ 리스크: 낮음
✅ 예상 시간: 40분 이내

👉 구현 진행 권장
```
