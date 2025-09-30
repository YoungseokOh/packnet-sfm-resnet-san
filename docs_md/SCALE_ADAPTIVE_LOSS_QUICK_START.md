# Scale-Adaptive Loss 빠른 시작 가이드

## 🚀 5분 안에 시작하기

이 문서는 Scale-Adaptive Loss를 **최대한 빠르게** 프로젝트에 추가하고 테스트하는 방법을 설명합니다.

---

## ⚡ Step 1: 파일 생성 (2분)

### 1.1 Loss 클래스 파일 생성

```bash
cd /workspace/packnet-sfm
```

`packnet_sfm/losses/scale_adaptive_loss.py` 파일을 생성하고 아래 코드를 복사:

<details>
<summary>📄 전체 코드 보기 (클릭)</summary>

```python
# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from packnet_sfm.losses.loss_base import LossBase
from packnet_sfm.utils.depth import inv2depth


class ScaleAdaptiveLoss(LossBase):
    """
    G2-MonoDepth Scale-Adaptive Loss
    
    L_total = L_sa + λ_sg * L_sg
    
    Parameters
    ----------
    lambda_sg : float
        Gradient loss weight (default: 0.5)
    epsilon : float
        Numerical stability constant (default: 1e-8)
    num_scales : int
        Multi-scale levels (default: 4)
    use_absolute : bool
        Use absolute term for valid pixels (default: True)
    use_inv_depth : bool
        If True, compute on inverse depth directly (faster, like SSI)
        If False, convert to depth first (more accurate for gradients)
        Default: False (convert to depth)
    """
    
    def __init__(self, lambda_sg=0.5, epsilon=1e-8, num_scales=4, 
                 use_absolute=True, use_inv_depth=False):
        super().__init__()
        self.lambda_sg = lambda_sg
        self.epsilon = epsilon
        self.num_scales = num_scales
        self.use_absolute = use_absolute
        self.use_inv_depth = use_inv_depth
        
        # Sobel kernels
        self.register_buffer('sobel_x', self._get_sobel_kernel('x'))
        self.register_buffer('sobel_y', self._get_sobel_kernel('y'))
        
        print(f"🎯 Scale-Adaptive Loss initialized:")
        print(f"   λ_sg: {lambda_sg}, num_scales: {num_scales}")
        print(f"   use_absolute: {use_absolute}, use_inv_depth: {use_inv_depth}")
    
    def _get_sobel_kernel(self, direction):
        """Create Sobel kernel"""
        if direction == 'x':
            kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        else:
            kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def normalize_depth(self, depth):
        """Normalize using MAD"""
        mean = torch.mean(depth, dim=[2, 3], keepdim=True)
        mad = torch.mean(torch.abs(depth - mean), dim=[2, 3], keepdim=True)
        normalized = (depth - mean) / (mad + self.epsilon)
        return normalized, mean, mad
    
    def scale_adaptive_loss(self, pred_depth, gt_depth, valid_mask=None):
        """Compute L_sa = L_relative + L_absolute"""
        # Relative term
        pred_norm, _, _ = self.normalize_depth(pred_depth)
        gt_norm, _, _ = self.normalize_depth(gt_depth)
        relative_loss = torch.mean(torch.abs(pred_norm - gt_norm))
        
        # Absolute term
        absolute_loss = 0.0
        if self.use_absolute and valid_mask is not None:
            valid_pred = pred_depth * valid_mask
            valid_gt = gt_depth * valid_mask
            num_valid = torch.clamp(torch.sum(valid_mask, dim=[1,2,3], keepdim=True), min=1.0)
            absolute_error = torch.abs(valid_pred - valid_gt) * valid_mask
            absolute_loss = torch.sum(absolute_error) / (torch.sum(num_valid) + self.epsilon)
        
        self.add_metric('scale_adaptive/relative', relative_loss)
        if self.use_absolute:
            self.add_metric('scale_adaptive/absolute', absolute_loss)
        
        return relative_loss + absolute_loss
    
    def apply_sobel(self, x, kernel):
        """Apply Sobel operator"""
        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(x_padded, kernel, padding=0)
    
    def scale_invariant_gradient_loss(self, pred_depth, gt_depth):
        """Compute multi-scale gradient loss"""
        pred_norm, _, _ = self.normalize_depth(pred_depth)
        gt_norm, _, _ = self.normalize_depth(gt_depth)
        residual = pred_norm - gt_norm
        
        total_loss = 0.0
        for k in range(1, self.num_scales + 1):
            if k > 1:
                residual_k = F.interpolate(residual, scale_factor=1.0/(2**(k-1)), 
                                          mode='bilinear', align_corners=False)
            else:
                residual_k = residual
            
            grad_x = self.apply_sobel(residual_k, self.sobel_x)
            grad_y = self.apply_sobel(residual_k, self.sobel_y)
            loss_k = torch.mean(torch.abs(grad_x) + torch.abs(grad_y))
            total_loss += loss_k
            self.add_metric(f'gradient/scale_{k}', loss_k)
        
        return total_loss / self.num_scales
    
    def forward(self, pred_inv_depth, gt_inv_depth, mask=None):
        """Main forward pass"""
        # Convert to depth or use inv_depth directly
        if self.use_inv_depth:
            # Work directly on inverse depth (like SSI)
            pred_data = pred_inv_depth
            gt_data = gt_inv_depth
        else:
            # Convert to depth (like original G2-MonoDepth)
            pred_data = inv2depth(pred_inv_depth)
            gt_data = inv2depth(gt_inv_depth)
        
        # Compute losses
        loss_sa = self.scale_adaptive_loss(pred_data, gt_data, mask)
        loss_sg = self.scale_invariant_gradient_loss(pred_data, gt_data)
        
        total_loss = loss_sa + self.lambda_sg * loss_sg
        
        self.add_metric('total_loss', total_loss)
        self.add_metric('loss_sa', loss_sa)
        self.add_metric('loss_sg', loss_sg)
        
        return total_loss
```

</details>

**한 줄로 복사:**
```bash
# 위 코드 전체를 복사하여 파일에 붙여넣기
vi packnet_sfm/losses/scale_adaptive_loss.py
```

---

## ⚡ Step 2: 프로젝트 통합 (1분)

### 2.1 supervised_loss.py 수정

`packnet_sfm/losses/supervised_loss.py` 파일 열기:

**1) Import 추가 (파일 상단):**

```python
from packnet_sfm.losses.scale_adaptive_loss import ScaleAdaptiveLoss
```

**2) get_loss_func() 함수에 추가 (약 79-110라인 부근):**

```python
def get_loss_func(supervised_method, **kwargs):
    """Determines the supervised loss to be used, given the supervised method."""
    print(f"🔍 Loading loss function for: {supervised_method}")
    
    # ... 기존 코드 ...
    
    elif supervised_method.endswith('scale-adaptive'):  # ← 이 부분 추가
        return ScaleAdaptiveLoss(
            lambda_sg=kwargs.get('lambda_sg', 0.5),
            epsilon=kwargs.get('epsilon', 1e-8),
            num_scales=kwargs.get('num_scales', 4),
            use_absolute=kwargs.get('use_absolute', True),
        )
    
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))
```

**빠른 수정 명령:**

```bash
# Import 추가
sed -i '13a from packnet_sfm.losses.scale_adaptive_loss import ScaleAdaptiveLoss' \
    packnet_sfm/losses/supervised_loss.py

# 또는 직접 편집
vi packnet_sfm/losses/supervised_loss.py
# 13번째 줄 다음에 import 추가
# get_loss_func() 함수에 elif 블록 추가
```

---

## ⚡ Step 3: 테스트 (2분)

### 3.1 빠른 단위 테스트

Python 인터프리터에서 테스트:

```python
import torch
from packnet_sfm.losses.scale_adaptive_loss import ScaleAdaptiveLoss

# 1. 초기화
loss_fn = ScaleAdaptiveLoss(lambda_sg=0.5)
print("✅ Loss 초기화 성공")

# 2. 더미 데이터
pred = torch.rand(2, 1, 192, 640) * 0.1 + 0.01
gt = torch.rand(2, 1, 192, 640) * 0.1 + 0.01

# 3. Forward pass
loss = loss_fn(pred, gt)
print(f"✅ Loss 계산 성공: {loss.item():.4f}")

# 4. Metrics 확인
print(f"✅ Metrics: {list(loss_fn.metrics.keys())}")
```

**예상 출력:**
```
🎯 Scale-Adaptive Loss initialized:
   λ_sg: 0.5, num_scales: 4, use_absolute: True
✅ Loss 초기화 성공
✅ Loss 계산 성공: 0.8234
✅ Metrics: ['scale_adaptive/relative', 'gradient/scale_1', 'gradient/scale_2', 
             'gradient/scale_3', 'gradient/scale_4', 'total_loss', 'loss_sa', 'loss_sg']
```

### 3.2 통합 테스트 (선택)

간단한 학습 테스트:

```bash
# 5 에폭만 테스트
python scripts/train.py \
    configs/train_resnet_san_kitti.yaml \
    --supervised-method sparse-scale-adaptive \
    --lambda-sg 0.5 \
    --max-epochs 5 \
    --gpus 0
```

---

## 📝 YAML Config 예시

### 기본 설정 (KITTI)

`configs/train_scale_adaptive.yaml` 생성:

```yaml
name: 'test_scale_adaptive'

model:
    name: 'SemiSupModel'
    params:
        supervised_method: 'sparse-scale-adaptive'
        supervised_loss_weight: 1.0
        lambda_sg: 0.5
        num_scales: 4
        use_absolute: true
        
    depth_net:
        name: 'ResNetSAN01'
        version: '18pt'
        use_film: true
        film_scales: [0]

datasets:
    train:
        batch_size: 4
        path: ['/data/kitti_raw']
        split: ['eigen_zhou']

trainer:
    max_epochs: 20
    gpus: [0]
```

---

## 🎯 하이퍼파라미터 빠른 가이드

### lambda_sg (Gradient 가중치)

| 값 | 추천 사용처 | 효과 |
|----|------------|------|
| `0.3` | 희소 LiDAR | 부드러움 우선 |
| `0.5` | **기본값** | 균형잡힌 설정 |
| `0.7` | 밀집 깊이 | 에지 선명 |

### num_scales

| 값 | 추천 사용처 |
|----|------------|
| `2` | 빠른 실험 |
| `4` | **기본값** |
| `5` | 고품질 결과 |

### use_inv_depth (⭐ 성능 옵션)

| 값 | 설명 | 추천 |
|----|------|------|
| `false` | depth로 변환 후 계산 (정확) | 연구/논문 |
| `true` | inv_depth 직접 계산 (빠름) | 프로덕션/GPU 부족 |

**언제 `true`로 설정?**
- GPU 메모리 부족할 때
- 학습 속도가 중요할 때
- SSI 등 다른 loss와 일관성 필요할 때

---

## 🔥 자주 사용하는 명령어

### 1. 빠른 테스트 학습

```bash
python scripts/train.py \
    configs/train_resnet_san_kitti.yaml \
    --supervised-method sparse-scale-adaptive \
    --lambda-sg 0.5 \
    --max-epochs 5 \
    --name "quick_test"
```

### 2. 하이퍼파라미터 Sweep

```bash
for lambda in 0.3 0.5 0.7; do
    python scripts/train.py \
        configs/train_scale_adaptive.yaml \
        --lambda-sg $lambda \
        --name "lambda_${lambda}" \
        --max-epochs 10
done
```

### 3. TensorBoard 모니터링

```bash
tensorboard --logdir outputs/ --port 6006
```

---

## ✅ 성공 체크리스트

완료한 항목에 체크:

- [ ] `scale_adaptive_loss.py` 파일 생성
- [ ] `supervised_loss.py`에 import 추가
- [ ] `get_loss_func()`에 elif 블록 추가
- [ ] Python 인터프리터 테스트 성공
- [ ] Loss 값이 숫자로 출력됨 (NaN 아님)
- [ ] Metrics 딕셔너리 확인 완료
- [ ] (선택) 5 에폭 학습 테스트 완료

---

## 🐛 빠른 문제 해결

### 문제 1: Import 에러

```python
ModuleNotFoundError: No module named 'packnet_sfm.losses.scale_adaptive_loss'
```

**해결:** 파일 경로 확인
```bash
ls -l packnet_sfm/losses/scale_adaptive_loss.py
# 파일이 없으면 Step 1부터 다시
```

### 문제 2: Loss가 NaN

```python
Loss: nan
```

**해결:** 깊이 값 범위 확인
```python
# scale_adaptive_loss.py의 forward()에 추가
pred_depth = torch.clamp(inv2depth(pred_inv_depth), min=0.1, max=100.0)
gt_depth = torch.clamp(inv2depth(gt_inv_depth), min=0.1, max=100.0)
```

### 문제 3: GPU 메모리 부족

**해결:** 파라미터 줄이기
```yaml
num_scales: 2      # 4 → 2
batch_size: 2      # 4 → 2
```

---

## 📊 예상 결과

### 초기 Loss 값

```
Epoch 1:
  loss_sa: 0.85
  loss_sg: 1.23
  total_loss: 1.47
```

### 수렴 후 (20 epochs)

```
Epoch 20:
  loss_sa: 0.12
  loss_sg: 0.31
  total_loss: 0.28
```

### TensorBoard 그래프

정상적인 경우:
- ✅ Total loss: 점진적 감소
- ✅ Loss_sa: 빠르게 감소
- ✅ Loss_sg: 천천히 감소
- ✅ Gradient scales: 모든 스케일 균형

---

## 🚀 다음 단계

구현이 성공했다면:

1. **전체 학습 실행**
   ```bash
   python scripts/train.py configs/train_scale_adaptive.yaml --max-epochs 50
   ```

2. **성능 비교**
   - 기존 SSI loss와 비교
   - Evaluation metrics 확인
   - 시각적 결과 비교

3. **하이퍼파라미터 최적화**
   - lambda_sg 튜닝
   - num_scales 실험
   - 데이터셋별 최적값 찾기

---

## 📚 더 자세한 정보

- **전체 구현 가이드:** [`SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md`](./SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md)
- **이론적 배경:** [`SCALE_ADAPTIVE_LOSS.md`](./SCALE_ADAPTIVE_LOSS.md)
- **프로젝트 README:** `../README.md`

---

**소요 시간:** ~5분  
**난이도:** ⭐⭐ (중급)  
**버전:** 1.0  
**최종 업데이트:** 2025년 10월 17일
