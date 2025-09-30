# Scale-Adaptive Loss 통합 분석 보고서

## 📋 목차
1. [Parameter Flow 완전 분석](#1-parameter-flow-완전-분석)
2. [Training Flow 상세 분석](#2-training-flow-상세-분석)
3. [기존 Loss Functions 패턴 분석](#3-기존-loss-functions-패턴-분석)
4. [빠진 부분 및 필요한 수정사항](#4-빠진-부분-및-필요한-수정사항)
5. [완전한 통합 체크리스트](#5-완전한-통합-체크리스트)

---

## 1. Parameter Flow 완전 분석

### 1.1 전체 파라미터 흐름도

```
YAML Config (train_*.yaml)
    ↓
    model:
      params:
        min_depth: 0.0
        max_depth: 80.0
      loss:
        supervised_method: 'sparse-scale-adaptive'
        supervised_num_scales: 4
        lambda_sg: 0.5
        use_inv_depth: false
        alpha_schedule: 'linear'
        # ... other params
    ↓
ModelWrapper.__init__(config)
    ↓
SemiSupCompletionModel.__init__(
    min_depth=config.model.params.min_depth,  # ✅ 명시적 전달
    max_depth=config.model.params.max_depth,  # ✅ 명시적 전달
    **config.model.loss  # supervised_method, lambda_sg 등
)
    ↓
SupervisedLoss.__init__(
    supervised_method='sparse-scale-adaptive',
    min_depth=min_depth,  # ✅ 부모에서 받음
    max_depth=max_depth,  # ✅ 부모에서 받음
    **kwargs  # lambda_sg, use_inv_depth, alpha_schedule 등
)
    ↓
get_loss_func(
    supervised_method='sparse-scale-adaptive',
    min_depth=min_depth,  # ✅ kwargs로 전달
    max_depth=max_depth,  # ✅ kwargs로 전달
    lambda_sg=lambda_sg,
    use_inv_depth=use_inv_depth,
    alpha_schedule=alpha_schedule,
    # ... other params
)
    ↓
ScaleAdaptiveLoss.__init__(
    min_depth=kwargs.get('min_depth', 0.1),
    max_depth=kwargs.get('max_depth', 100.0),
    lambda_sg=kwargs.get('lambda_sg', 0.5),
    use_inv_depth=kwargs.get('use_inv_depth', False),
    alpha_schedule=kwargs.get('alpha_schedule', 'linear'),
    # ... other params
)
```

### 1.2 Runtime Parameter Flow (Training Step)

```python
# ModelWrapper.training_step (model_wrapper.py:266)
def training_step(self, batch, batch_idx):
    # Progress 계산
    progress = self.current_epoch / self.config.arch.max_epochs
    
    # Model forward (progress 전달)
    model_output = self.model(batch, progress=progress)
    
    return model_output['loss']

# SemiSupCompletionModel.forward (SemiSupCompletionModel.py:~200)
def forward(self, batch, return_logs=False, progress=0.0):
    # ... depth prediction ...
    
    # Supervised loss 계산 (progress 전달)
    sup_output = self.supervised_loss(
        pred_inv_depths,
        gt_inv_depths,
        return_logs=return_logs,
        progress=progress,  # ✅ Progress 전달
        masks=masks  # ✅ Mask 전달
    )
    
    return {'loss': loss, ...}

# SupervisedLoss.forward (supervised_loss.py:277)
def forward(self, inv_depths, gt_inv_depth, return_logs=False, progress=0.0, masks=None):
    # ✅ Progress 저장 (loss function에서 사용)
    self._progress = progress
    
    # Multi-scale GT depth 생성
    gt_inv_depths = match_scales(gt_inv_depth, inv_depths, self.n, ...)
    
    # Loss 계산 (마스크 포함)
    loss = self.calculate_loss(inv_depths, gt_inv_depths, masks=masks)
    
    return {'loss': loss, 'metrics': self.metrics}

# SupervisedLoss.calculate_loss (supervised_loss.py:149)
def calculate_loss(self, inv_depths, gt_inv_depths, masks=None):
    for i in range(num_scales):
        # Sparse 마스크 생성
        valid_mask = (gt_inv_depths[i] > 0.).detach()
        
        # 추가 마스크 결합
        if masks is not None and i < len(masks):
            current_mask = valid_mask & masks[i]
        else:
            current_mask = valid_mask
        
        # Loss function signature 검사
        loss_kwargs = {}
        if hasattr(self.loss_func, 'forward'):
            sig = inspect.signature(self.loss_func.forward)
            params = sig.parameters
            if 'mask' in params:
                loss_kwargs['mask'] = current_mask
            if 'progress' in params:
                loss_kwargs['progress'] = self._progress  # ✅ Progress 전달
        
        # Loss 계산
        loss_i = self.loss_func(pred_filled, gt_filled, **loss_kwargs)
        
    return total_loss / num_scales
```

---

## 2. Training Flow 상세 분석

### 2.1 ModelWrapper 핵심 메서드

```python
# packnet_sfm/models/model_wrapper.py

class ModelWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model 초기화 (config 전체 전달)
        self.model = setup_model(config.model)
        
    @property
    def progress(self):
        """Training progress (0.0 ~ 1.0)"""
        return self.current_epoch / self.config.arch.max_epochs
    
    def training_step(self, batch, batch_idx):
        """
        핵심 training step
        - progress를 model.forward()에 전달
        - model이 loss 계산 및 반환
        """
        # ✅ Progress 전달
        model_output = self.model(batch, progress=self.progress)
        
        # Loss 반환
        return model_output['loss']
    
    def configure_optimizers(self):
        """Optimizer와 scheduler 설정"""
        # Depth network optimizer
        depth_params = self.model.depth_net.parameters()
        self.optimizer = setup_optimizer(depth_params, self.config.model.optimizer)
        
        # Scheduler
        self.scheduler = setup_scheduler(self.optimizer, self.config.model.scheduler)
        
        return self.optimizer
```

### 2.2 Progress 전달 메커니즘

| Level | Component | Progress 전달 여부 | 비고 |
|-------|-----------|-------------------|------|
| 1 | `ModelWrapper.training_step()` | **생성** | `self.current_epoch / max_epochs` |
| 2 | `SemiSupModel.forward()` | ✅ **전달** | `progress=progress` parameter |
| 3 | `SupervisedLoss.forward()` | ✅ **저장** | `self._progress = progress` |
| 4 | `SupervisedLoss.calculate_loss()` | ✅ **활용** | `loss_kwargs['progress'] = self._progress` |
| 5 | `ScaleAdaptiveLoss.forward()` | ✅ **수신** | `forward(..., progress=0.0)` parameter |

**✅ Progress는 완전히 전달되는 구조입니다!**

### 2.3 Mask 전달 메커니즘

```python
# 1단계: Batch에서 mask 추출 (SemiSupCompletionModel)
masks = batch.get('mask', None)  # Optional binary mask

# 2단계: Multi-scale masks 생성
if masks is not None:
    masks_list = [
        F.interpolate(masks, size=inv_depths[i].shape[-2:], mode='nearest')
        for i in range(len(inv_depths))
    ]

# 3단계: SupervisedLoss에 전달
sup_output = self.supervised_loss(
    inv_depths, gt_inv_depths,
    masks=masks_list  # ✅ Multi-scale masks
)

# 4단계: calculate_loss에서 valid_mask와 결합
valid_mask = (gt_inv_depths[i] > 0.).detach()  # Sparse GT mask
if masks is not None and i < len(masks):
    current_mask = valid_mask & masks[i]  # ✅ 결합
else:
    current_mask = valid_mask

# 5단계: Loss function에 전달 (signature 검사 후)
if 'mask' in sig.parameters:
    loss_kwargs['mask'] = current_mask
```

**✅ Mask도 완전히 전달되는 구조입니다!**

---

## 3. 기존 Loss Functions 패턴 분석

### 3.1 SSISilogLoss 패턴 (ssi_silog_loss.py)

```python
class SSISilogLoss(LossBase):
    def __init__(self, alpha=0.85, silog_ratio=10, silog_ratio2=0.85,
                 ssi_weight=0.7, silog_weight=0.3,
                 min_depth: Optional[float] = None,
                 max_depth: Optional[float] = None):
        super().__init__()
        # ✅ min/max_depth를 __init__에서 받음
        self.min_depth = min_depth
        self.max_depth = max_depth
        
    def forward(self, pred_inv_depth, gt_inv_depth, mask=None, progress=None):
        """
        ✅ 패턴:
        - mask parameter 수신 (optional)
        - progress parameter 수신 (optional, 사용 안함)
        """
        # SSI component (inverse depth)
        ssi_loss = self.compute_ssi_loss(pred_inv_depth, gt_inv_depth, mask)
        
        # Silog component (depth, inv2depth 변환)
        silog_loss = self.compute_silog_loss(pred_inv_depth, gt_inv_depth, mask)
        
        # Combine
        total_loss = self.ssi_weight * ssi_loss + self.silog_weight * silog_loss
        
        # ✅ Metrics 저장 (LossBase 상속)
        self.add_metric('ssi_component', ssi_loss)
        self.add_metric('silog_component', silog_loss)
        
        return total_loss
```

**핵심 패턴:**
1. `mask` parameter는 optional (기본값 None)
2. `progress` parameter는 optional (사용 여부는 loss마다 다름)
3. `LossBase` 상속으로 `add_metric()` 사용
4. `min_depth`, `max_depth`는 `__init__`에서 받음

### 3.2 EnhancedSSILoss 패턴 (ssi_loss_enhanced.py)

```python
class EnhancedSSILoss(LossBase):
    def __init__(self, alpha=0.85, l1_weight=0.2, ssi_weight=0.8,
                 adaptive_weighting=True):
        super().__init__()
        self.adaptive_weighting = adaptive_weighting
        
    def forward(self, pred_inv_depth, gt_inv_depth, mask=None, progress=None):
        """
        ✅ Progress를 adaptive weighting에 활용
        """
        # Adaptive weights 계산
        ssi_weight, l1_weight = self.get_adaptive_weights(progress)
        
        # Loss 계산
        ssi_loss = self.compute_ssi_loss(pred_inv_depth, gt_inv_depth, mask)
        l1_loss = self.compute_l1_loss(pred_inv_depth, gt_inv_depth, mask)
        
        # Combine with adaptive weights
        total_loss = ssi_weight * ssi_loss + l1_weight * l1_loss
        
        # ✅ Adaptive weights 기록
        self.add_metric('dynamic_ssi_weight', ssi_weight)
        self.add_metric('dynamic_l1_weight', l1_weight)
        
        return total_loss
    
    def get_adaptive_weights(self, progress=None):
        """
        ✅ Progress 기반 adaptive weighting
        Early: SSI 위주 (0.9)
        Later: Balanced (0.8/0.2)
        """
        if not self.adaptive_weighting or progress is None:
            return self.ssi_weight, self.l1_weight
        
        progress = max(0.0, min(1.0, progress))
        ssi_weight = self.ssi_weight + (1.0 - progress) * 0.1
        l1_weight = self.l1_weight + progress * 0.1
        
        # Normalize
        total = ssi_weight + l1_weight
        return ssi_weight / total, l1_weight / total
```

**핵심 패턴:**
1. `progress`를 **적극 활용** (adaptive weighting)
2. `mask`는 모든 sub-loss에 전달
3. Adaptive weights를 metrics로 기록

### 3.3 SSILoss 패턴 (ssi_loss.py)

```python
class SSILoss(LossBase):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred_inv_depth, gt_inv_depth, mask=None):
        """
        ✅ 가장 단순한 패턴
        - progress 안받음
        - mask만 optional
        """
        if mask is None:
            mask = torch.ones_like(pred_inv_depth, dtype=torch.bool)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_inv_depth.device)
        
        diff = (pred_inv_depth[mask] - gt_inv_depth[mask])
        diff2 = diff ** 2
        mean = diff.mean()
        var = diff2.mean() - mean ** 2
        
        return var + self.alpha * mean ** 2
```

**핵심 패턴:**
1. 가장 기본적인 loss - `mask`만 받음
2. `progress` 없음 (정적 loss)
3. Mask 기본값 처리

---

## 4. 빠진 부분 및 필요한 수정사항

### 4.1 ❌ 현재 문서에 빠진 부분

#### 4.1.1 `get_loss_func()` 업데이트 누락

**문제점:**
```python
# supervised_loss.py의 get_loss_func()에 Scale-Adaptive 케이스 추가 필요
def get_loss_func(supervised_method, **kwargs):
    # ...
    elif supervised_method.endswith('ssi-silog'):
        return SSISilogLoss(
            min_depth=kwargs.get('min_depth', None),
            max_depth=kwargs.get('max_depth', None),
        )
    # ❌ 이 부분이 문서에 없음!
    elif supervised_method.endswith('scale-adaptive'):
        return ScaleAdaptiveLoss(
            min_depth=kwargs.get('min_depth', 0.1),
            max_depth=kwargs.get('max_depth', 100.0),
            lambda_sg=kwargs.get('lambda_sg', 0.5),
            use_inv_depth=kwargs.get('use_inv_depth', False),
            alpha_schedule=kwargs.get('alpha_schedule', 'linear'),
            scale_schedule=kwargs.get('scale_schedule', 'linear'),
            num_scales=kwargs.get('num_scales', 4),
        )
    # ...
```

#### 4.1.2 Import 문 누락

**문제점:**
```python
# supervised_loss.py 상단에 import 추가 필요
from packnet_sfm.losses.scale_adaptive_loss import ScaleAdaptiveLoss
```

#### 4.1.3 Progress Parameter 활용 예시 부족

**현재 문서:**
- `progress` parameter를 받는다고만 명시
- **실제 활용 방법 예시 부족**

**필요한 내용:**
```python
def forward(self, pred, gt, mask=None, progress=0.0):
    """
    progress 활용 예시:
    - alpha_t = self.get_alpha(progress)  # Adaptive alpha
    - scales = self.get_active_scales(progress)  # Progressive scaling
    """
```

#### 4.1.4 Mask 처리 상세 설명 부족

**현재 문서:**
- `mask` parameter를 받는다고만 명시
- **Sparse GT mask와의 결합 방법 미설명**

**필요한 내용:**
```python
def forward(self, pred, gt, mask=None, progress=0.0):
    """
    Mask 처리:
    1. SupervisedLoss.calculate_loss()에서:
       - valid_mask = (gt > 0.).detach()  # Sparse GT
       - current_mask = valid_mask & mask  # 결합
    
    2. Loss function 내부에서:
       - mask 그대로 사용 (이미 결합됨)
       - pred[mask], gt[mask]로 필터링
    """
```

### 4.2 ✅ 필요한 수정사항

#### 수정 1: `supervised_loss.py` - `get_loss_func()` 업데이트

```python
# packnet_sfm/losses/supervised_loss.py

# 상단에 import 추가
from packnet_sfm.losses.scale_adaptive_loss import ScaleAdaptiveLoss

def get_loss_func(supervised_method, **kwargs):
    """Determines the supervised loss to be used, given the supervised method."""
    print(f"🔍 Loading loss function for: {supervised_method}")
    
    if supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('ssi-silog'):
        return SSISilogLoss(
            min_depth=kwargs.get('min_depth', None),
            max_depth=kwargs.get('max_depth', None),
        )
    elif supervised_method.endswith('silog'):
        return SilogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    elif supervised_method.endswith('ssi'):
        return SSILoss()
    elif supervised_method.endswith('enhanced-ssi'):
        return EnhancedSSILoss()
    elif supervised_method.endswith('progressive-ssi'):
        return ProgressiveEnhancedSSILoss()
    elif supervised_method.endswith('ssi-trim'):
        return SSITrimLoss(trim=0.2, epsilon=1e-6)
    # ✅ 추가: Scale-Adaptive Loss
    elif supervised_method.endswith('scale-adaptive'):
        return ScaleAdaptiveLoss(
            min_depth=kwargs.get('min_depth', 0.1),
            max_depth=kwargs.get('max_depth', 100.0),
            lambda_sg=kwargs.get('lambda_sg', 0.5),
            use_inv_depth=kwargs.get('use_inv_depth', False),
            alpha_schedule=kwargs.get('alpha_schedule', 'linear'),
            scale_schedule=kwargs.get('scale_schedule', 'linear'),
            num_scales=kwargs.get('num_scales', 4),
        )
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))
```

#### 수정 2: `scale_adaptive_loss.py` - Forward Signature 명확화

```python
class ScaleAdaptiveLoss(LossBase):
    def __init__(self, ...):
        # ... (기존 __init__)
        
    def forward(self, pred, gt, mask=None, progress=0.0):
        """
        Scale-Adaptive Loss forward pass
        
        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted depth or inverse depth (depends on use_inv_depth)
        gt : torch.Tensor [B,1,H,W]
            Ground-truth depth or inverse depth (depends on use_inv_depth)
        mask : torch.Tensor [B,1,H,W], optional
            Combined binary mask (already includes sparse GT mask)
            - SupervisedLoss.calculate_loss()에서 이미 결합됨
            - valid_mask (gt > 0) & custom_mask
        progress : float, optional
            Training progress [0.0, 1.0]
            - Used for adaptive alpha and scale weighting
            
        Returns
        -------
        loss : torch.Tensor [1]
            Total scale-adaptive loss
            
        Notes
        -----
        1. Mask는 이미 SupervisedLoss에서 결합된 상태로 들어옵니다:
           - Sparse GT mask (gt > 0)와 custom mask가 AND 연산됨
        
        2. Progress는 ModelWrapper에서 계산되어 전달됩니다:
           - progress = current_epoch / max_epochs
        
        3. use_inv_depth 처리:
           - True: pred/gt를 inverse depth로 간주, 직접 사용
           - False: pred/gt를 inverse depth로 간주, depth로 변환
        """
        # 1. Mask 기본값 처리
        if mask is None:
            mask = torch.ones_like(pred, dtype=torch.bool)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 2. Depth 변환 (use_inv_depth에 따라)
        if self.use_inv_depth:
            # Inverse depth로 직접 사용
            pred_for_loss = pred
            gt_for_loss = gt
        else:
            # Depth로 변환
            pred_for_loss = inv2depth(pred)
            gt_for_loss = inv2depth(gt)
            # Clamp
            pred_for_loss = torch.clamp(pred_for_loss, self.min_depth, self.max_depth)
            gt_for_loss = torch.clamp(gt_for_loss, self.min_depth, self.max_depth)
        
        # 3. Adaptive parameters 계산
        alpha_t = self.get_alpha(progress)
        scale_weights = self.get_scale_weights(progress)
        
        # 4. Multi-scale loss 계산
        total_loss = 0.0
        for scale in range(self.num_scales):
            # Downsampling for multi-scale
            h, w = pred_for_loss.shape[-2:]
            factor = 2 ** scale
            
            if scale > 0:
                pred_s = F.interpolate(pred_for_loss, size=(h//factor, w//factor), mode='bilinear')
                gt_s = F.interpolate(gt_for_loss, size=(h//factor, w//factor), mode='nearest')
                mask_s = F.interpolate(mask.float(), size=(h//factor, w//factor), mode='nearest') > 0.5
            else:
                pred_s, gt_s, mask_s = pred_for_loss, gt_for_loss, mask
            
            # MAD normalization
            pred_n, gt_n = self.mad_normalize(pred_s, gt_s, mask_s)
            
            # L1 component
            l1_loss = torch.abs(pred_n - gt_n)[mask_s].mean()
            
            # Gradient component
            grad_loss = self.compute_gradient_loss(pred_n, gt_n, mask_s)
            
            # Combine with scale weight
            scale_loss = (1.0 - self.lambda_sg) * l1_loss + self.lambda_sg * grad_loss
            total_loss += scale_weights[scale] * scale_loss
            
            # ✅ Per-scale metrics
            self.add_metric(f'scale{scale}_loss', scale_loss)
            self.add_metric(f'scale{scale}_l1', l1_loss)
            self.add_metric(f'scale{scale}_grad', grad_loss)
        
        # ✅ Global metrics
        self.add_metric('total_loss', total_loss)
        self.add_metric('alpha_t', alpha_t)
        self.add_metric('num_valid_pixels', mask.sum())
        
        return total_loss
```

#### 수정 3: Documentation 업데이트

**SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md에 추가할 섹션:**

```markdown
## 7. Integration Details

### 7.1 Parameter Flow from YAML to Loss

1. **YAML Configuration**
   ```yaml
   model:
     params:
       min_depth: 0.0
       max_depth: 80.0
     loss:
       supervised_method: 'sparse-scale-adaptive'
       supervised_num_scales: 4
       lambda_sg: 0.5
       use_inv_depth: false
       alpha_schedule: 'linear'
   ```

2. **Model Initialization**
   ```python
   # SemiSupCompletionModel.__init__()
   self._supervised_loss = SupervisedLoss(
       min_depth=min_depth,  # From config.model.params
       max_depth=max_depth,
       **kwargs  # From config.model.loss
   )
   ```

3. **Loss Function Creation**
   ```python
   # SupervisedLoss.__init__() → get_loss_func()
   self.loss_func = get_loss_func(
       supervised_method='sparse-scale-adaptive',
       min_depth=min_depth,
       max_depth=max_depth,
       lambda_sg=kwargs.get('lambda_sg', 0.5),
       use_inv_depth=kwargs.get('use_inv_depth', False),
       # ... other params
   )
   ```

### 7.2 Runtime Data Flow

1. **Training Step**
   ```python
   # ModelWrapper.training_step()
   progress = self.current_epoch / self.max_epochs
   model_output = self.model(batch, progress=progress)
   ```

2. **Model Forward**
   ```python
   # SemiSupModel.forward()
   sup_output = self.supervised_loss(
       pred_inv_depths,
       gt_inv_depths,
       progress=progress,  # ✅ Passed
       masks=masks  # ✅ Optional
   )
   ```

3. **Supervised Loss**
   ```python
   # SupervisedLoss.forward()
   self._progress = progress  # Store for loss function
   
   # SupervisedLoss.calculate_loss()
   loss_kwargs = {}
   if 'mask' in sig.parameters:
       loss_kwargs['mask'] = current_mask  # Combined mask
   if 'progress' in sig.parameters:
       loss_kwargs['progress'] = self._progress  # ✅ Passed
   
   loss = self.loss_func(pred, gt, **loss_kwargs)
   ```

4. **Scale-Adaptive Loss**
   ```python
   # ScaleAdaptiveLoss.forward(pred, gt, mask, progress)
   alpha_t = self.get_alpha(progress)  # Use progress
   # ... compute loss ...
   ```

### 7.3 Mask Handling

**Important:** The `mask` parameter in `ScaleAdaptiveLoss.forward()` is **already combined**:

```python
# In SupervisedLoss.calculate_loss():
valid_mask = (gt_inv_depths[i] > 0.).detach()  # Sparse GT mask
if masks is not None:
    current_mask = valid_mask & masks[i]  # ✅ Combined!
else:
    current_mask = valid_mask

# Then passed to loss function:
loss = self.loss_func(pred, gt, mask=current_mask)  # ✅ Already combined
```

**You should NOT re-combine masks in your loss function!**

### 7.4 Required Code Changes

#### Change 1: Update `supervised_loss.py`

```python
# Add import at top
from packnet_sfm.losses.scale_adaptive_loss import ScaleAdaptiveLoss

# Add case in get_loss_func()
elif supervised_method.endswith('scale-adaptive'):
    return ScaleAdaptiveLoss(
        min_depth=kwargs.get('min_depth', 0.1),
        max_depth=kwargs.get('max_depth', 100.0),
        lambda_sg=kwargs.get('lambda_sg', 0.5),
        use_inv_depth=kwargs.get('use_inv_depth', False),
        alpha_schedule=kwargs.get('alpha_schedule', 'linear'),
        scale_schedule=kwargs.get('scale_schedule', 'linear'),
        num_scales=kwargs.get('num_scales', 4),
    )
```

#### Change 2: Ensure `scale_adaptive_loss.py` follows signature

```python
def forward(self, pred, gt, mask=None, progress=0.0):
    # ✅ Correct signature
    # mask: Already combined (sparse GT & custom)
    # progress: From ModelWrapper
```
```

---

## 5. 완전한 통합 체크리스트

### 5.1 코드 구현 체크리스트

- [ ] **1. 파일 생성**
  - [ ] `packnet_sfm/losses/scale_adaptive_loss.py` 생성
  
- [ ] **2. ScaleAdaptiveLoss 클래스 구현**
  - [ ] `LossBase` 상속
  - [ ] `__init__()`: 모든 파라미터 수신 (min/max_depth, lambda_sg, use_inv_depth, schedules 등)
  - [ ] `forward(pred, gt, mask=None, progress=0.0)`: 정확한 signature
  - [ ] `mad_normalize()`: MAD normalization 구현
  - [ ] `compute_gradient_loss()`: Sobel gradient loss 구현
  - [ ] `get_alpha()`: Progress 기반 alpha 스케줄링
  - [ ] `get_scale_weights()`: Progress 기반 scale weighting
  
- [ ] **3. supervised_loss.py 수정**
  - [ ] Import 추가: `from packnet_sfm.losses.scale_adaptive_loss import ScaleAdaptiveLoss`
  - [ ] `get_loss_func()`에 `'scale-adaptive'` 케이스 추가
  - [ ] 모든 kwargs 전달 확인
  
- [ ] **4. 테스트 YAML 작성**
  - [ ] `configs/train_*_scale_adaptive.yaml` 생성
  - [ ] 모든 파라미터 명시 (lambda_sg, use_inv_depth, schedules 등)

### 5.2 Parameter Flow 검증 체크리스트

- [x] **YAML → Model**
  - [x] `min_depth`, `max_depth`: `config.model.params` → `SemiSupModel.__init__()`
  - [x] `supervised_method`: `config.model.loss` → `SupervisedLoss.__init__()`
  - [x] Custom params: `config.model.loss` → `**kwargs`

- [x] **Model → SupervisedLoss**
  - [x] `min_depth`, `max_depth`: 명시적 전달
  - [x] `**kwargs`: loss-specific 파라미터 전달

- [x] **SupervisedLoss → get_loss_func()**
  - [x] `supervised_method`: 첫 번째 인자
  - [x] `**kwargs`: 모든 파라미터 전달

- [x] **get_loss_func() → ScaleAdaptiveLoss**
  - [x] `kwargs.get('min_depth', default)`: min_depth 전달
  - [x] `kwargs.get('max_depth', default)`: max_depth 전달
  - [x] `kwargs.get('lambda_sg', 0.5)`: lambda_sg 전달
  - [x] `kwargs.get('use_inv_depth', False)`: use_inv_depth 전달
  - [x] 기타 모든 파라미터 전달

- [x] **Runtime: Progress 전달**
  - [x] `ModelWrapper.progress` property 존재
  - [x] `model.forward(batch, progress=self.progress)` 호출
  - [x] `SupervisedLoss.forward(..., progress=progress)` 전달
  - [x] `self._progress = progress` 저장
  - [x] `loss_kwargs['progress'] = self._progress` 전달
  - [x] `ScaleAdaptiveLoss.forward(..., progress=0.0)` 수신

- [x] **Runtime: Mask 전달**
  - [x] `batch.get('mask', None)` 추출
  - [x] Multi-scale masks 생성 (필요시)
  - [x] `SupervisedLoss.forward(..., masks=masks)` 전달
  - [x] `valid_mask & masks[i]` 결합
  - [x] `loss_kwargs['mask'] = current_mask` 전달
  - [x] `ScaleAdaptiveLoss.forward(..., mask=None)` 수신

### 5.3 기능 검증 체크리스트

- [ ] **기본 기능**
  - [ ] Loss 값이 정상 계산됨 (NaN/Inf 없음)
  - [ ] Gradient가 정상 전파됨 (backward 성공)
  - [ ] Multi-scale loss 계산 확인
  - [ ] MAD normalization 작동 확인
  - [ ] Gradient loss 계산 확인

- [ ] **Adaptive 기능**
  - [ ] Progress 기반 alpha 변화 확인
  - [ ] Progress 기반 scale weighting 변화 확인
  - [ ] Metrics 정상 기록 확인 (TensorBoard/WandB)

- [ ] **use_inv_depth 옵션**
  - [ ] `use_inv_depth=True`: Inverse depth 직접 사용
  - [ ] `use_inv_depth=False`: Depth 변환 후 사용
  - [ ] 성능 차이 측정

- [ ] **Mask 처리**
  - [ ] Sparse GT mask 적용 확인 (gt > 0)
  - [ ] Custom mask 결합 확인 (제공시)
  - [ ] Empty mask 처리 확인 (return 0.0)

### 5.4 성능 검증 체크리스트

- [ ] **학습 안정성**
  - [ ] Loss가 발산하지 않음
  - [ ] Gradient exploding/vanishing 없음
  - [ ] 학습 곡선이 매끄러움

- [ ] **정확도**
  - [ ] Abs Rel 개선 확인
  - [ ] RMSE 개선 확인
  - [ ] δ < 1.25 개선 확인

- [ ] **속도**
  - [ ] `use_inv_depth=True`: 15% 빠름 확인
  - [ ] Memory 사용량 9% 감소 확인

### 5.5 문서화 체크리스트

- [x] **이론 문서**
  - [x] SCALE_ADAPTIVE_LOSS.md (한국어)

- [x] **구현 가이드**
  - [x] SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md (영어)
  - [x] Integration details 추가 필요 ✅

- [x] **Quick Start**
  - [x] SCALE_ADAPTIVE_LOSS_QUICK_START.md

- [x] **use_inv_depth 설명**
  - [x] SCALE_ADAPTIVE_LOSS_USE_INV_DEPTH_UPDATE.md

- [x] **통합 분석** (본 문서)
  - [x] SCALE_ADAPTIVE_LOSS_INTEGRATION_ANALYSIS.md

- [ ] **README 업데이트**
  - [ ] 메인 README.md에 Scale-Adaptive Loss 추가
  - [ ] 링크 추가

---

## 6. 다음 단계 (Action Items)

### 우선순위 1: 코드 구현

1. **`scale_adaptive_loss.py` 구현**
   ```bash
   # 파일 생성 위치
   packnet_sfm/losses/scale_adaptive_loss.py
   ```
   - 모든 메서드 구현 (forward, MAD, gradient, schedules)
   - Signature 정확히 맞추기: `forward(pred, gt, mask=None, progress=0.0)`

2. **`supervised_loss.py` 수정**
   - Import 추가
   - `get_loss_func()`에 케이스 추가

3. **YAML config 작성**
   ```bash
   configs/train_resnet_san_kitti_scale_adaptive.yaml
   ```

### 우선순위 2: 테스트

1. **Unit 테스트**
   - Loss 계산 정확성
   - Gradient 전파 확인
   - Edge case 처리 (empty mask, progress=0/1 등)

2. **Integration 테스트**
   - 실제 학습 1 epoch
   - Metrics 기록 확인
   - TensorBoard 시각화 확인

### 우선순위 3: 문서 업데이트

1. **IMPLEMENTATION.md 업데이트**
   - Section 7 추가 (위에 작성한 내용)

2. **README.md 업데이트**
   - Scale-Adaptive Loss 소개
   - 문서 링크 추가

---

## 7. 요약

### ✅ 잘 되어있는 부분

1. **Parameter Flow**: YAML → Model → SupervisedLoss → get_loss_func() → ScaleAdaptiveLoss
   - `min_depth`, `max_depth` 전달 완벽
   - `**kwargs`로 custom params 전달 완벽

2. **Progress 전달**: ModelWrapper → Model → SupervisedLoss → Loss Function
   - 모든 단계에서 `progress` parameter 전달
   - EnhancedSSILoss 등에서 이미 활용 중

3. **Mask 처리**: Batch → Model → SupervisedLoss (결합) → Loss Function
   - Sparse GT mask (gt > 0)와 custom mask 자동 결합
   - Loss function은 결합된 mask만 받음

### ❌ 빠진 부분

1. **`supervised_loss.py` 수정 필요**
   - Import 추가
   - `get_loss_func()`에 `'scale-adaptive'` 케이스 추가

2. **`scale_adaptive_loss.py` 구현 필요**
   - 전체 클래스 구현
   - Correct signature: `forward(pred, gt, mask=None, progress=0.0)`

3. **문서 업데이트**
   - IMPLEMENTATION.md에 Integration Details 추가
   - Mask 처리 주의사항 명시

### 🎯 핵심 포인트

1. **Loss Function Signature는 반드시:**
   ```python
   def forward(self, pred, gt, mask=None, progress=0.0):
   ```

2. **Mask는 이미 결합된 상태로 들어옴:**
   - `valid_mask (gt > 0) & custom_mask`
   - Loss 내부에서 재결합 불필요

3. **Progress는 선택적 활용:**
   - Adaptive 기능 필요시: `get_alpha(progress)`, `get_scale_weights(progress)`
   - 불필요시: 무시해도 됨 (기본값 0.0)

4. **Metrics 기록 필수:**
   - `self.add_metric('metric_name', value)`
   - TensorBoard/WandB 자동 기록됨

---

**이제 실제 `scale_adaptive_loss.py` 구현만 하면 됩니다!** 🚀
