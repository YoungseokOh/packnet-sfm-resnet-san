# Evaluation 함수 상세 분석 및 수정 가이드

**작성일**: 2025-10-28  
**목적**: `evaluate_depth` 함수의 동작 원리와 Post-Processing 평가 구현을 위한 상세 가이드

---

## 📋 목차

1. [현재 evaluate_depth 함수 상세 분석](#1-현재-evaluate_depth-함수-상세-분석)
2. [새로운 evaluate_depth 함수 완전 구현](#2-새로운-evaluate_depth-함수-완전-구현)
3. [Helper 함수들 상세 분석](#3-helper-함수들-상세-분석)
4. [Validation Step 흐름](#4-validation-step-흐름)
5. [코드 변경 전후 비교](#5-코드-변경-전후-비교)

---

## 1. 현재 evaluate_depth 함수 상세 분석

### 1.1 함수 시그니처와 위치

**파일**: `packnet_sfm/models/model_wrapper.py`  
**라인**: 592-706  
**함수**: `def evaluate_depth(self, batch)`

---

### 1.2 현재 구현 전체 코드

```python
def evaluate_depth(self, batch):
    """
    Evaluate batch to produce depth metrics.
    
    현재 동작:
    1. 모델 forward → inv_depths (실제로는 depth) 반환
    2. Flip TTA 적용
    3. GT depth 준비
    4. 4가지 metrics 계산 (depth, depth_pp, depth_gt, depth_pp_gt)
    """
    
    # ========================================
    # STEP 1: 디버깅 로그 (GT depth 확인)
    # ========================================
    if 'depth' in batch and batch['depth'] is not None:
        raw_depth = batch['depth']
        if hasattr(raw_depth, 'max'):
            max_val = float(raw_depth.max())
            if not hasattr(self, '_batch_depth_logged'):
                self._batch_depth_logged = True
                print(f"\n[evaluate_depth] Incoming batch['depth']:")
                print(f"  Type: {type(raw_depth)}")
                print(f"  Shape: {raw_depth.shape if hasattr(raw_depth, 'shape') else 'N/A'}")
                print(f"  Max value: {max_val:.2f}")
                print(f"  Min value: {float(raw_depth.min()):.2f}")
                if max_val > 500:
                    print(f"  ⚠️ WARNING: Max > 500, seems like 256x scaled!")
    
    # ========================================
    # STEP 2: 모델 forward (정방향 예측)
    # ========================================
    inv_depths = self.model(batch)['inv_depths']  # list, 첫 스케일: (B,1,H,W)
    inv0 = inv_depths[0]  # 첫 번째 스케일만 사용
    depth = inv2depth(inv0)  # (B,1,H,W) - 실제로는 이미 depth라서 1/depth 계산
    
    # ========================================
    # STEP 3: Flip TTA (Test Time Augmentation)
    # ========================================
    # 3-1) 입력 이미지 좌우 반전
    batch['rgb'] = flip_lr(batch['rgb'])
    
    # 3-2) 반전된 이미지로 예측
    inv_depths_flipped = self.model(batch)['inv_depths']
    
    # 3-3) 예측 결과를 다시 반전 (원래 좌표로)
    inv0_flipped_back = flip_lr(inv_depths_flipped[0])
    
    # 3-4) Post-process: 원본과 반전 예측 결합
    inv_depth_pp = post_process_inv_depth(inv0, inv0_flipped_back, method='mean')
    depth_pp = inv2depth(inv_depth_pp)
    
    # 3-5) 입력 이미지 복원
    batch['rgb'] = flip_lr(batch['rgb'])
    
    # ========================================
    # STEP 4: Depth 텐서 정규화 (B,1,H,W)
    # ========================================
    device = inv0.device
    
    def _to_b1hw(x):
        """
        다양한 형태의 텐서를 (B,1,H,W) 형태로 정규화
        
        입력 가능 형태:
        - numpy array
        - torch.Tensor (0D, 2D, 3D, 4D)
        
        출력:
        - (B,1,H,W) torch.Tensor on correct device
        """
        if x is None:
            return None
        
        # NumPy → Torch
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        if not isinstance(x, torch.Tensor):
            return None
        
        # Device & dtype 변환
        x = x.to(device=device, dtype=torch.float32)
        
        # Dimension 정규화
        if x.dim() == 0:  # Scalar
            return x.view(1, 1, 1, 1)
        
        if x.dim() == 2:  # (H, W)
            return x.unsqueeze(0).unsqueeze(0)  # → (1, 1, H, W)
        
        if x.dim() == 3:  # (C, H, W) or (B, H, W)
            if x.size(0) in (1, 3):  # (C, H, W) - 채널 first
                x = x.unsqueeze(0)  # → (1, C, H, W)
                return x[:, :1, ...]  # → (1, 1, H, W)
            else:  # (B, H, W) - 배치 first
                return x.unsqueeze(1)  # → (B, 1, H, W)
        
        if x.dim() == 4:  # (B, C, H, W)
            if x.size(1) != 1:  # 채널이 1이 아니면
                return x[:, :1, ...]  # 첫 번째 채널만
            return x
        
        return None
    
    depth_pred    = _to_b1hw(depth)
    depth_pred_pp = _to_b1hw(depth_pp)
    depth_gt      = _to_b1hw(batch.get('depth', None))
    
    # ========================================
    # STEP 5: 임시 스케일 보정 (환경변수)
    # ========================================
    if os.environ.get('FORCE_DEPTH_DIV256', '0') == '1':
        def _div256(x):
            if x is None:
                return x
            # 값이 이미 물리 단위(최대 < 200 등)면 중복 나눔 피함
            if torch.is_tensor(x) and x.max() > 255:
                return x / 256.0
            return x
        
        depth_gt      = _div256(depth_gt)
        depth_pred    = _div256(depth_pred)
        depth_pred_pp = _div256(depth_pred_pp)
    
    # ========================================
    # STEP 6: Metrics 계산 (4가지)
    # ========================================
    metrics = OrderedDict()
    
    if depth_gt is not None and depth_pred is not None:
        # 6-1) depth: TTA 없음, GT scale 없음
        try:
            m_main = compute_depth_metrics(
                self.config.model.params, 
                gt=depth_gt, 
                pred=depth_pred, 
                use_gt_scale=False
            )
        except Exception:
            m_main = self._compute_depth_metrics_fallback(depth_gt, depth_pred)
        metrics['depth'] = m_main
        
        # 6-2) depth_pp: TTA 있음, GT scale 없음
        try:
            m_pp = compute_depth_metrics(
                self.config.model.params, 
                gt=depth_gt, 
                pred=depth_pred_pp, 
                use_gt_scale=False
            )
        except Exception:
            m_pp = self._compute_depth_metrics_fallback(depth_gt, depth_pred_pp)
        metrics['depth_pp'] = m_pp
        
        # 6-3) depth_gt: TTA 없음, GT scale 있음
        try:
            m_gt = compute_depth_metrics(
                self.config.model.params, 
                gt=depth_gt, 
                pred=depth_pred, 
                use_gt_scale=True
            )
        except Exception:
            m_gt = self._compute_depth_metrics_fallback(depth_gt, depth_pred)
        metrics['depth_gt'] = m_gt
        
        # 6-4) depth_pp_gt: TTA 있음, GT scale 있음
        try:
            m_pp_gt = compute_depth_metrics(
                self.config.model.params, 
                gt=depth_gt, 
                pred=depth_pred_pp, 
                use_gt_scale=True
            )
        except Exception:
            m_pp_gt = self._compute_depth_metrics_fallback(depth_gt, depth_pred_pp)
        metrics['depth_pp_gt'] = m_pp_gt
    
    # ========================================
    # STEP 7: 결과 반환
    # ========================================
    return {
        'metrics': metrics,  # OrderedDict with 4 entries
        'inv_depth': inv_depth_pp  # For visualization
    }
```

---

### 1.3 핵심 동작 분석

#### STEP 2: 모델 Forward

**문제점**: 
```python
inv_depths = self.model(batch)['inv_depths']  # 이름은 inv_depths
depth = inv2depth(inv0)  # 하지만 실제로는 이미 depth!
```

**이유**:
- `ResNetSAN01.run_network`에서 `disp_to_inv` 함수가 이미 depth를 반환
- 따라서 `inv2depth(inv0)` = `1 / depth` = inverse depth로 다시 변환
- 혼란스러운 명명

**해결책** (새 구현):
```python
sigmoid_outputs = self.model(batch)['inv_depths']  # 진짜 sigmoid [0,1]
# 변환은 evaluate_depth에서 처리
```

---

#### STEP 3: Flip TTA

**동작 원리**:
```python
# TTA (Test Time Augmentation) - Flip
# 목적: 좌우 반전 예측을 결합하여 정확도 향상

# 1. 원본 예측: inv0
# 2. 반전 예측: inv0_flipped_back
# 3. Post-process: 두 예측을 융합 (가장자리 smoothing)
```

**post_process_inv_depth 함수** (`utils/depth.py` line 229):
```python
def post_process_inv_depth(inv_depth, inv_depth_flipped, method='mean'):
    """
    가장자리 처리:
    - 이미지 좌측 5% → 원본 예측 사용
    - 이미지 우측 5% → 반전 예측 사용
    - 나머지 90% → 두 예측의 평균 (method='mean')
    
    이유:
    - 가장자리는 occlusion이 많아서 한쪽 예측이 더 정확
    - 중앙은 평균으로 노이즈 감소
    """
    B, C, H, W = inv_depth.shape
    inv_depth_hat = flip_lr(inv_depth_flipped)
    inv_depth_fused = fuse_inv_depth(inv_depth, inv_depth_hat, method=method)
    
    # Mask: 좌측 5% ~ 우측 5% smoothing
    xs = torch.linspace(0., 1., W, device=inv_depth.device,
                        dtype=inv_depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    
    return mask_hat * inv_depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused
```

---

#### STEP 6: Metrics 계산

**4가지 조합**:

| Metric | TTA | GT Scale | 설명 |
|--------|-----|----------|------|
| `depth` | ❌ | ❌ | 원본 예측, 원본 스케일 |
| `depth_pp` | ✅ | ❌ | TTA 적용, 원본 스케일 |
| `depth_gt` | ❌ | ✅ | 원본 예측, GT median scale |
| `depth_pp_gt` | ✅ | ✅ | TTA 적용, GT median scale |

**GT Scale이란**:
```python
# use_gt_scale=True일 때:
gt_median = torch.median(gt)
pred_median = torch.median(pred)
pred_scaled = pred * (gt_median / pred_median)  # Scale alignment

# 목적: Monocular depth는 절대 스케일 모름
# GT median으로 scaling하여 상대 정확도만 평가
```

---

### 1.4 Fallback 함수

**위치**: `packnet_sfm/models/model_wrapper.py` line 556

```python
def _compute_depth_metrics_fallback(self, gt, pred):
    """
    compute_depth_metrics가 실패할 때 fallback
    
    입력: gt/pred (B,1,H,W) float tensors
    출력: [abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3]
    """
    eps = 1e-6
    params = getattr(self.config.model, 'params', {})
    
    # Config에서 min/max depth 추출
    try:
        min_d = float(params.get('min_depth', 0.1))
        max_d = float(params.get('max_depth', 80.0))
    except Exception:
        min_d, max_d = 0.1, 80.0
    
    # Clamp depth range
    gt = gt.clamp(min=min_d, max=max_d)
    pred = pred.clamp(min=min_d, max=max_d)
    
    # Valid mask
    mask = torch.isfinite(gt) & torch.isfinite(pred) & \
           (gt > min_d) & (gt < max_d)
    
    if mask.float().sum() == 0:
        return torch.zeros(7, device=gt.device, dtype=torch.float32)
    
    gt_m = gt[mask]
    pred_m = pred[mask]
    
    # Metrics 계산
    abs_rel = (torch.abs(gt_m - pred_m) / (gt_m + eps)).mean()
    sqr_rel = (((gt_m - pred_m) ** 2) / (gt_m + eps)).mean()
    rmse = torch.sqrt(((gt_m - pred_m) ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(gt_m + eps) - torch.log(pred_m + eps)) ** 2).mean())
    
    thresh = torch.max(gt_m / (pred_m + eps), pred_m / (gt_m + eps))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    
    return torch.stack([abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3]).to(gt.dtype)
```

---

## 2. 새로운 evaluate_depth 함수 완전 구현

### 2.1 전체 코드 (Linear + Log 평가)

**파일**: `packnet_sfm/models/model_wrapper.py`  
**함수**: `evaluate_depth` (전체 교체)

```python
def evaluate_depth(self, batch):
    """
    Evaluate batch with both Linear and Log post-processing
    
    새로운 동작:
    1. 모델 forward → sigmoid output [0,1] 반환
    2. Post-processing: Linear & Log 변환
    3. GT depth 준비
    4. 4가지 metrics 계산 (Linear 2개 + Log 2개)
    
    선택적 기능 (주석 처리):
    - Flip TTA: 추론 시간 2배 증가, 복잡도 증가
    """
    
    # ========================================
    # STEP 1: 디버깅 로그 (동일)
    # ========================================
    if 'depth' in batch and batch['depth'] is not None:
        raw_depth = batch['depth']
        if hasattr(raw_depth, 'max'):
            max_val = float(raw_depth.max())
            if not hasattr(self, '_batch_depth_logged'):
                self._batch_depth_logged = True
                print(f"\n[evaluate_depth] Incoming batch['depth']:")
                print(f"  Type: {type(raw_depth)}")
                print(f"  Shape: {raw_depth.shape if hasattr(raw_depth, 'shape') else 'N/A'}")
                print(f"  Max value: {max_val:.2f}")
                print(f"  Min value: {float(raw_depth.min()):.2f}")
                if max_val > 500:
                    print(f"  ⚠️ WARNING: Max > 500, seems like 256x scaled!")
    
    # ========================================
    # STEP 2: 모델 forward → Sigmoid output
    # ========================================
    sigmoid_outputs = self.model(batch)['inv_depths']  # ✅ 이제 진짜 sigmoid!
    sigmoid0 = sigmoid_outputs[0]  # (B,1,H,W) ∈ [0, 1]
    
    # ========================================
    # STEP 3: Config에서 depth range 추출
    # ========================================
    min_depth = float(self.config.model.params.min_depth)
    max_depth = float(self.config.model.params.max_depth)
    
    if not hasattr(self, '_depth_range_logged'):
        self._depth_range_logged = True
        print(f"\n[evaluate_depth] Depth range: [{min_depth}, {max_depth}]m")
    
    # ========================================
    # STEP 4: Post-Processing 변환 (Linear & Log)
    # ========================================
    from packnet_sfm.utils.post_process_depth import (
        sigmoid_to_depth_linear,
        sigmoid_to_depth_log
    )
    
    # Linear transformation (기존 방식)
    depth_linear = sigmoid_to_depth_linear(sigmoid0, min_depth, max_depth)
    
    # Log transformation (INT8 최적화)
    depth_log = sigmoid_to_depth_log(sigmoid0, min_depth, max_depth)
    
    # ========================================
    # STEP 5: GT Depth 정규화
    # ========================================
    device = sigmoid0.device
    
    def _to_b1hw(x):
        """다양한 형태의 텐서를 (B,1,H,W) 형태로 정규화"""
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            return None
        x = x.to(device=device, dtype=torch.float32)
        
        if x.dim() == 0:
            return x.view(1, 1, 1, 1)
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.dim() == 3:
            if x.size(0) in (1, 3):
                x = x.unsqueeze(0)
                return x[:, :1, ...]
            else:
                return x.unsqueeze(1)
        if x.dim() == 4:
            if x.size(1) != 1:
                return x[:, :1, ...]
            return x
        return None
    
    depth_gt = _to_b1hw(batch.get('depth', None))
    
    # ========================================
    # STEP 6: 임시 스케일 보정 (환경변수)
    # ========================================
    if os.environ.get('FORCE_DEPTH_DIV256', '0') == '1':
        def _div256(x):
            if x is None:
                return x
            if torch.is_tensor(x) and x.max() > 255:
                return x / 256.0
            return x
        depth_gt = _div256(depth_gt)
    
    # ========================================
    # STEP 7: Metrics 계산 (4가지: Linear 2개 + Log 2개)
    # ========================================
    metrics = OrderedDict()
    
    if depth_gt is not None:
        # ========== Linear Metrics (2개) ==========
        # 7-1) depth_linear: GT scale 없음
        try:
            m_linear = compute_depth_metrics(
                self.config.model.params, 
                gt=depth_gt, 
                pred=depth_linear, 
                use_gt_scale=False
            )
        except Exception:
            m_linear = self._compute_depth_metrics_fallback(depth_gt, depth_linear)
        metrics['depth_linear'] = m_linear
        
        # 7-2) depth_linear_gt: GT scale 있음
        try:
            m_linear_gt = compute_depth_metrics(
                self.config.model.params, 
                gt=depth_gt, 
                pred=depth_linear, 
                use_gt_scale=True
            )
        except Exception:
            m_linear_gt = self._compute_depth_metrics_fallback(depth_gt, depth_linear)
        metrics['depth_linear_gt'] = m_linear_gt
        
        # ========== Log Metrics (2개) ==========
        # 7-3) depth_log: GT scale 없음
        try:
            m_log = compute_depth_metrics(
                self.config.model.params, 
                gt=depth_gt, 
                pred=depth_log, 
                use_gt_scale=False
            )
        except Exception:
            m_log = self._compute_depth_metrics_fallback(depth_gt, depth_log)
        metrics['depth_log'] = m_log
        
        # 7-4) depth_log_gt: GT scale 있음
        try:
            m_log_gt = compute_depth_metrics(
                self.config.model.params, 
                gt=depth_gt, 
                pred=depth_log, 
                use_gt_scale=True
            )
        except Exception:
            m_log_gt = self._compute_depth_metrics_fallback(depth_gt, depth_log)
        metrics['depth_log_gt'] = m_log_gt
    
    # ========================================
    # STEP 8: 결과 반환
    # ========================================
    return {
        'metrics': metrics,  # OrderedDict with 4 entries
        'inv_depth': sigmoid0,  # Sigmoid for visualization
        'depth_linear': depth_linear,  # For saving
        'depth_log': depth_log  # For saving
    }
    
    # ========================================
    # 🔧 OPTIONAL: Flip TTA 버전 (주석 처리)
    # ========================================
    # 필요 시 아래 코드를 활성화하여 TTA 적용 가능
    # 주의: 추론 시간 2배 증가!
    """
    # Flip TTA
    batch['rgb'] = flip_lr(batch['rgb'])
    sigmoid_outputs_flipped = self.model(batch)['inv_depths']
    sigmoid0_flipped_back = flip_lr(sigmoid_outputs_flipped[0])
    sigmoid_pp = post_process_inv_depth(sigmoid0, sigmoid0_flipped_back, method='mean')
    batch['rgb'] = flip_lr(batch['rgb'])
    
    # TTA 변환
    depth_linear_pp = sigmoid_to_depth_linear(sigmoid_pp, min_depth, max_depth)
    depth_log_pp = sigmoid_to_depth_log(sigmoid_pp, min_depth, max_depth)
    
    # TTA Metrics 추가 (4개 더)
    metrics['depth_linear_pp'] = compute_depth_metrics(..., depth_linear_pp, use_gt_scale=False)
    metrics['depth_linear_pp_gt'] = compute_depth_metrics(..., depth_linear_pp, use_gt_scale=True)
    metrics['depth_log_pp'] = compute_depth_metrics(..., depth_log_pp, use_gt_scale=False)
    metrics['depth_log_pp_gt'] = compute_depth_metrics(..., depth_log_pp, use_gt_scale=True)
    
    return {
        'metrics': metrics,  # 8 entries with TTA
        'inv_depth': sigmoid_pp,
        'depth_linear_pp': depth_linear_pp,
        'depth_log_pp': depth_log_pp
    }
    """
```

---

### 2.2 주요 변경점 요약

| 항목 | 기존 (현재) | 새로운 (변경 후) |
|------|------------|-----------------|
| **모델 출력** | `inv_depths` (실제로는 depth) | `sigmoid_outputs` (진짜 sigmoid) |
| **TTA 적용** | 항상 적용 (2배 시간) | ❌ 제거 (깔끔함) |
| **변환 위치** | 모델 내부 (`disp_to_inv`) | 평가 함수 (`sigmoid_to_depth_*`) ✅ |
| **변환 방식** | Linear만 | Linear + Log ✅ |
| **Metrics 수** | 4개 (TTA 포함) | **4개 (TTA 제외)** ✅ |
| **반환 값** | `inv_depth_pp` (depth) | `sigmoid0`, `depth_linear`, `depth_log` ✅ |

**TTA 관련**:
- 기본: TTA **제거** (추론 속도 2배 향상)
- 옵션: 필요 시 주석 해제하여 활성화 가능

---

## 3. Helper 함수들 상세 분석

### 3.1 inv2depth 함수

**위치**: `packnet_sfm/utils/depth.py` line 103

```python
def inv2depth(inv_depth):
    """
    Invert an inverse depth map to produce a depth map
    
    수식: depth = 1 / inv_depth
    
    입력: inv_depth (B,1,H,W) - Inverse depth
    출력: depth (B,1,H,W) - Depth
    
    ⚠️ 현재 문제:
    - 모델이 이미 depth를 반환하므로
    - inv2depth(depth) = 1/depth = inverse depth로 재변환
    - 혼란스러운 명명!
    """
    if is_seq(inv_depth):
        return [inv2depth(item) for item in inv_depth]
    else:
        return 1. / inv_depth.clamp(min=1e-6)
```

**새 구현에서는**:
```python
# inv2depth 사용하지 않음!
# 대신 sigmoid_to_depth_* 함수 사용
```

---

### 3.2 post_process_inv_depth 함수

**위치**: `packnet_sfm/utils/depth.py` line 229

**핵심 동작**:
```python
def post_process_inv_depth(inv_depth, inv_depth_flipped, method='mean'):
    """
    가장자리 smoothing + 중앙 평균
    
    동작:
    1. inv_depth_hat = flip(inv_depth_flipped)  # 좌표 복원
    2. inv_depth_fused = mean(inv_depth, inv_depth_hat)  # 평균
    3. Mask 생성:
       - xs < 0.05 (좌측 5%): mask = 1.0
       - xs > 0.95 (우측 5%): mask_hat = 1.0
       - 나머지: mask = mask_hat = 0.0
    4. 결합: mask_hat * inv + mask * inv_hat + (1-mask-mask_hat) * fused
    
    결과:
    - 좌측 5%: 원본 예측
    - 우측 5%: 반전 예측
    - 중앙 90%: 평균
    """
    B, C, H, W = inv_depth.shape
    
    # 반전 예측을 원래 좌표로
    inv_depth_hat = flip_lr(inv_depth_flipped)
    
    # 평균 계산
    inv_depth_fused = fuse_inv_depth(inv_depth, inv_depth_hat, method=method)
    
    # 좌측 가장자리 마스크 (0 ~ 0.05)
    xs = torch.linspace(0., 1., W, device=inv_depth.device,
                        dtype=inv_depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    # xs=0.0 → mask=1.0
    # xs=0.05 → mask=0.0
    # xs>0.05 → mask=0.0
    
    # 우측 가장자리 마스크 (0.95 ~ 1.0)
    mask_hat = flip_lr(mask)
    
    # 최종 결합
    return mask_hat * inv_depth + \
           mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused
```

**새 구현에서 사용**:
```python
# Sigmoid 공간에서 TTA 적용
sigmoid_pp = post_process_inv_depth(sigmoid0, sigmoid0_flipped_back, method='mean')

# 이후 Linear/Log 변환
depth_linear_pp = sigmoid_to_depth_linear(sigmoid_pp, ...)
depth_log_pp = sigmoid_to_depth_log(sigmoid_pp, ...)
```

---

### 3.3 compute_depth_metrics 함수

**위치**: `packnet_sfm/utils/depth.py` line 259

**시그니처**:
```python
def compute_depth_metrics(config, gt, pred, use_gt_scale=True):
    """
    입력:
    - config: CfgNode with min_depth, max_depth, crop, scale_output
    - gt: (B,1,H,W) GT depth
    - pred: (B,1,H,W) Predicted depth
    - use_gt_scale: bool - GT median scaling 여부
    
    출력:
    - metrics: torch.Tensor [7]
      [abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3]
    """
```

**동작 순서**:
1. GT/Pred 범위 필터링 (min_depth ~ max_depth)
2. Crop 적용 (config.crop)
3. GT median scaling (use_gt_scale=True일 때)
4. Valid mask 생성
5. Metrics 계산:
   - abs_rel: |GT - Pred| / GT (mean)
   - sqr_rel: (GT - Pred)² / GT (mean)
   - rmse: √((GT - Pred)²) (mean)
   - rmse_log: √((log(GT) - log(Pred))²) (mean)
   - a1: % pixels with max(GT/Pred, Pred/GT) < 1.25
   - a2: % pixels with max(GT/Pred, Pred/GT) < 1.25²
   - a3: % pixels with max(GT/Pred, Pred/GT) < 1.25³

---

## 4. Validation Step 흐름

### 4.1 전체 파이프라인

```
DataLoader 
    ↓
Batch: {
    'rgb': (B, 3, H, W),
    'depth': (B, 1, H, W),  # GT
    'idx': [...],
    'mask': (B, 1, H, W)  # Optional
}
    ↓
model_wrapper.validation_step(batch, batch_idx, dataset_idx)
    ↓
    ├─→ evaluate_depth(batch)
    │       ↓
    │   1. Model forward → sigmoid [0,1]
    │   2. Flip TTA (sigmoid space)
    │   3. Linear/Log transformation
    │   4. Metrics 계산 (8개)
    │       ↓
    │   Return: {
    │       'metrics': OrderedDict{
    │           'depth_linear': [7],
    │           'depth_linear_pp': [7],
    │           'depth_linear_gt': [7],
    │           'depth_linear_pp_gt': [7],
    │           'depth_log': [7],
    │           'depth_log_pp': [7],
    │           'depth_log_gt': [7],
    │           'depth_log_pp_gt': [7]
    │       },
    │       'inv_depth': sigmoid_pp,
    │       'depth_linear_pp': ...,
    │       'depth_log_pp': ...
    │   }
    │
    ├─→ Visualization (if loggers exist)
    │   - rgb_original
    │   - pred_inv_depth_masked
    │   - pred_inv_depth_unmasked
    │   - mask
    │
    └─→ Return: {
            'idx': batch['idx'],
            'depth_linear': [...],
            'depth_linear_pp': [...],
            ...
        }
    ↓
Trainer collects all validation outputs
    ↓
validation_epoch_end()
    ↓
Average metrics across all batches
    ↓
print_metrics() - 콘솔 출력
```

---

### 4.2 validation_step 함수

**위치**: `packnet_sfm/models/model_wrapper.py` line 337

```python
def validation_step(self, batch, batch_idx, dataset_idx):
    """
    Processes a validation batch.
    
    현재 동작:
    1. evaluate_depth 호출
    2. 시각화 (rgb, depth, mask)
    3. Metrics 반환
    """
    # 평가
    output = self.evaluate_depth(batch)
    
    # 시각화
    if self.loggers:
        rgb_original = batch['rgb'][0].cpu()
        
        # Sigmoid → 시각화 (colormap 적용)
        viz_pred_inv_depth = viz_inv_depth(output['inv_depth'][0])
        if isinstance(viz_pred_inv_depth, np.ndarray):
            viz_pred_inv_depth = torch.from_numpy(viz_pred_inv_depth).float()
        viz_pred_inv_depth = viz_pred_inv_depth.permute(2, 0, 1)
        
        # Mask 적용 (if exists)
        mask = None
        if 'mask' in batch and batch['mask'] is not None:
            mask = batch['mask'][0].cpu()
            if mask.dim() == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            viz_pred_inv_depth_masked = viz_pred_inv_depth * mask.unsqueeze(0).float()
        else:
            viz_pred_inv_depth_masked = viz_pred_inv_depth
        
        # Global step 계산
        total_batches_per_epoch = getattr(self, '_val_total_batches', 1000) or 1000
        global_step = self.current_epoch * total_batches_per_epoch + batch_idx
        
        # TensorBoard/W&B에 기록
        for logger in self.loggers:
            logger.writer.add_image('val/rgb_original', rgb_original, global_step=global_step)
            logger.writer.add_image('val/pred_inv_depth_masked', viz_pred_inv_depth_masked, global_step=global_step)
            logger.writer.add_image('val/pred_inv_depth_unmasked', viz_pred_inv_depth, global_step=global_step)
            if mask is not None:
                logger.writer.add_image('val/mask', mask.unsqueeze(0).float(), global_step=global_step)
    
    # Metrics 반환
    return {
        'idx': batch['idx'],
        **output['metrics'],  # 8개 metrics 모두 포함
    }
```

**변경 필요 없음**: 
- `output['metrics']`가 8개로 늘어나지만 자동으로 처리됨

---

## 5. 코드 변경 전후 비교

### 5.1 데이터 흐름 비교

#### 현재 (Before)

```
RGB Image
    ↓
ResNetSAN01.forward()
    ├─→ Encoder
    ├─→ Decoder → sigmoid [0,1]
    └─→ disp_to_inv(sigmoid) → depth [min, max]  ❌ 모델 내부 변환
    ↓
model_wrapper.evaluate_depth()
    ├─→ inv2depth(depth) = 1/depth  ❌ 혼란스러운 명명
    ├─→ Flip TTA (depth 공간)
    ├─→ Metrics 계산 (4개)
    └─→ Return
```

#### 새로운 (After)

```
RGB Image
    ↓
ResNetSAN01.forward()
    ├─→ Encoder
    ├─→ Decoder → sigmoid [0,1]
    └─→ Return sigmoid  ✅ 모델은 sigmoid만 반환
    ↓
model_wrapper.evaluate_depth()
    ├─→ sigmoid_to_depth_linear(sigmoid) → depth_linear  ✅ 명확한 변환
    ├─→ sigmoid_to_depth_log(sigmoid) → depth_log  ✅ INT8 최적화
    ├─→ Metrics 계산 (4개: Linear 2개 + Log 2개)
    └─→ Return
```

---

### 5.2 Metrics 구조 비교

#### 현재 (4개)

```python
metrics = OrderedDict({
    'depth': [abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3],
    'depth_pp': [...],
    'depth_gt': [...],
    'depth_pp_gt': [...]
})
```

#### 새로운 (4개, TTA 제외)

```python
metrics = OrderedDict({
    # Linear (2개)
    'depth_linear': [abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3],
    'depth_linear_gt': [...],
    
    # Log (2개)
    'depth_log': [abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3],
    'depth_log_gt': [...]
})
```

---

### 5.3 콘솔 출력 비교

#### 현재

```
|*************************************************************************************|
|                                   VALIDATION METRICS                                |
|*************************************************************************************|
|    Method      | abs_rel  | sqr_rel  |   rmse   | rmse_log |    a1    |    a2    |    a3    |
|*************************************************************************************|
| DEPTH          |  0.0329  |  0.0045  |  0.6627  |  0.0638  |  0.9846  |  0.9973  |  0.9991  |
| DEPTH_PP       |  0.0320  |  0.0043  |  0.6500  |  0.0625  |  0.9850  |  0.9975  |  0.9992  |
| DEPTH_GT       |  0.0312  |  0.0041  |  0.6400  |  0.0610  |  0.9860  |  0.9977  |  0.9993  |
| DEPTH_PP_GT    |  0.0305  |  0.0040  |  0.6300  |  0.0600  |  0.9870  |  0.9980  |  0.9994  |
|*************************************************************************************|
```

#### 새로운

```
|*************************************************************************************|
|                               LINEAR POST-PROCESSING                                |
|*************************************************************************************|
|    Method      | abs_rel  | sqr_rel  |   rmse   | rmse_log |    a1    |    a2    |    a3    |
|*************************************************************************************|
| DEPTH_LINEAR       |  0.0329  |  0.0045  |  0.6627  |  0.0638  |  0.9846  |  0.9973  |  0.9991  |
| DEPTH_LINEAR_GT    |  0.0312  |  0.0041  |  0.6400  |  0.0610  |  0.9860  |  0.9977  |  0.9993  |
|*************************************************************************************|
|                                LOG POST-PROCESSING                                  |
|*************************************************************************************|
|    Method      | abs_rel  | sqr_rel  |   rmse   | rmse_log |    a1    |    a2    |    a3    |
|*************************************************************************************|
| DEPTH_LOG          |  0.0330  |  0.0045  |  0.6650  |  0.0640  |  0.9845  |  0.9972  |  0.9991  |
| DEPTH_LOG_GT       |  0.0313  |  0.0042  |  0.6420  |  0.0612  |  0.9859  |  0.9976  |  0.9993  |
|*************************************************************************************|
```

---

## 6. 구현 시 주의사항

### 6.1 Import 추가

**파일**: `packnet_sfm/models/model_wrapper.py`

```python
# 기존 imports
from packnet_sfm.utils.depth import inv2depth, post_process_inv_depth, compute_depth_metrics, viz_inv_depth

# ✅ 새로 추가
from packnet_sfm.utils.post_process_depth import (
    sigmoid_to_depth_linear,
    sigmoid_to_depth_log
)
```

---

### 6.2 반환 값 활용

```python
# validation_step에서
output = self.evaluate_depth(batch)

# 사용 가능한 값들:
output['metrics']  # OrderedDict with 4 entries
output['inv_depth']  # sigmoid0 for visualization
output['depth_linear']  # Linear depth for saving
output['depth_log']  # Log depth for saving
```

---

### 6.3 시각화 수정 (선택)

**현재**: `viz_inv_depth(output['inv_depth'])`

**새로운 옵션**:
```python
# Option 1: Sigmoid 시각화 (기존과 동일)
viz_pred_inv_depth = viz_inv_depth(output['inv_depth'])

# Option 2: Linear depth 시각화
viz_pred_depth_linear = viz_depth(output['depth_linear_pp'])

# Option 3: Log depth 시각화
viz_pred_depth_log = viz_depth(output['depth_log_pp'])
```

---

## 7. 체크리스트

### 구현 전 확인

- [ ] `post_process_depth.py` 파일 생성 완료
- [ ] `sigmoid_to_depth_linear` 함수 테스트 완료
- [ ] `sigmoid_to_depth_log` 함수 테스트 완료

### evaluate_depth 수정

- [ ] Import 추가 (`sigmoid_to_depth_*`)
- [ ] STEP 2: 모델 출력 이름 변경 (`sigmoid_outputs`)
- [ ] STEP 3: Config에서 depth range 추출
- [ ] STEP 4: Linear/Log 변환 추가
- [ ] STEP 7: Metrics 4개로 구현 (TTA 제외)
- [ ] STEP 8: 반환 값 추가 (`depth_linear`, `depth_log`)
- [ ] ⚙️ Optional: TTA 코드 주석으로 추가

### print_metrics 수정

- [ ] Linear/Log 구분 출력
- [ ] 테이블 포맷 조정

### 테스트

- [ ] Unit test 통과
- [ ] Validation 실행 성공
- [ ] 콘솔 출력 확인 (8개 metrics)
- [ ] Linear/Log 비교 결과 확인

---

## 8. 예상 출력 예시

### 8.1 콘솔 로그

```bash
$ python scripts/train.py configs/eval_ncdb_640_val.yaml

[evaluate_depth] Depth range: [0.05, 80.0]m

Epoch 1/1: 100%|████████████| 91/91 [00:42<00:00,  2.16it/s]  ✅ TTA 제거로 2배 빠름!

|*************************************************************************************|
|                               LINEAR POST-PROCESSING                                |
|*************************************************************************************|
|    Method      | abs_rel  | sqr_rel  |   rmse   | rmse_log |    a1    |    a2    |    a3    |
|*************************************************************************************|
| DEPTH_LINEAR       |  0.0329  |  0.0045  |  0.6627  |  0.0638  |  0.9846  |  0.9973  |  0.9991  |
| DEPTH_LINEAR_GT    |  0.0312  |  0.0041  |  0.6400  |  0.0610  |  0.9860  |  0.9977  |  0.9993  |
|*************************************************************************************|
|                                LOG POST-PROCESSING                                  |
|*************************************************************************************|
|    Method      | abs_rel  | sqr_rel  |   rmse   | rmse_log |    a1    |    a2    |    a3    |
|*************************************************************************************|
| DEPTH_LOG          |  0.0330  |  0.0045  |  0.6650  |  0.0640  |  0.9845  |  0.9972  |  0.9991  |
| DEPTH_LOG_GT       |  0.0313  |  0.0042  |  0.6420  |  0.0612  |  0.9859  |  0.9976  |  0.9993  |
|*************************************************************************************|
```

---

### 8.2 Python Dict 형태

```python
output = {
    'metrics': OrderedDict({
        'depth_linear': tensor([0.0329, 0.0045, 0.6627, 0.0638, 0.9846, 0.9973, 0.9991]),
        'depth_linear_gt': tensor([0.0312, 0.0041, 0.6400, 0.0610, 0.9860, 0.9977, 0.9993]),
        'depth_log': tensor([0.0330, 0.0045, 0.6650, 0.0640, 0.9845, 0.9972, 0.9991]),
        'depth_log_gt': tensor([0.0313, 0.0042, 0.6420, 0.0612, 0.9859, 0.9976, 0.9993])
    }),
    'inv_depth': tensor([[[[0.123, 0.456, ...]]]], shape=(1,1,384,640)),  # sigmoid
    'depth_linear': tensor([[[[0.5, 1.2, ...]]]], shape=(1,1,384,640)),
    'depth_log': tensor([[[[0.52, 1.18, ...]]]], shape=(1,1,384,640))
}
```

---

## 9. 요약

### 핵심 변경사항

1. **모델 출력**: `depth` → `sigmoid [0,1]` ✅
2. **TTA 제거**: 추론 속도 2배 향상 ✅
3. **변환 위치**: 모델 내부 → 평가 함수 ✅
4. **변환 방식**: Linear만 → Linear + Log ✅
5. **Metrics 수**: 4개 (TTA 포함) → **4개 (깔끔함)** ✅

### 예상 효과

- **FP32**: Linear ≈ Log (거의 동일)
- **INT8**: Log >> Linear (13배 향상!)
- **유연성**: Post-Processing 분리로 다양한 변환 테스트 가능

### 다음 단계

1. 이 문서 검토
2. `evaluate_depth` 함수 수정
3. Validation 테스트
4. Linear vs Log 성능 비교

---

**버전**: 2.0  
**업데이트**: 2025-10-28  
**문서 상태**: 상세 분석 완료

