# 🔧 Critical Bug Fix: Direct Depth Training Issue

## 🐛 문제점 발견

### Issue
Training 중 abs_rel = 0.553 (매우 높음, 거의 학습 안됨)

### Root Cause
**SemiSupCompletionModel이 모델 출력을 잘못 해석함!**

```python
# ❌ 문제 코드 (Before):
sigmoid_outputs = self_sup_output['inv_depths']  # Actually DEPTH, not sigmoid!
bounded_inv_depths = [
    sigmoid_to_inv_depth(sig, min_depth, max_depth)  # Treating depth as sigmoid!
    for sig in sigmoid_outputs
]
```

**무슨 일이 일어났나:**
1. ResNetSAN01 (depth_output_mode='direct')가 **depth 값 (예: 5.0m)**을 출력
2. SemiSupCompletionModel이 이것을 **sigmoid [0,1]**로 착각
3. `sigmoid_to_inv_depth(5.0, 0.5, 15.0)` 호출
   ```python
   # sigmoid_to_inv_depth assumes input is [0,1]
   # But receives 5.0 (depth in meters!)
   inv = 1/15.0 + (1/0.5 - 1/15.0) × 5.0  # WRONG!
   inv = 0.0667 + 1.9333 × 5.0 = 9.7332
   depth = 1 / 9.7332 = 0.103m  # Completely wrong!
   ```
4. GT depth = 5.0m, Predicted = 0.103m → **abs_rel = 4.897 (매우 큼!)**

### Mathematical Evidence

```
Example:
  Model output (direct depth): 7.5m
  SemiSupCompletionModel interprets as sigmoid: 7.5 (invalid, should be [0,1])
  
  sigmoid_to_inv_depth(7.5, 0.5, 15.0):
    inv_min = 1/15.0 = 0.0667
    inv_max = 1/0.5 = 2.0
    inv = 0.0667 + (2.0 - 0.0667) × 7.5 = 14.567
    depth = 1 / 14.567 = 0.069m  ← WRONG! (Should be 7.5m)
  
  GT depth: 7.5m
  Predicted (after wrong conversion): 0.069m
  abs_rel = |7.5 - 0.069| / 7.5 = 0.991  ← 99% error!
```

## ✅ 해결방법

### Fix Applied

```python
# ✅ 수정된 코드 (After):
depth_output_mode = getattr(self.depth_net, 'depth_output_mode', 'sigmoid')

if depth_output_mode == 'direct':
    # Model already outputs depth directly!
    depth_outputs = self_sup_output['inv_depths']  # Contains depth values
    
    # Pass depth DIRECTLY to loss (no conversion!)
    sup_output = self.supervised_loss(
        depth_outputs, batch['depth'],  # Both in depth space
        return_logs=return_logs, progress=progress)
else:
    # Legacy sigmoid mode
    sigmoid_outputs = self_sup_output['inv_depths']
    bounded_inv_depths = [
        sigmoid_to_inv_depth(sig, min_depth, max_depth)
        for sig in sigmoid_outputs
    ]
    sup_output = self.supervised_loss(
        bounded_inv_depths, depth2inv(batch['depth']),
        return_logs=return_logs, progress=progress)
```

### Changes Made

1. **SemiSupCompletionModel.forward()** (line ~460):
   - Check `depth_output_mode` attribute
   - If 'direct': Pass depth directly to loss
   - If 'sigmoid': Apply sigmoid→inv_depth conversion (legacy)

2. **Loss Input Mode**:
   - Direct mode: `loss(pred_depth, gt_depth)` with `input_mode='depth'`
   - Sigmoid mode: `loss(pred_inv, gt_inv)` with `input_mode='inv_depth'`

## 📊 Expected Results After Fix

### Before Fix
```
abs_rel: 0.553 (매우 높음)
δ<1.25: 0.006 (0.6% accurate)
```

### After Fix (Expected)
```
abs_rel: ~0.3-0.5 (untrained, epoch 0)
abs_rel: ~0.03-0.04 (converged, epoch 20+)
δ<1.25: ~0.95 (95% accurate when converged)
```

## 🔍 How to Verify

1. **Check loss decrease**:
   ```
   Epoch 0: Loss ~0.4-0.5 (should be reasonable, not 8.27)
   Epoch 5: Loss ~0.1-0.2
   Epoch 20: Loss ~0.01-0.02
   ```

2. **Check validation metrics**:
   ```
   Epoch 0: abs_rel ~0.3-0.5 (random init)
   Epoch 10: abs_rel ~0.1-0.15
   Epoch 20: abs_rel ~0.03-0.04
   ```

3. **Check depth range**:
   ```python
   pred_depth = model(rgb)['inv_depths'][0]
   print(f"Pred range: [{pred_depth.min():.2f}, {pred_depth.max():.2f}]m")
   # Should be close to [0.5, 15.0]m
   ```

## 🎯 Summary

**문제**: Model outputs depth, but training code treats it as sigmoid
**해결**: Check `depth_output_mode` and handle both modes correctly
**영향**: Training이 이제 제대로 작동하고 loss가 감소할 것

**다음 단계**: Training을 재시작하여 loss가 정상적으로 감소하는지 확인
