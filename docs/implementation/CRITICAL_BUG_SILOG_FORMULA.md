# 🐛 CRITICAL BUG: Silog Loss Formula Error

## 문제 발견

**증상**: Loss가 1.42~1.45에서 멈춤, 학습이 전혀 진행되지 않음

**원인**: Silog Loss 공식이 잘못 구현됨

## 잘못된 코드 (Before)

### `ssi_silog_loss.py` Line 124-132
```python
# ❌ WRONG: Multiplies by ratio at the end
log_pred = torch.log(pred_depth_masked * self.silog_ratio)  # * 10
log_gt = torch.log(gt_depth_masked * self.silog_ratio)      # * 10
log_diff = log_pred - log_gt
silog1 = torch.mean(log_diff ** 2)
silog2 = self.silog_ratio2 * (log_diff.mean() ** 2)
silog_var = silog1 - silog2
silog_loss = torch.sqrt(silog_var + 1e-8) * self.silog_ratio  # ❌ * 10 AGAIN!
```

### `supervised_loss.py` Line 72-79
```python
# ❌ WRONG: Same issue
log_diff = torch.log(pred * self.ratio) - torch.log(gt * self.ratio)
silog1 = torch.mean(log_diff ** 2)
silog2 = self.ratio2 * (log_diff.mean() ** 2)
silog_loss = torch.sqrt(silog1 - silog2) * self.ratio  # ❌ * 10!
```

## 수학적 분석

### 원본 Silog 공식 (논문)
```
Silog = sqrt(E[d^2] - λ * E[d]^2)
where d = log(pred) - log(gt)
      λ = 0.85
```

### 잘못된 구현의 문제
```python
# Step 1: log(pred * 10) - log(gt * 10)
#       = log(pred) + log(10) - log(gt) - log(10)
#       = log(pred) - log(gt)  ← 이 부분은 OK (log 특성상 상쇄)

# Step 2: sqrt(E[d^2] - λ * E[d]^2) * 10  ← ❌ 문제!
#       Loss가 10배로 증폭됨!
```

**결과**: 
- Silog Loss ≈ 0.1~0.2 → **1.0~2.0**으로 증폭
- SSI Loss weight = 0.7, Silog weight = 0.3이므로
- Total Loss = 0.7 * SSI + 0.3 * (Silog * 10)
- Silog component가 지배적이 되어 학습 불안정

## 올바른 코드 (After)

### `ssi_silog_loss.py` (Fixed)
```python
# ✅ CORRECT: No multiplicative scaling
log_pred = torch.log(pred_depth_masked)
log_gt = torch.log(gt_depth_masked)
log_diff = log_pred - log_gt
silog1 = torch.mean(log_diff ** 2)
silog2 = self.silog_ratio2 * (log_diff.mean() ** 2)
silog_var = silog1 - silog2
silog_loss = torch.sqrt(silog_var + 1e-8)  # ✅ No * ratio!
```

### `supervised_loss.py` (Fixed)
```python
# ✅ CORRECT
log_diff = torch.log(pred) - torch.log(gt)
silog1 = torch.mean(log_diff ** 2)
silog2 = self.ratio2 * (log_diff.mean() ** 2)
silog_loss = torch.sqrt(silog1 - silog2)  # ✅ No * ratio!
```

## 예상 효과

### Before (잘못됨):
- Loss: 1.42~1.45 (멈춤)
- Silog component: ~1.0~1.5
- Gradient: 불안정 (너무 큼)

### After (올바름):
- Loss: ~0.15~0.25 예상
- Silog component: ~0.1~0.2
- Gradient: 안정적
- 학습 정상 진행 예상

## 참고 문헌

Original Silog Loss paper:
```
Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
Loss = sqrt(1/n * Σ(log d_i)^2 - λ/n^2 * (Σ log d_i)^2)
where d_i = log(pred_i) - log(gt_i)
```

## 수정 일시

- 2025.10.28
- Files: `ssi_silog_loss.py`, `supervised_loss.py`
- Reason: Loss가 1.42~1.45에서 멈춰서 분석 후 발견
