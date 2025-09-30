# Evaluation Scripts Fix (2025-10-15)

## 발견된 문제점

### 1. Object Mask 평가의 치명적 버그 ⚠️

**위치**: `scripts/evaluate_ncdb_object_depth_maps.py`

**문제**:
```python
# ❌ 잘못된 코드 (이전)
gt_masked = torch.tensor(gt_data * mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
metrics = compute_depth_metrics(eval_namespace, gt_masked, pred_tensor, use_gt_scale=args.use_gt_scale)
```

- GT depth에만 마스크를 곱하고, **Pred depth는 전체 이미지를 사용**
- 결과적으로 **완전히 다른 픽셀들을 비교**하게 되어 메트릭이 의미 없음
- GT는 마스크 영역만 유효한 값, Pred는 전체 영역에 값이 있어 불일치

**해결**:
```python
# ✅ 올바른 코드 (수정 후)
# GT와 Pred 모두 마스크를 적용하여 동일한 영역만 비교
gt_masked_full = gt_data * mask
pred_masked_full = prediction * mask

gt_masked_tensor = torch.tensor(gt_masked_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
pred_masked_tensor = torch.tensor(pred_masked_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

metrics = compute_depth_metrics(eval_namespace, gt_masked_tensor, pred_masked_tensor, use_gt_scale=args.use_gt_scale)
```

**영향**:
- 이전 모든 객체 마스크 평가 결과가 **부정확함**
- 특히 작은 객체일수록 영향이 큼 (마스크 영역이 작을수록 전체 이미지와의 차이가 커짐)
- Full image 평가는 정상 작동하여 그 결과는 신뢰 가능

### 2. Full Image 시각화 문제

**위치**: `scripts/evaluate_ncdb_full_image.py`

**문제**:
```python
# ❌ GT/Pred depth 컬러맵이 제대로 보이지 않음
im1 = axes[0, 1].imshow(gt_display, cmap='viridis', vmin=min_depth, vmax=max_depth)
```

- `min_depth=0.05`, `max_depth=100.0`으로 고정
- 실제 데이터 범위와 맞지 않아 대부분 픽셀이 어둡거나 밝게 뭉개짐
- 시각적으로 depth 분포를 파악하기 어려움

**해결**:
```python
# ✅ 99 percentile을 사용하여 outlier 제거한 범위 설정
gt_valid_values = gt_depth[valid_mask]
if len(gt_valid_values) > 0:
    gt_vmax = np.percentile(gt_valid_values, 99)
else:
    gt_vmax = max_depth

im1 = axes[0, 1].imshow(gt_display, cmap='viridis', vmin=0, vmax=gt_vmax)
```

- 각 이미지의 실제 depth 분포에 맞게 동적으로 범위 설정
- 99 percentile을 사용하여 극단적인 outlier 제외
- GT와 Pred가 동일한 범위를 사용하여 직접 비교 가능

## 수정 결과 비교

### Object Mask 평가 (3개 샘플, car 클래스)

**수정 후** (정확함):
```
Class  Count  abs_rel  sqr_rel  rmse    rmse_log  a1      a2      a3    
------------------------------------------------------------------------
car    3      0.0328   0.0100   0.1999  0.0489    0.9935  0.9991  0.9995
ALL    3      0.0328   0.0100   0.1999  0.0489    0.9935  0.9991  0.9995
```

거리별 평가:
```
Range        Pixels  abs_rel  sqr_rel  rmse    rmse_log  a1      a2      a3    
-------------------------------------------------------------------------------
D < 1m       304     0.0496   0.0016   0.0267  0.0795    0.9901  0.9967  0.9967
1m < D < 2m  70      0.0371   0.0024   0.0534  0.0465    1.0000  1.0000  1.0000
2m < D < 3m  29      0.0162   0.0011   0.0539  0.0202    1.0000  1.0000  1.0000
D > 3m       1173    0.0394   0.0185   0.2978  0.0591    0.9898  0.9991  1.0000
```

### Full Image 평가 (3개 샘플)

**수정 전/후 동일** (이미 정상이었음):
```
Samples: 3
--------------------------------------------------------------------------------
abs_rel     : 0.0260
sqr_rel     : 0.0221
rmse        : 0.5838
rmse_log    : 0.0544
a1          : 0.9900
a2          : 0.9988
a3          : 0.9996
```

## 시각화 개선

### Object Mask 시각화
- 4-panel 레이아웃:
  1. RGB + Mask Overlay (초록색)
  2. GT Depth (마스크 영역만, 99 percentile 범위)
  3. Pred Depth (마스크 영역만, GT와 동일 범위)
  4. Error Heatmap (Green→Yellow→Orange→Red gradient)
- 에러 분포 통계 포함

### Full Image 시각화
- 4-panel 레이아웃:
  1. RGB Image
  2. GT Depth (99 percentile 범위) ← **수정됨**
  3. Pred Depth (GT와 동일 범위) ← **수정됨**
  4. Error Heatmap + 에러 분포 통계
- GT/Pred depth 컬러맵이 명확하게 표시됨

## 권장 사항

1. **기존 객체 마스크 평가 결과 재평가 필요**
   - 모든 이전 결과가 부정확할 수 있음
   - 수정된 스크립트로 재평가 권장

2. **Full image 평가 결과는 신뢰 가능**
   - 계산 로직은 정상이었음
   - 시각화만 개선됨

3. **향후 평가 시 주의사항**
   - Object mask 평가: GT와 Pred 모두 동일한 마스크 적용 확인
   - 시각화: 각 이미지의 실제 데이터 범위 고려
   - 디버그 모드로 중간 값 확인 권장

## 수정 파일

1. `scripts/evaluate_ncdb_object_depth_maps.py`
   - Line ~860: 인스턴스별 메트릭 계산 (GT/Pred 마스크 적용)
   - Line ~910: 클래스별 통합 메트릭 계산 (GT/Pred 마스크 적용)

2. `scripts/evaluate_ncdb_full_image.py`
   - Line ~300: GT/Pred depth 시각화 범위 설정 (99 percentile)

## 테스트 명령어

### Object Mask 평가 (수정 후)
```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
  --dataset-root /workspace/data/ncdb-cls-640x384 \
  --split-files combined_test.json \
  --segmentation-root segmentation_results \
  --pred-root newest_depth_maps_pred \
  --gt-root newest_depth_maps \
  --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt \
  --image-shape 384 640 \
  --classes car \
  --max-samples 3 \
  --visualize-dir outputs/object_viz_fixed
```

### Full Image 평가 (수정 후)
```bash
python scripts/evaluate_ncdb_full_image.py \
  --dataset-root /workspace/data/ncdb-cls-640x384 \
  --split-files combined_test.json \
  --pred-root newest_depth_maps_pred \
  --gt-root newest_depth_maps \
  --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt \
  --image-shape 384 640 \
  --max-samples 3 \
  --visualize-dir outputs/full_viz_fixed
```
