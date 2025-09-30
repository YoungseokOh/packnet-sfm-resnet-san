# 📝 문서 업데이트 완료

## 수정된 파일들

### 1. `/workspace/packnet-sfm/docs_md/evaluate_ncdb_object_depth_maps.md`
✅ 실제 경로로 업데이트
- 체크포인트 경로: `checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/...`
- 예측 캐시 폴더: `newest_depth_maps_pred`
- GT 폴더: `newest_depth_maps`

### 2. `/workspace/packnet-sfm/docs_md/evaluate_ncdb_object_depth_maps_quick.md`
✅ Quick Reference 실제 명령어로 업데이트

### 3. `/workspace/packnet-sfm/docs_md/실제_사용_명령어.md` (신규)
✅ 실전에서 바로 사용 가능한 명령어 모음집
- 객체별 깊이 평가
- 전체 이미지 평가
- 학습/추론 명령어
- 체크포인트 경로 단축 팁

---

## 실제 사용 명령어

### 자동차만 평가 (전체 split)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --use-all-splits \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_maps_pred \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt \
    --image-shape 384 640 \
    --classes car \
    --output-file outputs/object_metrics_car.csv \
    --per-instance-json outputs/object_metrics_car_instances.json
```

### 경로 단축 팁

```bash
# 환경변수 설정
export CHECKPOINT_NEWEST=/workspace/packnet-sfm/checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt

# 또는 심볼릭 링크
cd /workspace/packnet-sfm
ln -s checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt checkpoints/latest.ckpt

# 간단하게 사용
python scripts/evaluate_ncdb_object_depth_maps.py \
    --checkpoint checkpoints/latest.ckpt \
    # ... 나머지 인자
```

---

## 주요 변경사항

| 항목 | 이전 | 현재 |
|------|------|------|
| 예측 캐시 폴더 | `depth_predictions_cache` | `newest_depth_maps_pred` |
| 체크포인트 | `checkpoints/resnetsan01/...` | 실제 긴 경로 |
| 출력 파일 | `.txt` | `.csv` |
| 예시 개수 | 5개 | 5개 (실제 경로) |

---

## 사용 가능한 문서

1. **상세 가이드**: `docs_md/evaluate_ncdb_object_depth_maps.md`
   - 개요, 디렉토리 구조, 작동 원리
   - 실전 예시 5개
   - 문제 해결, 고급 사용법

2. **빠른 참조**: `docs_md/evaluate_ncdb_object_depth_maps_quick.md`
   - 기본 명령어
   - 필수 인자 요약

3. **실제 명령어 모음**: `docs_md/실제_사용_명령어.md`
   - 객체별/전체 평가
   - 학습/추론 명령어
   - 경로 단축 팁
   - 자주 사용하는 조합

---

## 다음 단계

1. 심볼릭 링크 생성 (권장):
```bash
cd /workspace/packnet-sfm
ln -s checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt checkpoints/latest.ckpt
```

2. 빠른 검증 실행:
```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --split-files combined_val.json \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_maps_pred \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/latest.ckpt \
    --image-shape 384 640 \
    --classes car \
    --output-file outputs/quick_test.csv
```

3. 결과 확인:
```bash
cat outputs/quick_test.csv
```
