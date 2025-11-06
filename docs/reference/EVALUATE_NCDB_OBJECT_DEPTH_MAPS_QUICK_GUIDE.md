# 실제 사용 명령어 모음

## ⚠️ 중요: 예측 캐시 관리

평가 스크립트는 **체크포인트별로 자동으로 별도 폴더를 생성**합니다:
- `--pred-root newest_depth_maps_pred` 지정 시
- 실제 저장 위치: `newest_depth_maps_pred/{checkpoint_id}/`
- 예: `newest_depth_maps_pred/ResNet-SAN_0.5to100/`
- 예: `newest_depth_maps_pred/resnetsan01_640x384_newest_test_fixed_method_0.3_100_silog_1.0/`

이를 통해:
- ✅ 서로 다른 모델의 예측이 섞이지 않음
- ✅ 체크포인트 변경 시 자동으로 새 폴더 사용
- ✅ 이전 캐시 보존 (재평가 시 빠름)

**수동 캐시 삭제가 필요한 경우:**
```bash
# 특정 모델의 캐시만 삭제
rm -rf /workspace/data/ncdb-cls-640x384/*/synced_data/newest_depth_maps_pred/ResNet-SAN_0.5to100/

# 모든 캐시 삭제
find /workspace/data/ncdb-cls-640x384 -type f -name "*.npz" -path "*/newest_depth_maps_pred/*" -delete
```

---

## 1. 객체별 깊이 평가 (Object-Masked Depth Evaluation)

### 전체 데이터셋, 자동차만 평가

```bash
python scripts/evaluate_ncdb_object_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --use-all-splits --segmentation-root segmentation_results --pred-root newest_depth_maps_pred --gt-root newest_depth_maps --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt --image-shape 384 640 --classes car --output-file outputs/object_metrics_car.csv --per-instance-json outputs/object_metrics_car_instances.json
```

### Validation split, 모든 클래스, TTA 사용

```bash
python scripts/evaluate_ncdb_object_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --split-files combined_test.json --segmentation-root segmentation_results --pred-root newest_depth_maps_pred --gt-root newest_depth_maps --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt --image-shape 384 640 --flip-tta --use-gt-scale --output-file outputs/object_metrics_val_all.csv --per-instance-json outputs/object_metrics_val_all_instances.json --debug
```

### Test split, 자동차+트럭+사람

```bash
python scripts/evaluate_ncdb_object_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --split-files combined_test.json --segmentation-root segmentation_results --pred-root newest_depth_maps_pred --gt-root newest_depth_maps --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt --image-shape 384 640 --classes car truck person --use-gt-scale --output-file outputs/object_metrics_test_vehicles.csv
```

---

## 2. 전체 이미지 깊이 평가 (Full Image Depth Evaluation)

### On-the-fly 평가 (체크포인트만)

```bash
python scripts/evaluate_ncdb_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --split splits/combined_test.json --pred-root newest_depth_maps_pred --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt --resize_h 384 --resize_w 640 --output_file outputs/full_image_metrics.txt
```

### 저장된 예측 평가

```bash
python scripts/evaluate_ncdb_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --split splits/combined_test.json --pred_folder outputs/saved_depth_predictions --pred_ext png --use_gt_scale --min_depth 0.3 --max_depth 100 --output_file outputs/saved_pred_metrics.txt
```

### 비교 모드 (저장된 예측 vs 실시간)

```bash
python scripts/evaluate_ncdb_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --split splits/combined_test.json --pred_folder outputs/saved_depth_predictions --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt --resize_h 384 --resize_w 640 --use_gt_scale --flip_tta --output_file outputs/comparison_metrics.txt --per_sample_report outputs/per_sample.json
```

---

## 3. 학습 (Training)

### ResNetSAN 학습 (NCDB 640x384)

```bash
python scripts/train.py configs/train_resnet_san_ncdb_640x384.yaml
```

### Enhanced 모델 학습

```bash
python scripts/train.py configs/train_resnet_san_kitti_enhanced.yaml
```

---

## 4. 추론 (Inference)

### 단일 이미지 깊이 예측

```bash
python scripts/infer.py --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt --input /workspace/data/test_images/sample.png --output outputs/predicted_depth.png --image-shape 384 640
```

---

## 5. 유용한 유틸리티

### Split 파일 경로 업데이트

```bash
python scripts/update_split_paths.py --dataset-root /workspace/data/ncdb-cls-640x384 --split-files splits/combined_train.json splits/combined_val.json splits/combined_test.json --output-dir splits/updated
```

---

## 체크포인트 경로 단축 (Alias)

긴 체크포인트 경로를 매번 입력하기 번거로우므로 환경변수 사용:

```bash
export CHECKPOINT_NEWEST=/workspace/packnet-sfm/checkpoints/ResNet-SAN_0.05to100.ckpt
```

사용 예시:

```bash
python scripts/evaluate_ncdb_object_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --use-all-splits --segmentation-root segmentation_results --pred-root newest_depth_maps_pred --gt-root newest_depth_maps --checkpoint $CHECKPOINT_NEWEST --image-shape 384 640 --classes car --output-file outputs/metrics.csv
```

또는 심볼릭 링크 생성:

```bash
ln -s checkpoints/ResNet-SAN_0.05to100.ckpt checkpoints/latest.ckpt
```

---

## 자주 사용하는 조합

### 빠른 검증 (5분 이내)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --split-files combined_test.json --segmentation-root segmentation_results --pred-root newest_depth_maps_pred --gt-root newest_depth_maps --checkpoint checkpoints/latest.ckpt --image-shape 384 640 --classes car --output-file outputs/quick_check.csv
```

### 완전한 평가 (1-2시간)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --use-all-splits --segmentation-root segmentation_results --pred-root newest_depth_maps_pred --gt-root newest_depth_maps --checkpoint $CHECKPOINT_NEWEST --image-shape 384 640 --flip-tta --use-gt-scale --output-file outputs/full_evaluation.csv --per-instance-json outputs/full_instances.json --debug
```

### 논문용 최종 결과

```bash
python scripts/evaluate_ncdb_object_depth_maps.py --dataset-root /workspace/data/ncdb-cls-640x384 --split-files combined_test.json --segmentation-root segmentation_results --pred-root newest_depth_maps_pred_final --gt-root newest_depth_maps --checkpoint $CHECKPOINT_NEWEST --image-shape 384 640 --flip-tta --use-gt-scale --output-file outputs/paper_results.csv --per-instance-json outputs/paper_instances.json
```
