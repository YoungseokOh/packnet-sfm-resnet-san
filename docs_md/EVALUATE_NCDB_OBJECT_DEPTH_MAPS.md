# NCDB Object-Masked Depth Evaluation Guide

## 개요

`evaluate_ncdb_object_depth_maps.py`는 Mask2Former로 생성된 인스턴스 세그멘테이션 마스크를 활용하여 **객체별 깊이 예측 품질**을 정량 평가하는 스크립트입니다.

### 핵심 기능

1. **객체 마스크 기반 평가**: 전체 이미지가 아닌 특정 객체(자동차, 사람 등) 영역만 평가
2. **On-the-fly 추론**: 체크포인트로부터 직접 깊이 예측 수행
3. **캐싱 지원**: 한 번 추론한 깊이맵은 재사용 가능
4. **클래스별 집계**: 각 객체 클래스별 메트릭 평균 제공
5. **인스턴스 레벨 분석**: 개별 객체 인스턴스마다 세부 메트릭 저장 가능

---

## 디렉토리 구조

```
/workspace/data/ncdb-cls-640x384/
├── splits/
│   ├── combined_train.json
│   ├── combined_val.json
│   └── combined_test.json
├── synced_data/
│   ├── image_a6/                         # RGB 입력 이미지
│   │   └── 20230101_120000.png
│   ├── newest_depth_maps/                # GT 깊이맵
│   │   └── 20230101_120000.png
│   ├── segmentation_results/             # 세그멘테이션 결과
│   │   └── class_masks/                  # 클래스별 마스크
│   │       ├── car/
│   │       │   ├── 20230101_120000_0.png    # 첫 번째 car 인스턴스
│   │       │   └── 20230101_120000_1.png    # 두 번째 car 인스턴스
│   │       ├── person/
│   │       │   └── 20230101_120000_0.png
│   │       └── truck/
│   └── newest_depth_maps_pred/           # 캐시된 예측 (자동 생성)
│       └── 20230101_120000.npz
└── sequence_xxx/
    └── synced_data/
        └── (동일 구조)
```

---

## 사용 방법

### 기본 명령어 (실제 경로 예시)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --split-files combined_val.json \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_maps_pred \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt \
    --image-shape 384 640 \
    --flip-tta \
    --use-gt-scale \
    --output-file outputs/object_metrics.csv
```

### 간단 버전 (상대 경로)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --split-files combined_val.json \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_maps_pred \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/resnetsan01/resnetsan01.ckpt \
    --image-shape 384 640 \
    --flip-tta \
    --use-gt-scale \
    --output-file outputs/metrics_object_masks.txt
```

### 필수 인자

| 인자 | 설명 | 예시 |
|------|------|------|
| `--dataset-root` | 데이터셋 최상위 폴더 | `/workspace/data/ncdb-cls-640x384` |
| `--segmentation-root` | 세그멘테이션 결과 폴더 | `segmentation_results` |
| `--pred-root` | 예측 깊이맵 저장/로드 폴더 | `depth_predictions_cache` |
| `--gt-root` | GT 깊이맵 폴더 | `newest_depth_maps` |
| `--checkpoint` | 모델 체크포인트 경로 | `checkpoints/model.ckpt` |

### Split 선택

**옵션 1: 특정 split 파일 지정**
```bash
--split-files combined_val.json combined_test.json
```

**옵션 2: 모든 split 사용**
```bash
--use-all-splits  # train/val/test 모두
```

### 선택 인자

#### 모델 설정
```bash
--image-shape 384 640           # 모델 입력 크기 (H W)
--flip-tta                      # 좌우 플립 TTA 사용 (정확도 향상)
--device cuda:0                 # GPU 디바이스 지정
--dtype fp16                    # FP16 또는 FP32
```

#### 평가 설정
```bash
--use-gt-scale                  # GT median scaling 적용
--min-depth 0.3                 # 최소 평가 깊이 (m)
--max-depth 100.0               # 최대 평가 깊이 (m)
--crop garg                     # Crop 방식 ('' 또는 'garg')
--scale-output top-center       # 예측→GT 해상도 조정 방식
```

#### 출력 설정
```bash
--output-file metrics.txt                     # 요약 메트릭 저장
--per-instance-json instances.json            # 인스턴스별 세부 메트릭
--debug                                        # 상세 로그 출력
```

#### 클래스 필터링
```bash
--classes car truck person      # 특정 클래스만 평가 (미지정 시 전체)
```

---

## 실전 예시

### 1. 빠른 검증 (validation split, 특정 클래스만)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --split-files combined_val.json \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_maps_pred \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt \
    --image-shape 384 640 \
    --classes car \
    --use-gt-scale \
    --output-file outputs/quick_eval_car.csv
```

### 2. 전체 데이터셋 평가 (train+val+test, 모든 클래스)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --use-all-splits \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_maps_pred \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt \
    --image-shape 384 640 \
    --flip-tta \
    --use-gt-scale \
    --output-file outputs/full_eval_all_classes.csv \
    --per-instance-json outputs/full_instances.json
```

### 3. 특정 클래스들만 평가 (자동차, 트럭, 사람)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --use-all-splits \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_maps_pred \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt \
    --image-shape 384 640 \
    --classes car truck person \
    --use-gt-scale \
    --output-file outputs/vehicles_and_persons.csv
```

### 4. 정밀 평가 (test split only, TTA, 상세 로그)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --split-files combined_test.json \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_maps_pred \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt \
    --image-shape 384 640 \
    --flip-tta \
    --use-gt-scale \
    --output-file outputs/test_eval_detailed.csv \
    --per-instance-json outputs/test_instances.json \
    --debug
```

### 5. 다른 모델과 비교 (예측 캐시 별도 관리)

```bash
# 모델 A (newest_test)
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --split-files combined_test.json \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_pred_modelA \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/default_config-train_resnet_san_ncdb_640x384-2025.10.01-02h29m07s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt \
    --image-shape 384 640 \
    --flip-tta \
    --use-gt-scale \
    --output-file outputs/modelA_objects.csv

# 모델 B (enhanced)
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --split-files combined_test.json \
    --segmentation-root segmentation_results \
    --pred-root newest_depth_pred_modelB \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/resnetsan01_enhanced/model.ckpt \
    --image-shape 384 640 \
    --flip-tta \
    --use-gt-scale \
    --output-file outputs/modelB_objects.csv
```

---

## 출력 형식

### 콘솔 출력

```
Evaluating: 100%|████████████████████| 1000/1000 [05:23<00:00, 3.09it/s]

평가 요약 (객체 마스크 기준)
Class     Count  abs_rel  sqr_rel  rmse     rmse_log  a1      a2      a3
-----------------------------------------------------------------------------------
car       1523   0.0842   0.1156   0.4523   0.1123    0.9234  0.9756  0.9912
person    456    0.1234   0.2345   0.6234   0.1534    0.8756  0.9456  0.9834
truck     234    0.0956   0.1456   0.5123   0.1256    0.9123  0.9678  0.9889
ALL       2213   0.0923   0.1345   0.5012   0.1234    0.9123  0.9678  0.9878
```

### 텍스트 파일 (`metrics_object_masks.txt`)

```csv
Class,Count,abs_rel,sqr_rel,rmse,rmse_log,a1,a2,a3
car,1523,0.084200,0.115600,0.452300,0.112300,0.923400,0.975600,0.991200
person,456,0.123400,0.234500,0.623400,0.153400,0.875600,0.945600,0.983400
truck,234,0.095600,0.145600,0.512300,0.125600,0.912300,0.967800,0.988900
ALL,2213,0.092300,0.134500,0.501200,0.123400,0.912300,0.967800,0.987800
```

### JSON 파일 (`instances.json`)

```json
{
  "metric_names": ["abs_rel", "sqr_rel", "rmse", "rmse_log", "a1", "a2", "a3"],
  "instances": [
    {
      "stem": "20230101_120000",
      "class": "car",
      "mask_path": "/workspace/data/.../class_masks/car/20230101_120000_0.png",
      "valid_pixels": 12345,
      "metrics": [0.0842, 0.1156, 0.4523, 0.1123, 0.9234, 0.9756, 0.9912]
    },
    {
      "stem": "20230101_120000",
      "class": "car",
      "mask_path": "/workspace/data/.../class_masks/car/20230101_120000_1.png",
      "valid_pixels": 8765,
      "metrics": [0.0756, 0.1023, 0.4234, 0.1056, 0.9345, 0.9823, 0.9934]
    }
  ]
}
```

---

## 작동 원리

### 1. 초기화 단계

```python
# Split 파일 로드
samples = load_split_entries(split_paths)

# 모델 준비
model_context = prepare_model(checkpoint)

# 클래스 자동 탐지 (첫 샘플의 segmentation_root에서)
detected_classes = ["car", "person", "truck", ...]
```

### 2. 샘플별 처리

```python
for sample in samples:
    # 1. GT 깊이맵 로드
    gt_data = load_depth(sample.gt_path)
    
    # 2. 예측 깊이맵 준비
    if cache_exists(sample.prediction_path):
        prediction = load_prediction(sample.prediction_path)
    else:
        # On-the-fly 추론
        prediction = run_inference(model, sample.image_path, flip_tta)
        save_prediction(sample.prediction_path, prediction)
    
    # 3. 객체 마스크 수집
    mask_groups = collect_masks_for_stem(sample.stem)
    # {
    #   "car": ["20230101_120000_0.png", "20230101_120000_1.png"],
    #   "person": ["20230101_120000_0.png"]
    # }
    
    # 4. 각 객체 인스턴스 평가
    for class_name, mask_paths in mask_groups.items():
        for mask_path in mask_paths:
            mask = load_mask(mask_path, gt_data.shape)
            
            # 마스크 영역만 평가
            gt_masked = gt_data * mask
            metrics = compute_depth_metrics(gt_masked, prediction)
            
            # 클래스별 누산
            class_accumulators[class_name].add(metrics)
            overall_accumulator.add(metrics)
```

### 3. 집계 및 출력

```python
# 클래스별 평균 계산
class_metrics = {
    class_name: accumulator.mean()
    for class_name, accumulator in class_accumulators.items()
}

# 전체 평균
overall_metrics = overall_accumulator.mean()

# 출력
print_summary_table(class_metrics, overall_metrics)
save_to_file(output_file)
```

---

## 주요 기능 상세

### On-the-fly 추론

- **캐시 체크**: `pred-root/<stem>.npz` 존재 여부 확인
- **추론 수행**: 캐시 없으면 모델로 예측
- **자동 저장**: 추론 결과를 `.npz` 형식으로 자동 저장
- **재사용**: 다음 실행 시 캐시 재활용으로 시간 절약

```python
# 캐시 경로 예시
newest_depth_maps_pred/
├── 20230101_120000.npz  # 384x640 float32 array
├── 20230101_120100.npz
└── ...
```

### Flip TTA (Test-Time Augmentation)

`--flip-tta` 옵션 사용 시:

1. 원본 이미지로 예측
2. 좌우 반전 이미지로 예측
3. 두 예측 평균 (post_process_inv_depth)

→ **정확도 향상** (약 1-2% abs_rel 개선), **속도 2배 느림**

### GT Median Scaling

`--use-gt-scale` 옵션:

- GT와 예측의 median 값으로 스케일 조정
- 절대 깊이 값이 아닌 상대적 정확도 평가
- 일반적으로 더 공정한 비교

### 클래스 자동 탐지

`--classes` 미지정 시:

1. 첫 샘플의 `segmentation_root/class_masks/` 디렉토리 탐색
2. 서브디렉토리 이름을 클래스로 인식
3. 모든 샘플에 대해 해당 클래스들만 평가

```
class_masks/
├── car/          → "car" 클래스로 인식
├── person/       → "person" 클래스로 인식
└── truck/        → "truck" 클래스로 인식
```

---

## 메트릭 설명

| 메트릭 | 설명 | 단위 | 낮을수록 좋음/높을수록 좋음 |
|--------|------|------|---------------------------|
| `abs_rel` | Absolute Relative Error | - | ↓ |
| `sqr_rel` | Squared Relative Error | - | ↓ |
| `rmse` | Root Mean Squared Error | m | ↓ |
| `rmse_log` | RMSE in log space | - | ↓ |
| `a1` | δ < 1.25 (threshold accuracy) | % | ↑ |
| `a2` | δ < 1.25² | % | ↑ |
| `a3` | δ < 1.25³ | % | ↑ |

**좋은 성능 기준** (일반적):
- `abs_rel` < 0.10
- `a1` > 0.90
- `rmse` < 0.50m (가까운 거리 기준)

---

## 문제 해결

### 1. 세그멘테이션 마스크가 없음

```
FileNotFoundError: class_masks 디렉토리를 찾을 수 없습니다
```

**해결**:
- `--segmentation-root` 경로 확인
- Mask2Former 세그멘테이션 먼저 수행 필요

### 2. GT 깊이맵 해상도 불일치

```
ValueError: GT 해상도가 일관되지 않습니다
```

**해결**:
- 모든 GT 깊이맵이 동일 해상도인지 확인
- `--gt-root` 경로에 다른 해상도 파일이 섞여있지 않은지 체크

### 3. 예측 해상도 불일치 경고

```
예측/GT 해상도 불일치: pred (384, 640), gt (1536, 1920)
```

**정상 동작**: `compute_depth_metrics`가 자동으로 resize 수행
- `--scale-output` 옵션으로 조정 방식 변경 가능

### 4. 메모리 부족

**해결**:
- `--dtype fp16` 사용
- Split을 나눠서 평가
- 배치 크기 조정 (현재는 샘플별 처리)

### 5. 느린 추론 속도

**해결**:
- `--flip-tta` 제거 (2배 빨라짐)
- 첫 실행 후 캐시 활용
- GPU 사용 확인 (`--device cuda:0`)

---

## 고급 사용법

### 다중 데이터셋 평가

```bash
# 640x384 데이터셋
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-640x384 \
    --split-files combined_test.json \
    --segmentation-root segmentation_results \
    --pred-root depth_pred_640 \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/model_640.ckpt \
    --image-shape 384 640 \
    --use-gt-scale \
    --output-file outputs/eval_640.txt

# 1920x1536 데이터셋
python scripts/evaluate_ncdb_object_depth_maps.py \
    --dataset-root /workspace/data/ncdb-cls-1920x1536 \
    --split-files combined_test.json \
    --segmentation-root segmentation_results \
    --pred-root depth_pred_1920 \
    --gt-root newest_depth_maps \
    --checkpoint checkpoints/model_1920.ckpt \
    --image-shape 1536 1920 \
    --use-gt-scale \
    --output-file outputs/eval_1920.txt
```

### 배치 평가 스크립트

```bash
#!/bin/bash
# evaluate_all_models.sh

MODELS=(
    "checkpoints/resnetsan01/resnetsan01.ckpt"
    "checkpoints/resnetsan01_enhanced/resnetsan01_enhanced.ckpt"
    "checkpoints/packnetsan01/packnetsan01.ckpt"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename $(dirname $MODEL))
    echo "Evaluating $MODEL_NAME..."
    
    python scripts/evaluate_ncdb_object_depth_maps.py \
        --dataset-root /workspace/data/ncdb-cls-640x384 \
        --split-files combined_test.json \
        --segmentation-root segmentation_results \
        --pred-root "depth_pred_${MODEL_NAME}" \
        --gt-root newest_depth_maps \
        --checkpoint "$MODEL" \
        --image-shape 384 640 \
        --flip-tta \
        --use-gt-scale \
        --output-file "outputs/${MODEL_NAME}_objects.txt" \
        --per-instance-json "outputs/${MODEL_NAME}_instances.json"
done

echo "All evaluations complete!"
```

### 결과 비교 스크립트

```python
#!/usr/bin/env python3
"""Compare object evaluation results from multiple models."""

import sys
from pathlib import Path

def parse_results(file_path):
    results = {}
    with open(file_path) as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                class_name = parts[0]
                abs_rel = float(parts[2])
                results[class_name] = abs_rel
    return results

model_files = [
    ("ResNetSAN", "outputs/resnetsan01_objects.txt"),
    ("ResNetSAN+Enhanced", "outputs/resnetsan01_enhanced_objects.txt"),
    ("PackNetSAN", "outputs/packnetsan01_objects.txt"),
]

print("Model Comparison (abs_rel by class)")
print("=" * 80)

all_classes = set()
model_results = {}

for model_name, file_path in model_files:
    results = parse_results(file_path)
    model_results[model_name] = results
    all_classes.update(results.keys())

# Print header
print(f"{'Class':<15}", end="")
for model_name, _ in model_files:
    print(f"{model_name:>20}", end="")
print()
print("-" * 80)

# Print results
for class_name in sorted(all_classes):
    print(f"{class_name:<15}", end="")
    for model_name, _ in model_files:
        value = model_results[model_name].get(class_name, float('nan'))
        print(f"{value:>20.4f}", end="")
    print()
```

---

## 성능 최적화 팁

### 1. 캐시 활용
첫 실행 후 `--pred-root` 폴더 보존하여 재실행 시 추론 생략

### 2. 병렬 처리
여러 GPU가 있다면 split을 나눠서 병렬 실행:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_ncdb_object_depth_maps.py \
    --split-files split_0.json --device cuda:0 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_ncdb_object_depth_maps.py \
    --split-files split_1.json --device cuda:0 &

wait
```

### 3. FP16 사용
`--dtype fp16` 으로 메모리 절약 및 속도 향상 (약 2배)

---

## 참고 자료

- [evaluate_ncdb_depth_maps.md](./evaluate_ncdb_depth_maps.md) - 전체 이미지 평가
- [NCDB Dataset](../README.md) - 데이터셋 구조
- [Mask2Former Segmentation](./mask2former_segmentation.md) - 세그멘테이션 준비

---

## 요약

이 스크립트는:
- ✅ **객체별** 깊이 예측 품질 평가
- ✅ **On-the-fly** 추론으로 유연성 제공
- ✅ **캐싱**으로 재실행 효율성
- ✅ **클래스별 집계**로 심층 분석 가능
- ✅ **인스턴스 레벨** 세부 정보 저장

전체 이미지 평가와 달리, 이 스크립트는 **실제 관심 객체**의 깊이 정확도를 측정하여 더 의미 있는 평가를 제공합니다.
