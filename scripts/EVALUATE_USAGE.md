# NCDB Object Depth Evaluation - 사용 가이드

## 📋 개요

이 스크립트는 NCDB 데이터셋에서 객체 마스크 기반 depth 평가를 수행하고, 모든 결과를 체계적으로 저장합니다.

## 🎯 주요 기능

1. **자동 출력 구조**: `output/{checkpoint_id}_results/` 형태로 자동 생성
2. **5가지 출력 카테고리**: RGB, GT, Pred, Viz, Metrics
3. **완전한 재현성**: README.txt에 모든 실행 설정 기록

## 📁 출력 구조

```
output/
└── {checkpoint_id}_results/
    ├── rgb/              # 원본 RGB 이미지 (--save-rgb 시)
    ├── gt/               # Ground Truth depth (--save-gt 시)
    ├── pred/             # 예측 depth (16-bit PNG, --save-pred 시)
    ├── viz/              # 4-panel 시각화 (항상 저장)
    ├── metrics/          # 평가 메트릭
    │   ├── summary.csv         # 클래스별 요약
    │   └── per_instance.json   # 인스턴스별 상세
    └── README.txt        # 실행 정보
```

## 🚀 사용 방법

### 기본 사용 (시각화만)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
  --dataset-root /workspace/data/ncdb-cls-640x384 \
  --use-all-splits \
  --segmentation-root segmentation_results \
  --pred-root predictions \
  --gt-root newest_depth_maps \
  --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt \
  --output-root output
```

### 모든 파일 저장 (RGB + GT + Pred)

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
  --dataset-root /workspace/data/ncdb-cls-640x384 \
  --use-all-splits \
  --segmentation-root segmentation_results \
  --pred-root predictions \
  --gt-root newest_depth_maps \
  --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt \
  --save-rgb \
  --save-gt \
  --save-pred \
  --output-root output
```

### 1개 샘플 테스트

```bash
python scripts/evaluate_ncdb_object_depth_maps.py \
  --dataset-root /workspace/data/ncdb-cls-640x384 \
  --use-all-splits \
  --segmentation-root segmentation_results \
  --pred-root predictions \
  --gt-root newest_depth_maps \
  --checkpoint checkpoints/ResNet-SAN_0.05to100.ckpt \
  --max-samples 1 \
  --save-rgb \
  --save-gt \
  --save-pred
```

## 📊 출력 설명

### RGB 이미지 (`--save-rgb`)
- **형식**: RGB PNG
- **크기**: 원본 해상도 (640x384)
- **용도**: 시각적 확인, 디버깅

### GT Depth (`--save-gt`)
- **형식**: 16-bit PNG (I mode)
- **인코딩**: 원본 GT 그대로 복사
- **용도**: GT 참조, 비교 분석

### Pred Depth (`--save-pred`)
- **형식**: 16-bit PNG (I mode)
- **인코딩**: meter × 256 = PNG value
- **디코딩**: PNG value ÷ 256 = meter
- **용도**: 예측 결과 저장, 외부 도구 분석

### Visualization (`viz/`)
- **형식**: 4-panel PNG (항상 저장)
- **구성**: RGB | GT | Pred | Error Map
- **파일명**: `{idx:04d}_{stem}_{class}_ALL.png`
- **용도**: 시각적 결과 확인

### Metrics (`metrics/`)

#### summary.csv
- 클래스별 평균 메트릭
- 컬럼: Class, Count, abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3

#### per_instance.json
- 각 인스턴스별 상세 메트릭
- 정보: stem, class, mask_path, valid_pixels, gt_mean_depth, gt_median_depth, metrics

#### README.txt
- 실행 설정 정보
- 체크포인트, 데이터셋, depth 범위, 처리된 샘플 수 등

## 💡 Tips

1. **디스크 공간 절약**: 기본적으로 시각화만 저장 (플래그 없이 실행)
2. **완전한 기록**: `--save-rgb --save-gt --save-pred` 모두 사용
3. **빠른 테스트**: `--max-samples 1`로 먼저 확인
4. **여러 체크포인트 비교**: checkpoint_id가 자동으로 구분됨

## 📌 예제 출력

```
output/ResNet-SAN_0.05to100_results/
├── rgb/0000000278.png           # 371K
├── gt/0000000278.png            # 63K
├── pred/0000000278.png          # 162K
├── viz/
│   ├── 0000_0000000278_car_ALL.png   # 1.1M
│   └── 0000_0000000278_road_ALL.png  # 1.4M
├── metrics/
│   ├── summary.csv              # 클래스별 요약
│   └── per_instance.json        # 상세 메트릭
└── README.txt                   # 실행 정보
```

