# Depth Visualization Guide

## 개요

Dual-Head ResNet-SAN 모델의 Depth Prediction 결과를 시각화하는 통합 도구.
GT vs Prediction 비교를 위한 다양한 시각화 모드 지원.

## 스크립트

### 통합 스크립트 (권장)

```
scripts/visualization/visualize_depth.py
```

### 지원 모드

| 모드 | 설명 | 용도 |
|------|------|------|
| `sparse` | GT depth 위치 기반 LiDAR 스타일 포인트 | 정성적 평가, 발표 자료 |
| `dense` | 전체 픽셀 colormap + 에러맵 | 정량적 분석, 디버깅 |
| `comparison` | GT/Pred side-by-side overlay | 빠른 비교 확인 |

---

## 사용법

### 1. Sparse 모드 (LiDAR 스타일)

GT depth가 존재하는 픽셀 위치에서만 샘플링하여 LiDAR 포인트처럼 시각화.
상/하위 클리핑과 서브샘플링으로 깔끔한 시각화 가능.

```bash
python scripts/visualization/visualize_depth.py \
    --checkpoint checkpoints/resnetsan01_dual_head_ncdb_v2_640x384_0.5_to_15m_use_film/.../epoch=49...ckpt \
    --image_dir /workspace/data/ncdb-cls-640x384/synchronized_data_pangyo_optimized/640x384_newest \
    --output_dir outputs/depth_viz \
    --mode sparse \
    --num_samples 20 --random \
    --clip_percentile 15 \
    --subsample_ratio 0.3 \
    --point_size 3
```

**파라미터:**
- `--clip_percentile 15`: 상/하위 15% 깊이값 제거 (outlier 제거)
- `--subsample_ratio 0.3`: GT 포인트의 30%만 표시 (가독성 향상)
- `--point_size 3`: 포인트 크기 (픽셀)

**출력 레이아웃:**
```
+-------------------+-------------------+
|  GT Sparse        |  Pred Sparse      |
|  (LiDAR 포인트)    |  (동일 위치)       |
+-------------------+-------------------+
|  GT Dense         |  Pred Dense       |
+-------------------+-------------------+
```

### 2. Dense 모드 (Colormap)

전체 픽셀에 대한 colormap 시각화와 에러맵 표시.

```bash
python scripts/visualization/visualize_depth.py \
    --checkpoint checkpoints/.../epoch=49...ckpt \
    --image_dir /workspace/data/ncdb-cls-640x384/.../640x384_newest \
    --output_dir outputs/depth_viz \
    --mode dense \
    --num_samples 10
```

**출력 레이아웃:**
```
+-------------------+-------------------+
|  RGB Input        |  GT Depth         |
+-------------------+-------------------+
|  Pred Depth       |  Error Map        |
+-------------------+-------------------+
```

### 3. Comparison 모드 (Overlay)

RGB에 depth를 overlay하여 side-by-side 비교.

```bash
python scripts/visualization/visualize_depth.py \
    --checkpoint checkpoints/.../epoch=49...ckpt \
    --image_dir /workspace/data/ncdb-cls-640x384/.../640x384_newest \
    --output_dir outputs/depth_viz \
    --mode comparison \
    --num_samples 5
```

**출력 레이아웃:**
```
+------------------------+------------------------+
|  RGB + GT Overlay      |  RGB + Pred Overlay    |
+------------------------+------------------------+
```

---

## 공통 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--checkpoint` | (필수) | 모델 체크포인트 경로 |
| `--image_dir` | (필수) | 데이터 디렉토리 (RGB + GT depth) |
| `--output_dir` | `outputs/depth_viz` | 출력 디렉토리 |
| `--mode` | `sparse` | 시각화 모드 |
| `--num_samples` | `20` | 처리할 샘플 수 (0 = 전체) |
| `--random` | `False` | 랜덤 샘플 선택 |
| `--min_depth` | `0.5` | 최소 깊이 (m) |
| `--max_depth` | `15.0` | 최대 깊이 (m) |
| `--colormap` | `turbo` | 컬러맵 (turbo/jet/viridis/plasma) |
| `--depth_subdir` | `newest_original_depth_maps` | GT depth 서브디렉토리 |

---

## 데이터 구조

### 입력 데이터

```
image_dir/
├── image_a6/                        # RGB 이미지
│   ├── 0000050001.jpg
│   ├── 0000050002.jpg
│   └── ...
└── newest_original_depth_maps/      # GT Depth (16-bit PNG)
    ├── 0000050001.png
    ├── 0000050002.png
    └── ...
```

### GT Depth 포맷

- **형식**: 16-bit PNG
- **단위**: `pixel_value / 256 = depth_in_meters`
- **예시**: pixel=2560 → depth=10.0m

### 출력 데이터

```
output_dir/
├── 0000050001_sparse.png
├── 0000050002_sparse.png
├── ...
└── (모드별 suffix: _sparse, _dense, _comparison)
```

---

## 기술 상세

### Dual-Head Depth 계산

```python
depth = integer_sigmoid * max_depth + fractional_sigmoid
```

- `integer_sigmoid`: 정수부 (0~max_depth 범위의 coarse depth)
- `fractional_sigmoid`: 소수부 (0~1m 범위의 fine detail)
- `max_depth`: 모델 설정값 (15.0m for FiLM 모델)

### 캘리브레이션 (NCDB v3)

```python
intrinsic = {
    "k": [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391],
    "s": 1.0447,
    "div": 0.0021,
    "ux": 44.9516,
    "uy": 2.48822
}
original_size = (1920, 1536)
target_size = (640, 384)
scale_x = 640 / 1920 = 1/3
scale_y = 384 / 1536 = 1/4
```

### 사용 모델

| 모델 | 체크포인트 | 범위 | 특징 |
|------|-----------|------|------|
| FiLM (권장) | `epoch=49...0.5_to_15m_use_film` | 0.5~15m | FiLM 활성화, 최고 성능 |
| No FiLM | `epoch=23...0.1_to_30m` | 0.1~30m | FiLM 비활성화 |

---

## 예제 실행

### 빠른 테스트 (5샘플)

```bash
cd /workspace/packnet-sfm

python scripts/visualization/visualize_depth.py \
    --checkpoint "checkpoints/resnetsan01_dual_head_ncdb_v2_640x384_0.5_to_15m_use_film/default_config-train_resnet_san_ncdb_dual_head_640x384-2025.11.26-06h32m26s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt" \
    --image_dir /workspace/data/ncdb-cls-640x384/synchronized_data_pangyo_optimized/640x384_newest \
    --output_dir outputs/depth_viz_test \
    --mode sparse \
    --num_samples 5
```

### 전체 테스트셋 (랜덤 20샘플)

```bash
python scripts/visualization/visualize_depth.py \
    --checkpoint "checkpoints/resnetsan01_dual_head_ncdb_v2_640x384_0.5_to_15m_use_film/default_config-train_resnet_san_ncdb_dual_head_640x384-2025.11.26-06h32m26s/epoch=49_ncdb-cls-640x384-combined_val-loss=0.000.ckpt" \
    --image_dir /workspace/data/ncdb-cls-640x384/synchronized_data_pangyo_optimized/640x384_newest \
    --output_dir outputs/depth_viz_full \
    --mode sparse \
    --num_samples 20 --random \
    --clip_percentile 15 \
    --subsample_ratio 0.3
```

### 결과 확인 (HTTP 서버)

```bash
cd outputs/depth_viz_full && python -m http.server 8888
# 브라우저에서 http://localhost:8888/ 접속
```

---

## 성능 지표 (참고)

FiLM 모델 (epoch=49, 0.5~15m) 기준:

| 메트릭 | 값 |
|--------|-----|
| Average GT Depth | 4.23m |
| Average Pred Depth | 4.08m |
| Average Error | 0.14m (3.4%) |

---

## 관련 파일

- **스크립트**: `scripts/visualization/visualize_depth.py`
- **문서**: `docs/visualization/DEPTH_VISUALIZATION.md`
- **체크포인트**: `checkpoints/resnetsan01_dual_head_ncdb_v2_640x384_0.5_to_15m_use_film/`

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-12-04 | 통합 스크립트 `visualize_depth.py` 생성 |
| 2025-12-03 | Sparse/Dense/Comparison 모드 구현 |
| 2025-12-03 | GT 기반 sparse 마스크 방식으로 수정 |
