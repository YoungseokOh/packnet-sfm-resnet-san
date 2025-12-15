# LiDAR Style Pixel Mask Visualization

## 개요

학습된 Dual-Head ResNet-SAN 모델의 Depth Prediction 결과를 LiDAR 스타일의 층별(layered) sparse 픽셀 마스크로 시각화하는 도구.

## 목적

1. **Ground Truth 비교**: RGB + LiDAR Depth 투영 vs RGB + Prediction Depth 투영
2. **층별 분석**: 1m 간격으로 깊이를 층으로 나누어 sparse하게 시각화
3. **정성적 평가**: 모델이 거리별로 얼마나 정확하게 예측하는지 직관적 확인

## 기술 사양

### 입력
- **RGB 이미지**: 640×384 해상도
- **GT Depth Maps**: LiDAR 기반 sparse depth
- **모델 체크포인트**: Dual-Head ResNet-SAN (epoch 49)

### 캘리브레이션 (v3)
```python
intrinsic = {
    "u2d": [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391],
    "d2u": [1.0447, 0.0021, 44.9516, 2.48822],
    "affine": [0, 0.9965, -0.0067, -0.0956, 0.1006, -0.054, 0.0106]
}
extrinsic = [0.119933, -0.129544, -0.54216, -0.0333289, -0.166123, -0.0830659]
original_size = (1920, 1536)
target_size = (640, 384)
```

### 층 구성 (프로토타입: 1m 간격)
```
Layer 1:  0.1m ~ 1.0m  (빨강)
Layer 2:  1.0m ~ 2.0m  (주황)
Layer 3:  2.0m ~ 3.0m  (노랑)
Layer 4:  3.0m ~ 4.0m  (연두)
Layer 5:  4.0m ~ 5.0m  (초록)
...
Layer 30: 29.0m ~ 30.0m (보라)
```

### 출력 형식
```
+---------------------------+---------------------------+
|     GT Depth 투영         |   Prediction Depth 투영    |
|    (RGB + LiDAR 마스크)    |    (RGB + 예측 마스크)      |
+---------------------------+---------------------------+
|         Color Legend (depth range per layer)          |
+-------------------------------------------------------+
```

## 구현 세부사항

### 픽셀 마스크 생성 방식

1. **층별 샘플링**: 각 깊이 층에서 일정 비율로 픽셀 샘플링
2. **컬러 매핑**: 깊이 층마다 고유 색상 할당 (jet colormap)
3. **투명도**: RGB 이미지 위에 마스크를 반투명하게 오버레이

### Sparse 샘플링 전략

```python
def create_sparse_mask(depth, layer_depth, spacing=8):
    """
    spacing: 샘플링 간격 (8 = 8x8 그리드에서 1개)
    """
    mask = np.zeros_like(depth, dtype=bool)
    mask[::spacing, ::spacing] = True
    layer_mask = (depth >= layer_depth[0]) & (depth < layer_depth[1])
    return mask & layer_mask
```

## 사용법

```bash
# 단일 이미지 테스트
python scripts/visualization/visualize_lidar_style_depth.py \
    --checkpoint checkpoints/resnetsan01_dual_head_fixed_ncdb_640x384_0.1_to_30m/...epoch=49...ckpt \
    --image_dir /workspace/data/ncdb-cls-640x384/synchronized_data_pangyo_optimized/640x384_newest \
    --output_dir outputs/lidar_style_viz \
    --layer_interval 1.0 \
    --num_samples 5

# 전체 테스트셋
python scripts/visualization/visualize_lidar_style_depth.py \
    --checkpoint ... \
    --image_dir ... \
    --output_dir ... \
    --all
```

## 출력 예시

```
outputs/lidar_style_viz/
├── sample_001_comparison.png
├── sample_002_comparison.png
├── ...
└── summary.html  (옵션)
```

## 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--layer_interval` | 1.0 | 층 간격 (미터) |
| `--spacing` | 8 | 샘플링 간격 (픽셀) |
| `--min_depth` | 0.1 | 최소 깊이 (미터) |
| `--max_depth` | 30.0 | 최대 깊이 (미터) |
| `--alpha` | 0.6 | 오버레이 투명도 |

## 관련 파일

- 스크립트: `scripts/visualization/visualize_lidar_style_depth.py`
- 캘리브레이션: `scripts/refrence_code/ref_calibration_data.py`
- 모델: `packnet_sfm/networks/depth/ResNetSAN01.py`

## 향후 개선

1. **가변 층 간격**: 근거리는 0.5m, 원거리는 2m 등
2. **3D 시각화**: Point Cloud 형태로 변환
3. **비디오 생성**: 연속 프레임 시각화
4. **오차 히트맵**: GT vs Prediction 차이 시각화
