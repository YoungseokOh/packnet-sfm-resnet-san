# NCDB Video Projection Visualization

## 프로젝트 개요

NCDB Video 데이터셋의 RGB 이미지에 GT depth와 NPU depth를 시각화하여 겹쳐서 표현하는 프로젝트입니다.

## 요구사항 분석

### 1. 입력 데이터 구조
```
ncdb_video/
├── RGB/
├── GT/
└── NPU/
```

### 2. 출력 형식
- **저장 위치**: `res/` 디렉토리
- **파일명 형식**: `[filename]_res.jpg`
- **이미지 크기**: 1280 × 384 (640×384 × 2)

### 3. 이미지 구성
```
[왼쪽 640×384]          [오른쪽 640×384]
RGB + GT Depth       RGB + NPU Depth
(Colormap 오버레이)  (Colormap 오버레이)
```

## 핵심 기술 명세

### 3.1 Depth 시각화 방식
- **방법**: RGB 이미지와 Colormap된 Depth를 알파 블렌딩으로 겹침
- **Colormap**: 기존 visualization 코드의 progressive coarsening 커스텀 colormap 사용
  - 근거리 (0.5-0.75m): Pure Red (완전 빨강)
  - 근거리 (0.75-3.0m): 0.1m 스텝 (Red → Orange → Yellow)
  - 중거리 (3.0-6.0m): 0.2m 스텝 (Yellow → Green)
  - 원거리 (6.0-15.0m): 0.5-1m 스텝 (Cyan → Blue)

### 3.2 알파 블렌딩 파라미터
- **투명도 (Alpha)**: 조절 가능한 파라미터 (기본값: 0.6)
  - 형식: `alpha = 0.0 ~ 1.0` (0=투명, 1=불투명)

### 3.3 포인트 크기 파라미터
- **마커 크기**: 조절 가능한 파라미터 (기본값: 30)
  - Depth 포인트의 scatter plot 크기를 조절하여 시각성 개선

### 3.4 유효 픽셀 마스킹
- **동일 좌표 사용**: GT에서 유효한 픽셀(GT depth > 0)만 추출
- **양쪽 패널 일관성**: 왼쪽(GT) 및 오른쪽(NPU) 모두 동일한 유효 픽셀 좌표에서만 표시
- **목적**: GT 기준의 신뢰할 수 있는 영역에서만 비교 시각화

## 구현 방식

### Phase 1: 단일 이미지 테스트
1. 샘플 1개 선택
2. RGB, GT depth, NPU depth 로드
3. GT 유효 픽셀 마스킹
4. 양쪽 depth를 colormap + alpha blending으로 RGB에 오버레이
5. 1280×384 좌우 결합 이미지 생성
6. JPG 저장

### Phase 2: 배치 처리
- Phase 1 로직을 전체 `ncdb_video/` 데이터셋에 적용
- 모든 이미지에 대해 결과 생성

## 파라미터 설정

```python
# 조절 가능한 파라미터
MARKER_SIZE = 30          # Depth 포인트 크기 (기본값: 30)
ALPHA_BLEND = 0.6         # RGB-Depth 알파 블렌딩 투명도 (기본값: 0.6)
MIN_DEPTH = 0.5           # 최소 거리 (m)
MAX_DEPTH = 15.0          # 최대 거리 (m)
```

## 향후 확장 계획

### LiDAR 마스킹 방식
현재는 GT depth 기준의 마스킹을 사용하지만, 향후 확장:

1. **LiDAR 스파스 마스크 생성**
   - LiDAR 스캔 데이터로부터 sparse mask 추출
   - Raw RGB-to-Depth 모델의 입력으로 사용

2. **Masked Inference**
   - Raw depth 모델에서 LiDAR 마스크 적용
   - 신뢰할 수 있는 LiDAR 영역에서만 depth 추정
   - 시뮬레이션을 통한 현실적 성능 평가

3. **비교 구조**
   ```
   Raw Model + LiDAR Mask → NPU Inference
   → RGB + (GT, NPU) 시각화
   ```

## 파일 구조

```
scripts/
└── visualize_ncdb_video_projection.py  # 메인 시각화 스크립트

res/
└── [filename]_res.jpg                   # 결과 이미지 (배치 생성)

docs/
└── ncdb_video_projection_visualization.md  # 이 문서
```

## 기술 스택

- **언어**: Python 3.9+
- **주요 라이브러리**:
  - PIL/Pillow: 이미지 로드/저장
  - NumPy: 배열 연산
  - Matplotlib: Colormap, 시각화
  - OpenCV: 이미지 처리 (선택사항)

## 실행 방식

```bash
# 테스트 플로우 (단일 이미지)
python scripts/visualize_ncdb_video_projection.py --test --sample_idx 0

# 배치 처리 (전체 데이터셋)
python scripts/visualize_ncdb_video_projection.py --batch
```

## 색상 코드 참조

| 거리 범위 | 색상 | RGB 값 |
|---------|-----|-------|
| 0.5-0.75m | Pure Red | (1.0, 0.0, 0.0) |
| 0.75-1.5m | Red-Orange | (1.0, 0.2-0.7, 0.0) |
| 1.5-2.5m | Orange | (1.0, 0.7-0.9, 0.0) |
| 2.5-3.5m | Yellow | (1.0, 1.0, 0.0-0.2) |
| 3.5-5.5m | Green-Yellow | (0.5-0.9, 1.0, 0.0-0.3) |
| 5.5-6.5m | Cyan | (0.0, 1.0, 1.0) |
| 6.5-10m | Light Blue | (0.0, 0.3-1.0, 1.0) |
| 10-15m | Deep Blue | (0.0, 0.0, 1.0) |

---

**문서 버전**: 1.0  
**작성일**: 2025-11-13  
**상태**: 구현 전 명세 문서
