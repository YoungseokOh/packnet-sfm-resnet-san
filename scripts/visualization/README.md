# Scripts - Visualization (시각화)

모델 결과를 시각화하는 스크립트들입니다.

## 📖 주요 스크립트

### `visualize_fp32_vs_int8_comparison.py`
**FP32 vs INT8 비교 시각화**

FP32 (기본 모델)과 INT8 (양자화 모델) 출력을 시각적으로 비교합니다.  
Best 5와 Worst 5 샘플을 표시합니다.

```bash
python scripts/visualization/visualize_fp32_vs_int8_comparison.py \
  --fp32_dir outputs/fp32/ \
  --int8_dir outputs/int8/ \
  --output_dir outputs/comparison_viz/
```

---

### `visualize_fp32_vs_npu.py`
**FP32 vs NPU 비교**

PyTorch FP32 모델과 NPU 출력을 비교합니다.

```bash
python scripts/visualization/visualize_fp32_vs_npu.py \
  --fp32_dir outputs/fp32/ \
  --npu_dir outputs/npu/ \
  --output_dir outputs/fp32_vs_npu_viz/
```

---

### `visualize_fp32_vs_npu_vs_gt.py`
**3개 모델 비교 (FP32 vs NPU vs GT)**

PyTorch, NPU, Ground Truth를 모두 비교합니다.

```bash
python scripts/visualization/visualize_fp32_vs_npu_vs_gt.py \
  --fp32_dir outputs/fp32/ \
  --npu_dir outputs/npu/ \
  --gt_dir /path/to/groundtruth/ \
  --output_dir outputs/triple_comparison_viz/
```

---

### `visualize_with_inverse_depth_and_gt_overlay.py`
**역깊이 시각화 + GT 오버레이**

역깊이(Inverse Depth)를 시각화하고 GT를 오버레이합니다.

```bash
python scripts/visualization/visualize_with_inverse_depth_and_gt_overlay.py \
  --depth_dir outputs/predictions/ \
  --gt_dir /path/to/groundtruth/ \
  --output_dir outputs/inverse_depth_viz/
```

---

### `visualize_ncdb_video_projection.py`
**NCDB 비디오 프로젝션 시각화**

NCDB 데이터셋의 비디오 프로젝션을 시각화합니다.  
3D 포인트 클라우드나 깊이 맵을 렌더링합니다.

```bash
# 단일 샘플 테스트
python scripts/visualization/visualize_ncdb_video_projection.py \
  --test \
  --sample_idx 0

# 전체 데이터셋 시각화
python scripts/visualization/visualize_ncdb_video_projection.py \
  --data_path /path/to/ncdb/ \
  --output_dir outputs/ncdb_viz/
```

---

### `create_fin_test_viz_index.py`
**FIN 테스트 시각화 색인 생성**

FIN 테스트 세트의 시각화 결과 HTML 인덱스를 생성합니다.

```bash
python scripts/visualization/create_fin_test_viz_index.py \
  --viz_dir outputs/fin_test_viz/ \
  --output_file outputs/fin_test_viz/index.html
```

---

## 🎯 사용 시나리오

| 목적 | 사용 스크립트 |
|------|-------------|
| 양자화 효과 비교 | `visualize_fp32_vs_int8_comparison.py` |
| NPU vs PyTorch 비교 | `visualize_fp32_vs_npu.py` |
| 3가지 모두 비교 | `visualize_fp32_vs_npu_vs_gt.py` |
| 역깊이 시각화 | `visualize_with_inverse_depth_and_gt_overlay.py` |
| NCDB 데이터 시각화 | `visualize_ncdb_video_projection.py` |
| 결과 정리 | `create_fin_test_viz_index.html` |

---

## 📊 출력 형식

### visualize_fp32_vs_int8_comparison.py
```
outputs/comparison_viz/
├── best_5/
│   ├── sample_001.jpg
│   └── ...
├── worst_5/
│   ├── sample_095.jpg
│   └── ...
└── summary.txt
```

### visualize_ncdb_video_projection.py
```
outputs/ncdb_viz/
├── sample_000/
│   ├── rgb.jpg
│   ├── depth.jpg
│   └── projection.jpg
└── ...
```

---

## 💡 팁

- 시각화 전 출력 디렉토리가 있는지 확인
- 대량 시각화 시 시간이 오래 걸릴 수 있음
- PNG 형식 권장 (품질 유지)
- 결과 이미지는 자동으로 저장됨
