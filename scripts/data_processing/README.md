# Scripts - Data Processing (데이터 처리)

데이터 전처리, 변환, 관리 관련 스크립트들입니다.

## 📖 주요 스크립트

### `create_combined_splits.py`
**데이터 Split 생성**

데이터셋을 train/val/test로 분할합니다.

```bash
python scripts/data_processing/create_combined_splits.py \
  --dataset_path /path/to/dataset/ \
  --output_dir outputs/splits/ \
  --train_ratio 0.8 \
  --val_ratio 0.1
```

---

### `create_calibration_split.py`
**양자화 Calibration Split 생성**

INT8 양자화를 위한 calibration 데이터셋을 생성합니다.

```bash
python scripts/data_processing/create_calibration_split.py \
  --dataset_path /path/to/dataset/ \
  --num_calibration_samples 100 \
  --output_dir outputs/calibration_split/
```

---

### `create_ncdb_metadata.py`
**NCDB 메타데이터 생성**

NCDB 데이터셋의 메타데이터(카메라 파라미터, 경로 정보 등)를 생성합니다.

```bash
python scripts/data_processing/create_ncdb_metadata.py \
  --data_path /path/to/ncdb/ \
  --output_file outputs/ncdb_metadata.json
```

---

### `copy_calibration_images.py`
**Calibration 이미지 복사**

Calibration 데이터셋 이미지를 지정 위치로 복사합니다.

```bash
python scripts/data_processing/copy_calibration_images.py \
  --source_dir /path/to/dataset/ \
  --dest_dir /path/to/calibration_images/ \
  --num_samples 100
```

---

### `create_and_populate_fin_test_set.py`
**FIN 테스트 세트 생성 및 구성**

최종 평가용 FIN 테스트 세트를 생성하고 데이터를 구성합니다.

```bash
python scripts/data_processing/create_and_populate_fin_test_set.py \
  --source_dataset /path/to/ncdb/ \
  --output_dir /path/to/fin_test_set/ \
  --num_test_samples 1000
```

---

### `copy_npu_outputs_to_fin_test_set.py`
**NPU 출력을 FIN 테스트 세트로 복사**

NPU가 생성한 결과 파일을 FIN 테스트 세트 구조로 복사합니다.

```bash
python scripts/data_processing/copy_npu_outputs_to_fin_test_set.py \
  --npu_output_dir outputs/npu_results/ \
  --fin_test_set_dir /path/to/fin_test_set/ \
  --output_subdir npu_predictions/
```

---

### `convert_fp32_npy_to_png.py`
**NPY → PNG 변환**

NumPy 형식(`.npy`) 깊이 맵을 PNG 이미지로 변환합니다.

```bash
python scripts/data_processing/convert_fp32_npy_to_png.py \
  --input_dir outputs/predictions_npy/ \
  --output_dir outputs/predictions_png/ \
  --depth_min 0.5 \
  --depth_max 100.0
```

---

### `convert_npz_to_separate_dirs.py`
**NPZ → 분리된 디렉토리로 변환**

Dual-Head NPZ 파일을 Integer/Fractional로 분리된 디렉토리 구조로 변환합니다.

```bash
python scripts/data_processing/convert_npz_to_separate_dirs.py \
  --input_npz outputs/dual_head.npz \
  --output_dir outputs/dual_head_separated/
```

---

## 🎯 사용 시나리오

| 목적 | 사용 스크립트 |
|------|-------------|
| 데이터셋 준비 | `create_combined_splits.py` |
| 양자화 calibration | `create_calibration_split.py` + `copy_calibration_images.py` |
| 메타데이터 생성 | `create_ncdb_metadata.py` |
| 최종 평가 세트 | `create_and_populate_fin_test_set.py` |
| NPU 결과 정렬 | `copy_npu_outputs_to_fin_test_set.py` |
| 파일 형식 변환 | `convert_fp32_npy_to_png.py` + `convert_npz_to_separate_dirs.py` |

---

## 💡 팁

- 데이터 처리 전 충분한 디스크 공간 확보
- `create_combined_splits.py` 실행 후 다른 스크립트 실행
- NPY/PNG 변환 시 깊이 범위(`depth_min`, `depth_max`) 확인
- 대량 데이터 처리 시 시간이 오래 걸릴 수 있음

---

## 📊 데이터 구조 예시

### FIN Test Set
```
fin_test_set/
├── images/
│   ├── sample_0001.jpg
│   └── ...
├── gt_depth/
│   ├── sample_0001.npy
│   └── ...
├── fp32_predictions/
│   ├── sample_0001.npy
│   └── ...
└── npu_predictions/
    ├── sample_0001.npy
    └── ...
```
