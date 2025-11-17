# Scripts 폴더 분석 및 정리 계획

**작성일**: 2025-11-17  
**상태**: 분석 완료 - 정리 제안

---

## 📊 현재 상황

```
scripts/
├── __pycache__/
├── ref/                                  # 📁 참고 폴더
└── 36개의 Python 스크립트 (지저분함)
```

**문제점**: 
- 용도가 다른 스크립트들이 모두 섞여있음
- 핵심 스크립트와 실험용/보조 스크립트 구분 안됨
- 각 기능별 분류 없음

---

## 🏗️ 제안하는 새로운 구조

```
scripts/
│
├── 🔴 core/                             # 핵심 스크립트 (무조건 필요)
│   ├── train.py                         ⭐ 모델 학습
│   ├── infer.py                         ⭐ 단일 이미지 추론
│   ├── eval.py                          ⭐ 공식 평가
│   └── eval_official.py                 ⭐ 공식 평가 (수정 버전)
│
├── 🔵 evaluation/                       # 평가/검증 도구
│   ├── generate_pytorch_predictions.py  # PyTorch 예측 생성 (공식 파이프라인)
│   ├── eval_precomputed_simple.py       # 미리계산된 깊이 평가
│   ├── evaluate_npu_direct_depth_official.py  # NPU Direct Depth 평가
│   ├── evaluate_dual_head.py            # Dual-Head NPU 평가
│   ├── evaluate_dual_head_simple.py     # Dual-Head 간편 평가
│   ├── verify_dual_head_output.py       # Dual-Head 출력 검증
│   └── verify_gt_rgb_matching.py        # GT-RGB 매칭 검증
│
├── 🟢 visualization/                    # 시각화 도구
│   ├── visualize_fp32_vs_int8_comparison.py     # FP32 vs INT8 비교 (Best/Worst 5)
│   ├── visualize_fp32_vs_npu.py         # FP32 vs NPU 시각화
│   ├── visualize_fp32_vs_npu_vs_gt.py   # FP32 vs NPU vs GT
│   ├── visualize_with_inverse_depth_and_gt_overlay.py  # 역깊이 시각화
│   ├── visualize_ncdb_video_projection.py  # NCDB 비디오 프로젝션
│   └── create_fin_test_viz_index.py     # FIN 테스트 시각화 색인
│
├── 🟡 data_processing/                  # 데이터 처리
│   ├── create_combined_splits.py        # 데이터 split 생성
│   ├── create_calibration_split.py      # 양자화 calibration split
│   ├── create_ncdb_metadata.py          # NCDB 메타데이터 생성
│   ├── copy_calibration_images.py       # Calibration 이미지 복사
│   ├── create_and_populate_fin_test_set.py  # FIN 테스트 세트 생성
│   ├── copy_npu_outputs_to_fin_test_set.py  # NPU 출력 복사
│   ├── convert_fp32_npy_to_png.py       # NPY → PNG 변환
│   └── convert_npz_to_separate_dirs.py  # NPZ → 디렉토리 변환
│
├── 🟠 onnx_conversion/                  # ONNX 변환
│   ├── convert_to_onnx.py               # 기본 ONNX 변환
│   ├── convert_dual_head_to_onnx.py     # Dual-Head ONNX 변환
│   ├── validate_dual_head_onnx.py       # Dual-Head ONNX 검증
│   ├── test_onnx_with_real_image.py     # ONNX 실제 이미지 테스트
│   └── save_dual_head_outputs.py        # Dual-Head 출력 저장
│
├── 🟣 analysis/                         # 분석 도구
│   ├── compare_dual_head_components.py  # Dual-Head 컴포넌트 비교
│   ├── create_fp32_vs_int8_table.py     # FP32 vs INT8 표
│   ├── create_fp32_vs_npu_table.py      # FP32 vs NPU 표
│   ├── create_direct_depth_comparison_table.py  # Direct Depth 비교 표
│   └── create_distance_based_metrics_table.py   # 거리 기반 메트릭
│
└── ref/                                 # 참고 폴더 (기존 유지)
```

---

## 📋 파일 분류 상세

### 🔴 CORE (4개) - 필수 스크립트

| 파일명 | 용도 | 사용 빈도 | 상태 |
|--------|------|----------|------|
| train.py | 모델 학습 | ⭐⭐⭐ 높음 | ✅ 필수 |
| infer.py | 단일/배치 추론 | ⭐⭐ 중간 | ✅ 필수 |
| eval.py | 공식 평가 | ⭐⭐⭐ 높음 | ✅ 필수 |
| eval_official.py | 수정된 평가 | ⭐⭐⭐ 높음 | ✅ 필수 |

**특징**: 매일 사용하는 핵심 스크립트

---

### 🔵 EVALUATION (7개) - 평가/검증

| 파일명 | 용도 | 필요도 | 상태 |
|--------|------|--------|------|
| generate_pytorch_predictions.py | PyTorch 공식 파이프라인 예측 | ⭐⭐⭐ | ✅ 유지 |
| eval_precomputed_simple.py | 미리계산된 깊이로 배치 평가 | ⭐⭐ | ✅ 유지 |
| evaluate_npu_direct_depth_official.py | NPU Direct Depth 평가 | ⭐⭐⭐ | ✅ 유지 |
| evaluate_dual_head.py | Dual-Head NPU 평가 | ⭐⭐⭐ | ✅ 유지 |
| evaluate_dual_head_simple.py | Dual-Head 간편 평가 | ⭐⭐ | ✅ 유지 |
| verify_dual_head_output.py | Dual-Head 출력 검증 | ⭐⭐ | ✅ 유지 |
| verify_gt_rgb_matching.py | GT-RGB 매칭 검증 | ⭐ | ✅ 유지 |

**특징**: 평가 관련 스크립트들

---

### 🟢 VISUALIZATION (6개) - 시각화

| 파일명 | 용도 | 필요도 | 상태 |
|--------|------|--------|------|
| visualize_fp32_vs_int8_comparison.py | FP32 vs INT8 비교 시각화 | ⭐⭐⭐ | ✅ 유지 |
| visualize_fp32_vs_npu.py | FP32 vs NPU 비교 | ⭐⭐⭐ | ✅ 유지 |
| visualize_fp32_vs_npu_vs_gt.py | 3개 모델 비교 | ⭐⭐ | ✅ 유지 |
| visualize_with_inverse_depth_and_gt_overlay.py | 역깊이 + GT 오버레이 | ⭐⭐ | ✅ 유지 |
| visualize_ncdb_video_projection.py | NCDB 비디오 프로젝션 | ⭐⭐ | ✅ 유지 |
| create_fin_test_viz_index.py | FIN 테스트 시각화 색인 | ⭐ | ⚠️ 검토필요 |

**특징**: 결과 시각화 도구들

---

### 🟡 DATA_PROCESSING (8개) - 데이터 처리

| 파일명 | 용도 | 필요도 | 상태 |
|--------|------|--------|------|
| create_combined_splits.py | 데이터 split 생성 | ⭐⭐⭐ | ✅ 유지 |
| create_calibration_split.py | Calibration split 생성 | ⭐⭐ | ✅ 유지 |
| create_ncdb_metadata.py | NCDB 메타데이터 | ⭐⭐ | ✅ 유지 |
| copy_calibration_images.py | Calibration 이미지 복사 | ⭐⭐ | ✅ 유지 |
| create_and_populate_fin_test_set.py | FIN 테스트 세트 생성 | ⭐⭐ | ⚠️ 검토필요 |
| copy_npu_outputs_to_fin_test_set.py | NPU 출력 복사 | ⭐ | ⚠️ 검토필요 |
| convert_fp32_npy_to_png.py | NPY → PNG | ⭐ | ⚠️ 검토필요 |
| convert_npz_to_separate_dirs.py | NPZ → 디렉토리 | ⭐ | ⚠️ 검토필요 |

**특징**: 데이터 전처리 및 변환 도구들

---

### 🟠 ONNX_CONVERSION (5개) - ONNX 변환

| 파일명 | 용도 | 필요도 | 상태 |
|--------|------|--------|------|
| convert_to_onnx.py | 기본 ONNX 변환 | ⭐⭐⭐ | ✅ 유지 |
| convert_dual_head_to_onnx.py | Dual-Head ONNX | ⭐⭐⭐ | ✅ 유지 |
| validate_dual_head_onnx.py | Dual-Head ONNX 검증 | ⭐⭐⭐ | ✅ 유지 |
| test_onnx_with_real_image.py | ONNX 실제 이미지 테스트 | ⭐⭐ | ✅ 유지 |
| save_dual_head_outputs.py | Dual-Head 출력 저장 | ⭐⭐ | ✅ 유지 |

**특징**: ONNX 변환 및 검증 스크립트들

---

### 🟣 ANALYSIS (5개) - 분석 도구

| 파일명 | 용도 | 필요도 | 상태 |
|--------|------|--------|------|
| compare_dual_head_components.py | Integer vs Fractional 비교 | ⭐⭐⭐ | ✅ 유지 |
| create_fp32_vs_int8_table.py | 평가 지표 표 | ⭐⭐⭐ | ✅ 유지 |
| create_fp32_vs_npu_table.py | FP32 vs NPU 표 | ⭐⭐⭐ | ✅ 유지 |
| create_direct_depth_comparison_table.py | Bounded Inverse vs Direct Depth | ⭐⭐ | ✅ 유지 |
| create_distance_based_metrics_table.py | 거리 기반 메트릭 | ⭐⭐ | ✅ 유지 |

**특징**: 결과 분석 및 표 생성 도구들

---

## 🔍 의심스러운 파일들 (검토 필요)

### 논의 필요한 파일들

다음 파일들은 특정 프로젝트에만 필요한 것 같습니다. **사용 여부를 확인하시고 직접 삭제 결정**:

```python
# ⚠️ FIN_TEST_SET 관련 (프로젝트 특화)
create_and_populate_fin_test_set.py      # FIN 테스트 세트 전용
copy_npu_outputs_to_fin_test_set.py      # FIN 테스트 세트 전용
create_fin_test_viz_index.py             # FIN 테스트 시각화 전용

# ⚠️ 변환 유틸리티 (일회성?)
convert_fp32_npy_to_png.py               # NPY → PNG 변환 (한 번만 사용?)
convert_npz_to_separate_dirs.py          # NPZ → 디렉토리 (한 번만 사용?)

# 💭 실제 사용 여부 불명확
verify_gt_rgb_matching.py                # GT-RGB 매칭 검증 (사용 빈도?)
create_distance_based_metrics_table.py   # 거리 기반 메트릭 (특화?)
```

**확인 필요**: 이 파일들을 정말 사용하시나요?

---

## 🚀 정리 실행 단계

### Phase 1: 폴더 구조 생성
```bash
mkdir -p scripts/{core,evaluation,visualization,data_processing,onnx_conversion,analysis}
```

### Phase 2: 파일 이동 (자동 실행 가능)
```bash
# core
mv scripts/{train,infer,eval}.py scripts/core/
mv scripts/eval_official.py scripts/core/

# evaluation
mv scripts/{generate_pytorch_predictions,eval_precomputed_simple,evaluate_npu_direct_depth_official}.py scripts/evaluation/
mv scripts/{evaluate_dual_head,evaluate_dual_head_simple,verify_dual_head_output,verify_gt_rgb_matching}.py scripts/evaluation/

# visualization
mv scripts/{visualize_fp32_vs_int8_comparison,visualize_fp32_vs_npu,visualize_fp32_vs_npu_vs_gt}.py scripts/visualization/
mv scripts/{visualize_with_inverse_depth_and_gt_overlay,visualize_ncdb_video_projection,create_fin_test_viz_index}.py scripts/visualization/

# data_processing
mv scripts/{create_combined_splits,create_calibration_split,create_ncdb_metadata,copy_calibration_images}.py scripts/data_processing/
mv scripts/{create_and_populate_fin_test_set,copy_npu_outputs_to_fin_test_set,convert_fp32_npy_to_png,convert_npz_to_separate_dirs}.py scripts/data_processing/

# onnx_conversion
mv scripts/{convert_to_onnx,convert_dual_head_to_onnx,validate_dual_head_onnx,test_onnx_with_real_image,save_dual_head_outputs}.py scripts/onnx_conversion/

# analysis
mv scripts/{compare_dual_head_components,create_fp32_vs_int8_table,create_fp32_vs_npu_table,create_direct_depth_comparison_table,create_distance_based_metrics_table}.py scripts/analysis/
```

### Phase 3: README 생성
각 폴더에 README.md 생성 (사용 설명서)

### Phase 4: 의심파일 정리
사용하지 않는 파일 → `scripts/deprecated/` 또는 삭제

---

## 📝 각 폴더별 README 예시

### scripts/core/README.md
```markdown
# Core Scripts (핵심 스크립트)

이 폴더의 스크립트는 매일 사용하는 핵심 기능입니다.

## 주요 스크립트

- **train.py**: 모델 학습
- **infer.py**: 추론 실행
- **eval.py**: 평가 수행
- **eval_official.py**: 공식 평가 (수정 버전)
```

### scripts/evaluation/README.md
```markdown
# Evaluation Scripts (평가/검증)

모델 평가 및 검증 관련 스크립트들입니다.

## 주요 스크립트

- **generate_pytorch_predictions.py**: PyTorch 공식 파이프라인으로 예측 생성
- **evaluate_npu_direct_depth_official.py**: NPU Direct Depth 평가
- **evaluate_dual_head.py**: Dual-Head 모델 평가
- ...
```

---

## 📊 정리 전후 비교

### Before (지금)
```
scripts/
├── 36개의 파이썬 파일 섞여있음
└── 무엇을 어디서 찾는지 불명확
```

### After (정리 후)
```
scripts/
├── core/              [4개 - 핵심]
├── evaluation/        [7개 - 평가]
├── visualization/     [6개 - 시각화]
├── data_processing/   [8개 - 데이터]
├── onnx_conversion/   [5개 - ONNX]
├── analysis/          [5개 - 분석]
└── ref/              [기존]
```

**장점**: 
- ✅ 목적별로 명확히 구분
- ✅ 필요한 스크립트 빠르게 찾기 쉬움
- ✅ 새로운 팀원이 이해하기 쉬움

---

## ✨ 사용 예시

### Before
```bash
# 어디 있는 파일이지?
ls scripts/ | grep eval   # 여러 개 나옴
```

### After
```bash
# 평가 스크립트가 필요하면?
ls scripts/evaluation/

# 데이터 처리?
ls scripts/data_processing/

# ONNX 변환?
ls scripts/onnx_conversion/
```

---

## 🎯 실행 계획

1. **폴더 생성** → scripts/{core,evaluation,visualization,data_processing,onnx_conversion,analysis}
2. **파일 이동** → 각 폴더로 자동 이동
3. **README 작성** → 각 폴더마다 사용 설명서
4. **의심파일 검토** → 사용자가 직접 확인 후 삭제 결정

---

## 📌 의심 파일 최종 목록 (사용자 검토 필요)

다음 파일들은 **특정 프로젝트나 일회성 용도**로 보입니다.  
**사용 여부를 확인하시고 직접 결정**해주세요:

| 파일명 | 이유 | 삭제할까? |
|--------|------|----------|
| create_and_populate_fin_test_set.py | FIN_TEST_SET 프로젝트 특화 | ❓ |
| copy_npu_outputs_to_fin_test_set.py | FIN_TEST_SET 프로젝트 특화 | ❓ |
| create_fin_test_viz_index.py | FIN_TEST_SET 프로젝트 특화 | ❓ |
| convert_fp32_npy_to_png.py | 일회성 변환 유틸리티 | ❓ |
| convert_npz_to_separate_dirs.py | 일회성 변환 유틸리티 | ❓ |
| verify_gt_rgb_matching.py | 사용 빈도 불명확 | ❓ |
| create_distance_based_metrics_table.py | 특화된 메트릭 | ❓ |

**의견**: 지우지 말고 `scripts/deprecated/` 폴더에 옮겨놓고, 나중에 필요 없다고 확신될 때 지우는 것이 낫습니다.

---

## 🎉 완료!

이 분석을 바탕으로 정리를 진행하겠습니다!
