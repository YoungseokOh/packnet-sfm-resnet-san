# 파일 정리 분류 (2025-11-06)

## 📁 Category 1: 지우면 안되는 파일들 (KEEP)

### 핵심 코드베이스
```
packnet_sfm/                          # 메인 라이브러리 (전체 보존 필요)
├── networks/                         # 모델 아키텍처
│   ├── depth/ResNetSAN01.py         # ✅ 현재 사용 중인 메인 모델
│   └── ...
├── models/                          # 모델 래퍼
│   ├── SemiSupCompletionModel.py    # ✅ 현재 학습에 사용
│   ├── model_wrapper.py             # ✅ Inference에 필수
│   └── ...
├── losses/                          # Loss 함수
│   ├── ssi_silog_loss.py            # ✅ Direct Depth 학습에 사용
│   └── ...
├── datasets/                        # 데이터 로더
│   ├── ncdb_dataset.py              # ✅ NCDB 데이터셋 로더
│   └── ...
├── utils/                           # 유틸리티
│   ├── depth.py                     # ✅ compute_depth_metrics 등
│   ├── post_process_depth.py        # ✅ sigmoid_to_depth_linear 등
│   └── ...
└── trainers/                        # 학습 트레이너
    └── horovod_trainer.py           # ✅ 학습/평가에 필수
```

### 현재 프로젝트 핵심 스크립트 (Direct Depth 평가/분석)
```
scripts/
├── generate_pytorch_predictions.py      # ✅ 공식 파이프라인으로 정확한 예측 생성
├── eval_official.py                     # ✅ 공식 평가 스크립트 (수정 버전)
├── eval_precomputed_simple.py           # ✅ Batch evaluation (검증 완료)
├── evaluate_npu_direct_depth_official.py # ✅ NPU Direct Depth 공식 평가
├── visualize_fp32_vs_int8_comparison.py  # ✅ FP32 vs INT8 비교 시각화 (Best/Worst 5)
                                           # ✅ RGB 로딩 버그 수정 완료 (Phase 190)
├── create_fp32_vs_int8_table.py         # ✅ FP32 vs INT8 비교 표 (4 decimals)
└── create_direct_depth_comparison_table.py # ✅ Bounded Inverse vs Direct Depth 비교
```

### 설정 파일
```
configs/
├── train_resnet_san_ncdb_640x384_direct_depth.yaml  # ✅ Direct Depth 학습 설정
├── eval_ncdb_640_test.yaml                          # ✅ 테스트 평가 설정
└── default_config.py                                # ✅ 기본 설정
```

### 체크포인트 (중요!)
```
checkpoints/
└── resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/
    └── epoch=29_ncdb-cls-640x384-combined_val-loss=0.000.ckpt  # ✅ 최종 학습 모델
```

### 빌드/배포 관련
```
Makefile                             # ✅ Docker 빌드/실행 스크립트
docker/                              # ✅ Docker 설정
.gitignore                           # ✅ Git 설정
```

### 공식 스크립트 (scripts/)
```
scripts/
├── train.py                         # ✅ 학습 스크립트
├── eval.py                          # ✅ 공식 평가 스크립트
├── infer.py                         # ✅ Inference 스크립트
├── convert_to_onnx.py               # ✅ ONNX 변환
└── create_combined_splits.py        # ✅ 데이터셋 split 생성
```

---

## 🗑️ Category 2: 지워도 될 것 같은 파일들 (CAN DELETE)

### 실험/디버깅용 임시 스크립트 (목적 달성/중복)
```
# ✅ 삭제 완료 (2025-11-06)
# Bounded Inverse 관련 (Direct Depth로 대체됨)
extract_raw_sigmoid.py               # ✅ 삭제됨
compare_sigmoid_outputs.py           # ✅ 삭제됨
verify_sigmoid_outputs.py            # ✅ 삭제됨

# 변환 방법 비교 (결론 도출 완료)
evaluate_npu_transformation_comparison.py  # ✅ 삭제됨
analyze_transformation_methods.py          # ✅ 삭제됨
explain_linear_relative_error.py           # ✅ 삭제됨
analyze_int8_resolution.py                 # ✅ 삭제됨

# 디버깅/테스트 스크립트 (목적 달성)
debug_direct_depth_output.py         # ✅ 삭제됨
test_direct_depth_setup.py           # ✅ 삭제됨
verify_direct_depth_onnx.py          # ✅ 삭제됨

# 중복된 평가 스크립트 (최종 버전은 _official.py)
evaluate_npu_direct_depth.py         # ✅ 삭제됨
evaluate_npu_official.py             # ✅ 삭제됨
eval_precomputed_depths.py           # ✅ 삭제됨
eval_all_models.py                   # ✅ 삭제됨

# 중복된 Inference 스크립트
infer_pytorch_direct_depth.py        # ✅ 삭제됨
infer_pytorch_fp32_direct_depth.py   # ✅ 삭제됨
infer_onnx_fp32_direct_depth.py      # ✅ 삭제됨

# 중복된 시각화 스크립트 (최종: visualize_fp32_vs_int8_comparison.py)
visualize_direct_depth_best_worst.py      # ✅ 삭제됨
visualize_onnx_fp32_vs_int8.py            # ✅ 삭제됨
visualize_onnx_fp32_vs_npu_int8.py        # ✅ 삭제됨
visualize_pytorch_vs_onnx_vs_npu.py       # ✅ 삭제됨

# 중복된 비교 테이블 스크립트
create_comparison_table.py           # ✅ 삭제됨
create_fp32_vs_int8_comparison.py    # ✅ 삭제됨

# 기타 분석 스크립트 (일회성 분석 완료)
analyze_direct_depth_int8.py         # ✅ 삭제됨
analyze_gt_depth_range.py            # ✅ 삭제됨
analyze_loss_scale.py                # ✅ 삭제됨
collect_gt_depths.py                 # ✅ 삭제됨
compare_npu_gpu_gt.py                # ✅ 삭제됨
validate_checkpoint_metrics.py       # ✅ 삭제됨

# ONNX INT8 (ONNX Runtime 한계로 NPU 사용)
quantize_and_infer_onnx_int8.py      # ✅ 삭제됨
```

### 실험용 이미지/로그
```
quantization_error_analysis.png      # 🔸 분석 완료 (문서에 포함됨)
train_direct_depth.log               # 🔸 학습 로그 (epoch 29 완료, 백업 후 삭제 가능)
```

### 사용하지 않는 configs (실험/레거시)
```
configs/
├── train_omnicam.yaml               # 🔸 Omnicam (사용 안함)
├── train_yolov8_san_kitti*.yaml     # 🔸 YOLOv8 실험 (사용 안함)
├── eval_ddad.yaml                   # 🔸 DDAD 데이터셋 (사용 안함)
├── train_ddad.yaml                  # 🔸 DDAD 데이터셋 (사용 안함)
├── overfit_*.yaml                   # 🔸 Overfit 테스트 (개발용)
├── train_kitti.yaml                 # 🔸 KITTI (NCDB 사용 중)
├── eval_kitti.yaml                  # 🔸 KITTI (NCDB 사용 중)
├── train_packnet_san_*.yaml         # 🔸 PackNet 아키텍처 (ResNet 사용 중)
├── eval_packnet_san_kitti.yaml      # 🔸 PackNet 평가 (사용 안함)
└── train_resnet_san_ncdb.yaml       # 🔸 구 버전 (640x384_direct_depth 사용)
```

### 사용하지 않는 scripts/ (레거시/실험)
```
scripts/
├── ref_*.py                         # 🔸 Reference 코드 (사용 안함, 백업 목적으로 보존)
├── check_mask.py                    # ✅ 삭제됨 (마스크 확인)
├── compare_*.py                     # ✅ 삭제됨 (비교 스크립트들)
├── analyze_*.py                     # ✅ 삭제됨 (분석 스크립트들)
├── visualize_*.py                   # ✅ 삭제됨 (구 시각화, visualize_fp32_vs_int8_comparison.py 제외)
├── convert_png_to_jpg.py            # ✅ 삭제됨 (변환 유틸)
├── create_kitti_sample.py           # ✅ 삭제됨 (KITTI 샘플)
├── create_ncdb_sample.py            # ✅ 삭제됨 (NCDB 샘플)
├── create_vadas_lookup_table.py     # ✅ 삭제됨 (VADAS LUT)
├── advanced_verify.py               # ✅ 삭제됨 (고급 검증)
├── verify_*.py                      # ✅ 삭제됨 (검증 스크립트들)
├── update_split_paths.py            # ✅ 삭제됨 (Split 경로 업데이트)
├── check_yolov8_model_type.py       # ✅ 삭제됨 (YOLOv8 확인)
├── eval_onnx.py                     # ✅ 삭제됨 (ONNX 평가)
├── eval_pytorch_onnx_comparison.py  # ✅ 삭제됨 (PyTorch-ONNX 비교)
├── evaluate_depth_maps.py           # ✅ 삭제됨 (Depth map 평가)
├── evaluate_ncdb_*.py               # ✅ 삭제됨 (NCDB 평가 스크립트들)
├── infer_ncdb.py                    # ✅ 삭제됨 (NCDB inference)
├── prepare_data.py                  # ✅ 삭제됨 (데이터 준비)
├── create_depth_maps.py             # ✅ 삭제됨 (Depth map 생성)
└── EVALUATE_USAGE.md                # ✅ 삭제됨 (평가 사용법 문서)
```

### 문서/로그 (정리 가능)
```
todo/                                # ✅ 삭제됨 (오래된 작업 목록)
daily_work_log/                      # 🔸 일일 작업 로그 (백업 후 삭제 가능)
docs/                                # ✅ 삭제됨 (Sphinx HTML 문서, 14MB 절약)
docs_md/                             # 🔸 마크다운 문서 (필요시 확인 후 정리)
scripts/EVALUATE_USAGE.md            # 🔸 평가 사용법 (오래됨)
```

---

## 📊 요약

### 보존 (KEEP)
- **핵심 라이브러리**: `packnet_sfm/` 전체
- **최종 검증 스크립트**: 
  - `scripts/generate_pytorch_predictions.py`
  - `scripts/eval_precomputed_simple.py`
  - `scripts/evaluate_npu_direct_depth_official.py`
  - `scripts/visualize_fp32_vs_int8_comparison.py` (RGB 버그 수정 완료)
  - `scripts/create_fp32_vs_int8_table.py`
- **최종 문서**: `DIRECT_DEPTH_EVALUATION_SUMMARY.md`
- **최종 체크포인트**: `checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_mask_true/epoch=29*.ckpt`
- **공식 스크립트**: `scripts/{train,eval,infer,convert_to_onnx}.py`
- **필수 설정**: `configs/train_resnet_san_ncdb_640x384_direct_depth.yaml`, `configs/eval_ncdb_640_test.yaml`

### 삭제 가능 (CAN DELETE)
- **실험 완료 스크립트**: ~30개 ✅ **삭제됨** (분석/디버깅/중복 스크립트)
- **사용 안하는 configs**: ~15개 (DDAD, KITTI, YOLOv8, PackNet 등)
- **사용 안하는 scripts**: ~17개 (ref_* 제외, analyze_*, compare_* 등)
- **오래된 문서/로그**: todo/ ✅ **삭제됨**, daily_work_log/

**삭제 가능 파일 수**: 약 **32개** (전체의 약 10%)

---

## ⚠️ 주의사항

1. **삭제 전 백업**: 중요한 분석 결과나 로그는 백업
2. **Git 히스토리**: Git에 커밋되어 있으므로 필요시 복구 가능
3. **단계적 삭제**: 한 번에 삭제하지 말고 카테고리별로 확인하며 삭제
4. **outputs/ 폴더**: 용량이 크면 정리 (재생성 가능한 결과물)

