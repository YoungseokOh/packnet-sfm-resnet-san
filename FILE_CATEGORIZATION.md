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
# Bounded Inverse 관련 (Direct Depth로 대체됨)
extract_raw_sigmoid.py               # 🔸 Sigmoid 값 추출 (실험용, 더 이상 필요 없음)
compare_sigmoid_outputs.py           # 🔸 PyTorch vs NPU sigmoid 비교 (분석 완료)
verify_sigmoid_outputs.py            # 🔸 Sigmoid 검증 (분석 완료)

# 변환 방법 비교 (결론 도출 완료)
evaluate_npu_transformation_comparison.py  # 🔸 Linear vs Bounded Inverse 비교 (결론: Direct Depth 선택)
analyze_transformation_methods.py          # 🔸 변환 방법 분석 (설명 완료)
explain_linear_relative_error.py           # 🔸 Linear 상대 오차 설명 (문서화 완료)
analyze_int8_resolution.py                 # 🔸 INT8 해상도 분석 (이론 검증 완료)

# 디버깅/테스트 스크립트 (목적 달성)
debug_direct_depth_output.py         # 🔸 모델 출력 디버깅 (검증 완료)
test_direct_depth_setup.py           # 🔸 Direct Depth 설정 테스트 (검증 완료)
verify_direct_depth_onnx.py          # 🔸 ONNX 모델 검증 (검증 완료)

# 중복된 평가 스크립트 (최종 버전은 _official.py)
evaluate_npu_direct_depth.py         # 🔸 NPU 평가 (evaluate_npu_direct_depth_official.py로 대체)
evaluate_npu_official.py             # 🔸 Bounded Inverse NPU 평가 (더 이상 사용 안함)
eval_precomputed_depths.py           # 🔸 복잡한 버전 (eval_precomputed_simple.py로 대체)
eval_all_models.py                   # 🔸 통합 평가 (개별 평가로 대체)

# 중복된 Inference 스크립트
infer_pytorch_direct_depth.py        # 🔸 초기 버전 (generate_pytorch_predictions.py로 대체)
infer_pytorch_fp32_direct_depth.py   # 🔸 중복 (generate_pytorch_predictions.py 사용)
infer_onnx_fp32_direct_depth.py      # 🔸 ONNX FP32 inference (필요 시 재생성 가능)

# 중복된 시각화 스크립트 (최종: visualize_fp32_vs_int8_comparison.py)
visualize_direct_depth_best_worst.py      # 🔸 구 버전 (새 레이아웃으로 대체)
visualize_onnx_fp32_vs_int8.py            # 🔸 ONNX 비교 (NPU 결과 사용)
visualize_onnx_fp32_vs_npu_int8.py        # 🔸 ONNX+NPU 비교 (중복)
visualize_pytorch_vs_onnx_vs_npu.py       # 🔸 3-way 비교 (복잡, 필요시 재생성)

# 중복된 비교 테이블 스크립트
create_comparison_table.py           # 🔸 Bounded Inverse 비교 (더 이상 사용 안함)
create_fp32_vs_int8_comparison.py    # 🔸 중복 (create_fp32_vs_int8_table.py 사용)

# 기타 분석 스크립트 (일회성 분석 완료)
analyze_direct_depth_int8.py         # 🔸 INT8 분석 (현재 열려있지만 분석 완료)
analyze_gt_depth_range.py            # 🔸 GT depth 범위 분석 (확인 완료)
analyze_loss_scale.py                # 🔸 Loss scale 분석 (실험 완료)
collect_gt_depths.py                 # 🔸 GT depth 수집 (일회성 작업)
compare_npu_gpu_gt.py                # 🔸 NPU/GPU/GT 3-way 비교 (결론 도출)
validate_checkpoint_metrics.py       # 🔸 체크포인트 검증 (epoch 29 검증 완료)

# ONNX INT8 (ONNX Runtime 한계로 NPU 사용)
quantize_and_infer_onnx_int8.py      # 🔸 ONNX INT8 양자화 (ConvInteger 미지원으로 사용 불가)
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
├── ref_*.py                         # 🔸 Reference 코드 (사용 안함)
├── check_mask.py                    # 🔸 마스크 확인 (개발용)
├── compare_*.py                     # 🔸 비교 스크립트들 (일회성 분석)
├── analyze_*.py                     # 🔸 분석 스크립트들 (일회성)
├── visualize_*.py                   # 🔸 시각화 스크립트들 (루트에 최신 버전)
├── convert_png_to_jpg.py            # 🔸 변환 유틸 (필요시 재생성)
├── create_kitti_sample.py           # 🔸 KITTI 샘플 (사용 안함)
├── create_ncdb_sample.py            # 🔸 NCDB 샘플 생성 (일회성)
├── create_vadas_lookup_table.py     # 🔸 VADAS LUT (사용 안함)
├── advanced_verify.py               # 🔸 검증 스크립트 (개발용)
├── verify_*.py                      # 🔸 검증 스크립트들 (개발용)
├── update_split_paths.py            # 🔸 Split 경로 업데이트 (일회성)
└── check_yolov8_model_type.py       # 🔸 YOLOv8 확인 (사용 안함)
```

### 문서/로그 (정리 가능)
```
todo/                                # 🔸 TODO 목록 (오래된 작업 목록)
daily_work_log/                      # 🔸 일일 작업 로그 (백업 후 삭제 가능)
docs/                                # 🔸 문서 (필요시 확인 후 정리)
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
- **실험 완료 스크립트**: ~30개 (분석/디버깅/중복 스크립트)
- **사용 안하는 configs**: ~15개 (DDAD, KITTI, YOLOv8, PackNet 등)
- **사용 안하는 scripts**: ~30개 (ref_*, analyze_*, compare_* 등)
- **오래된 문서/로그**: todo/, daily_work_log/

**삭제 가능 파일 수**: 약 **75-80개** (전체의 약 25-30%)

---

## ⚠️ 주의사항

1. **삭제 전 백업**: 중요한 분석 결과나 로그는 백업
2. **Git 히스토리**: Git에 커밋되어 있으므로 필요시 복구 가능
3. **단계적 삭제**: 한 번에 삭제하지 말고 카테고리별로 확인하며 삭제
4. **outputs/ 폴더**: 용량이 크면 정리 (재생성 가능한 결과물)

