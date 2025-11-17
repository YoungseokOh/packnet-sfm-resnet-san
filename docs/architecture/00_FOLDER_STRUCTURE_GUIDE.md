# 📁 코드베이스 폴더 구조 가이드

## 최종 정리 완료 (2025-11-17)

모든 파일이 폴더별로 체계적으로 정리되었습니다.

---

## 📂 폴더 구조

### 루트 디렉토리 (Root)
```
/workspace/packnet-sfm/
├── packnet_sfm/           # ✅ 핵심 모델 코드 (변경 금지)
├── scripts/               # ✅ 학습/평가 스크립트 (변경 금지)
├── configs/               # ✅ 학습 설정 파일 (변경 금지)
├── docs/                  # 📚 문서 (계층적 정렬)
├── analysis_results/      # 📊 분석 결과 저장소
├── outputs/               # 🎯 모델 출력/결과
├── checkpoints/           # 💾 학습된 모델
├── Makefile               # 🐳 Docker 명령어
└── ...
```

---

## 📚 /docs 폴더 구조 (완전 정리)

### 계층 1: 주요 섹션
```
docs/
├── README.md                      # 📖 전체 문서 가이드
├── FOLDER_STRUCTURE_GUIDE.md      # 📁 이 파일 (폴더 가이드)
│
├── analysis/                      # 📊 분석 & 검증 자료
│   ├── 00_INDEX.md               # 분석 자료 색인
│   ├── 01_WEIGHT_JUSTIFICATION_SUMMARY.md
│   ├── 02_NUMERICAL_VALIDATION.md
│   ├── 02_NUMERICAL_VALIDATION_RESULTS.md
│   ├── 03_MIN_DEPTH_EFFECTS.md
│   ├── 04_48_IMPACT.md
│   ├── 05_CONSISTENCY_WEIGHT.md
│   │
│   ├── images/                   # 시각화 자료
│   │   ├── dual_head_weight_analysis.png
│   │   ├── experimental_weight_validation.png
│   │   ├── loss_weight_justification.png
│   │   ├── numerical_validation.png
│   │   └── quantization_level_analysis.png
│   │
│   ├── reference_scripts/        # 분석 스크립트 저장소 (9개)
│   │   ├── analyze_dual_head_loss.py
│   │   ├── validate_loss_weight_numerically.py
│   │   ├── analyze_training_range_effects.py
│   │   ├── analyze_min_depth_effects.py
│   │   ├── visualize_consistency_and_48.py
│   │   ├── experimental_weight_validation.py
│   │   ├── analyze_loss_weight_justification.py
│   │   ├── test_st2_implementation.py
│   │   └── test_integration_training.py
│   │
│   └── legacy/                   # 레거시 분석 문서
│       ├── GT_DEPTH_ANALYSIS_*.md
│       ├── INT8_*.md
│       └── USE_LOG_SPACE_ANALYSIS.md
│
├── implementation/                # 🏗️ 구현 문서
│   ├── README.md                 # 구현 가이드
│   ├── CODEBASE_CLEANUP_PLAN.md  # 정리 계획 (참고용)
│   ├── FILE_CATEGORIZATION.md    # 파일 분류 (참고용)
│   ├── ST2_FINAL_VALIDATION_REPORT.md
│   │
│   ├── Dual-Head/               # Dual-Head 구현 문서
│   │   ├── DUAL_HEAD_LOSS_WEIGHT_JUSTIFICATION.md
│   │   ├── DUAL_HEAD_WEIGHT_NECESSITY_ANALYSIS.md
│   │   ├── DUAL_HEAD_DOCUMENTATION_SUMMARY.md
│   │   ├── DUAL_HEAD_OUTPUT_STRUCTURE.md
│   │   ├── DUAL_HEAD_OUTPUT_SUMMARY.md
│   │   ├── DUAL_HEAD_SAVE_REPORT.md
│   │   ├── EVALUATE_DUAL_HEAD.md
│   │   └── PTQ_DUAL_HEAD_DESIGN.md
│   │
│   ├── Direct-Depth/            # Direct Depth 구현 문서
│   │   ├── DIRECT_DEPTH_BUG_FIX.md
│   │   ├── DIRECT_DEPTH_MODEL_ARCHITECTURE.md
│   │   ├── DIRECT_DEPTH_OUTPUT_PLAN.md
│   │   └── DUAL_HEAD_ONNX_CONVERSION.md
│   │
│   ├── INT8-Quantization/       # INT8 양자화 문서
│   │   ├── INT8_*.md (6개 파일)
│   │   └── LOG_SPACE_IMPLEMENTATION_COMPLETE.md
│   │
│   └── Range-Adjustment/        # Range 조정 분석
│       └── RANGE_ADJUSTMENT_ANALYSIS_REPORT.md
│
├── architecture/                  # 🏛️ 아키텍처 설명 (생성됨)
│   └── [내용 추가 예정]
│
├── training/                      # 🎓 학습 가이드 (생성됨)
│   └── [내용 추가 예정]
│
├── quantization/                  # 🔢 양자화 가이드 (생성됨)
│   └── [내용 추가 예정]
│
├── reference/                     # 📖 참고 자료 (생성됨)
│   ├── papers/                   # 논문/PDF 저장소
│   └── [링크 추가 예정]
│
└── futurework/                    # 🚀 향후 계획
    └── [기존 내용 유지]
```

---

## 📊 /analysis_results 폴더 구조

```
analysis_results/
├── batch_process.log             # 배치 처리 로그
├── fast_test_dual_head.log       # Dual-Head 테스트 로그
├── ncdb_test_fp32_results.json   # FP32 테스트 결과
├── ncdb_test_fp32_results_CORRECTED.json
└── ncdb_test_fp32_results_FINAL.json
```

**용도**: 분석 실행 결과 저장 (임시)

---

## 🎯 사용자별 문서 네비게이션

### 👤 신규 사용자 (처음 시작)
1. `docs/README.md` - 전체 개요
2. `docs/architecture/` - 아키텍처 이해
3. `docs/training/` - 학습 방법
4. `scripts/train.py` - 학습 시작

### 👨‍💼 구현자 (코드 수정/개선)
1. `docs/implementation/` - 구현 상세
2. `docs/implementation/Dual-Head/` - Dual-Head 구현
3. `packnet_sfm/losses/dual_head_depth_loss.py` - 손실 함수 코드
4. `docs/analysis/` - 분석 자료 참고

### 🔬 연구자 (새로운 기능 추가)
1. `docs/analysis/` - 기존 분석 검토
2. `docs/analysis/reference_scripts/` - 분석 스크립트 실행
3. `docs/quantization/` - 양자화 관련
4. `docs/reference/papers/` - 논문 참고

### 📊 데이터 분석가
1. `docs/analysis/00_INDEX.md` - 분석 색인
2. `analysis_results/` - 기존 결과 검토
3. `docs/analysis/reference_scripts/` - 자신의 데이터로 스크립트 실행

---

## 📝 파일 정리 현황

### ✅ 완료 항목

| 카테고리 | 파일 수 | 위치 | 상태 |
|---------|--------|------|------|
| 분석 문서 (MD) | 13 | `docs/analysis/` | ✅ 정렬 완료 |
| 시각화 (PNG) | 5 | `docs/analysis/images/` | ✅ 정렬 완료 |
| 분석 스크립트 | 9 | `docs/analysis/reference_scripts/` | ✅ 정렬 완료 |
| 구현 문서 (MD) | 31+ | `docs/implementation/` | ✅ 정렬 완료 |
| 로그 파일 | 2 | `analysis_results/` | ✅ 정렬 완료 |
| 결과 JSON | 3 | `analysis_results/` | ✅ 정렬 완료 |

### 🗑️ 삭제된 항목

| 파일명 | 이유 | 상태 |
|--------|------|------|
| test_backward_compatibility.py | 일회성 검증 완료 | ✅ 삭제됨 |
| test_upsample_fix.py | 버그 수정 검증 완료 | ✅ 삭제됨 |
| test_semisup_model_fix.py | 모델 수정 검증 완료 | ✅ 삭제됨 |
| test_sparse_ssi_silog.py | Loss 검증 완료 | ✅ 삭제됨 |

---

## 🔑 주요 파일 위치

### 핵심 모델 코드
```
packnet_sfm/
├── losses/dual_head_depth_loss.py      # Dual-Head 손실 함수
├── networks/depth/ResNetSAN01.py       # Dual-Head 모델
├── models/SemiSupCompletionModel.py    # 학습 모델
└── utils/depth.py                      # 깊이 계산 함수
```

### 학습/평가
```
scripts/
├── train.py                            # 학습 스크립트
├── eval.py                             # 평가 스크립트
└── infer.py                            # 추론 스크립트
```

### 문서 (중요)
```
docs/
├── analysis/00_INDEX.md                # 분석 자료 색인 ⭐
├── analysis/01_WEIGHT_JUSTIFICATION_SUMMARY.md
├── analysis/02_NUMERICAL_VALIDATION.md
├── analysis/03_MIN_DEPTH_EFFECTS.md
├── analysis/04_48_IMPACT.md            # "48이 왜 48?" 답
├── analysis/05_CONSISTENCY_WEIGHT.md
└── implementation/CODEBASE_CLEANUP_PLAN.md
```

---

## 💡 폴더 정리 철학

### 원칙
1. **폴더별 관리**: 관련 파일끼리 같은 폴더에
2. **계층적 구조**: 주제별 → 세부 폴더로 구성
3. **명확한 네이밍**: 파일명으로 내용 유추 가능
4. **우선순위 구분**: 자주 보는 파일 앞에 배치

### 폴더별 용도

| 폴더 | 용도 | 수정 빈도 |
|------|------|----------|
| `packnet_sfm/` | 핵심 모델 코드 | 거의 없음 |
| `scripts/` | 학습/평가 스크립트 | 낮음 |
| `configs/` | 학습 설정 | 중간 |
| `docs/` | 모든 문서 | 중간 |
| `analysis_results/` | 분석 결과 (임시) | 높음 |
| `outputs/` | 모델 출력 | 높음 |

---

## 🚀 다음 단계 (선택사항)

### Phase 2 (향후 개선)
- [ ] 각 섹션 README.md 작성 (architecture/, training/, quantization/)
- [ ] `docs/reference/papers/` 폴더에 논문 링크 추가
- [ ] `docs/training/` 폴더에 학습 튜토리얼 추가
- [ ] Git에 commit 후 push

### Phase 3 (문서화 강화)
- [ ] 이미지 참고 링크를 마크다운에 추가
- [ ] 각 분석 문서에 "폴더 이동됨" 주석 추가
- [ ] 상호참조 링크 정리

---

## ✨ 요약

✅ **모든 파일이 폴더별로 체계적으로 정렬되었습니다!**

- **분석 자료**: `docs/analysis/` (+ images 폴더)
- **구현 문서**: `docs/implementation/` (계층화)
- **분석 결과**: `analysis_results/` (로그 및 JSON)
- **핵심 코드**: 루트에서 제거, 각 모듈에 유지

폴더 구조를 참고하여 필요한 문서를 빠르게 찾을 수 있습니다! 🎉
