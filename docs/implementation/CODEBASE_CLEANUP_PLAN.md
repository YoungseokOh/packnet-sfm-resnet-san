# 코드베이스 정리 및 구조 재설계

## 현재 상태 분석

### ✅ 사용 중인 파일들 (절대 삭제 금지)

```
packnet_sfm/                         # ← 핵심 코드
├── losses/
│   ├── dual_head_depth_loss.py     ✅ 사용 중 (Dual-Head Loss)
│   ├── ssi_silog_loss.py           ✅ 사용 중 (대체 Loss)
│   └── ...
├── models/
│   ├── model_wrapper.py            ✅ 사용 중 (추론/평가)
│   ├── SemiSupCompletionModel.py   ✅ 사용 중 (학습)
│   └── ...
├── networks/
│   └── depth/
│       ├── ResNetSAN01.py          ✅ 사용 중 (Dual-Head 모델)
│       ├── YOLOv8SAN01.py          ✅ 사용 중 (대체 모델)
│       └── ...
└── ...

scripts/                            # ← 필수 스크립트
├── train.py                        ✅ 사용 중 (학습)
├── infer.py                        ✅ 사용 중 (추론)
├── eval.py                         ✅ 사용 중 (평가)
├── eval_official.py                ✅ 사용 중 (공식 평가)
└── ...

configs/                            # ← 학습 설정
├── train_resnet_san_ncdb_dual_head_640x384.yaml  ✅ 사용 중
├── train_resnet_san_kitti.yaml     ✅ 사용 중
└── ...
```

---

### ⚠️ 분석/테스트 파일들 (정리 필요)

#### 즉시 정리 (더 이상 필요 없음)

```
❌ test_backward_compatibility.py      # 구 버전 호환성 테스트 (이제 불필요)
❌ test_upsample_fix.py                # 업샘플 버그 수정 검증 (완료)
❌ test_semisup_model_fix.py           # 세미슈퍼바이즈드 모델 수정 (완료)
❌ test_sparse_ssi_silog.py            # Sparse Silog Loss 테스트 (완료)
```

**이유**: 한 번 수행했던 검증 테스트이며, 현재 코드에서 통과 확인됨

---

#### 문서 폴더로 이동 (reference/analysis)

```
✅ → 이동 (reference 목적)
  - experimental_weight_validation.py
  - analyze_loss_weight_justification.py
  - analyze_dual_head_loss.py
  - validate_loss_weight_numerically.py
  - analyze_min_depth_effects.py
  - analyze_training_range_effects.py
  - visualize_consistency_and_48.py
  - test_st2_implementation.py
  - test_integration_training.py

이유: 분석/검증 목적 스크립트 (학습/추론에 필요 없음)
새로운 사람이 Dual-Head 이해할 때 참고할 reference
```

---

## 새로운 폴더 구조 (제안)

### 루트 폴더

```
/workspace/packnet-sfm/
├── packnet_sfm/                    # ← 핵심 모델/손실 코드 (변경X)
│   ├── losses/
│   ├── models/
│   ├── networks/
│   └── ...
│
├── scripts/                        # ← 학습/추론 스크립트 (변경X)
│   ├── train.py
│   ├── infer.py
│   ├── eval.py
│   └── ...
│
├── configs/                        # ← 학습 설정 파일 (변경X)
│   ├── train_resnet_san_ncdb_dual_head_640x384.yaml
│   └── ...
│
├── docs/                           # ← 문서 (재구조화!)
│   ├── README.md                   # ← 메인 문서
│   │
│   ├── architecture/               # ← 아키텍처 설명
│   │   ├── 00_OVERVIEW.md          # 전체 개요
│   │   ├── 01_DUAL_HEAD_DESIGN.md
│   │   ├── 02_INTEGER_FRACTIONAL.md
│   │   └── 03_MODEL_VARIANTS.md    # (ResNetSAN, YOLOv8SAN)
│   │
│   ├── training/                   # ← 학습 관련
│   │   ├── 00_GETTING_STARTED.md   # 학습 방법
│   │   ├── 01_LOSS_FUNCTION.md
│   │   ├── 02_WEIGHT_SELECTION.md  # fractional_weight=10.0 설명
│   │   ├── 03_HYPERPARAMETERS.md   # min_depth, max_depth 등
│   │   └── 04_TRAINING_TIPS.md     # 팁/문제 해결
│   │
│   ├── quantization/               # ← 양자화 관련
│   │   ├── 00_PTQ_OVERVIEW.md
│   │   ├── 01_ST2_INTEGER_FRACTIONAL.md
│   │   ├── 02_DEPLOYMENT.md
│   │   └── 03_CALIBRATION.md
│   │
│   ├── analysis/                   # ← 분석/검증 자료 (새로운 폴더!)
│   │   ├── 00_INDEX.md             # 분석 자료 가이드
│   │   ├── 01_WEIGHT_JUSTIFICATION.md     # 왜 10.0?
│   │   ├── 02_RANGE_EFFECTS.md     # max_depth 영향 분석
│   │   ├── 03_MIN_DEPTH_EFFECTS.md # min_depth 영향 분석
│   │   ├── 04_48_IMPACT.md         # 48 레벨 영향 분석
│   │   ├── 05_CONSISTENCY_WEIGHT.md # consistency_weight 분석
│   │   └── reference_scripts/      # 분석 스크립트 저장소
│   │       ├── analyze_dual_head_loss.py
│   │       ├── validate_loss_weight_numerically.py
│   │       ├── analyze_training_range_effects.py
│   │       ├── analyze_min_depth_effects.py
│   │       ├── visualize_consistency_and_48.py
│   │       └── experimental_weight_validation.py
│   │
│   └── reference/                  # ← 참고 자료
│       ├── 01_KITTI_DATASET.md
│       ├── 02_NCDB_DATASET.md
│       └── papers/                 # 관련 논문 요약
│
├── analysis_results/               # ← 분석 결과 저장 (새로운 폴더!)
│   ├── loss_weight_analysis/
│   ├── range_effects/
│   ├── consistency_analysis/
│   └── README.md
│
├── outputs/                        # ← 모델 출력/시각화 (기존)
│   ├── depth/
│   ├── comparison/
│   └── ...
│
└── README.md                       # ← 루트 가이드
```

---

## 🎯 실행 계획

### Phase 1: 안전한 정리 (이번 단계)

```
1. 불필요한 테스트 파일 확인
   - test_backward_compatibility.py  → 삭제 후보
   - test_upsample_fix.py           → 삭제 후보
   - test_semisup_model_fix.py      → 삭제 후보
   - test_sparse_ssi_silog.py       → 삭제 후보

2. 분석 스크립트 보관
   → docs/analysis/reference_scripts/ 폴더로 이동
   - analyze_dual_head_loss.py
   - validate_loss_weight_numerically.py
   - analyze_training_range_effects.py
   - analyze_min_depth_effects.py
   - visualize_consistency_and_48.py
   - experimental_weight_validation.py

3. 분석 결과 정리
   → docs/analysis/ 폴더로 마크다운 이동/정리
   - ANALYSIS_48_IMPACT_ON_TRAINING.md
   - RANGE_ADJUSTMENT_ANALYSIS_REPORT.md
   - MIN_DEPTH_EFFECTS_ANALYSIS_REPORT.md
   - CONSISTENCY_WEIGHT_AND_48_LEVELS_EXPLANATION.md
   - NUMERICAL_VALIDATION_RESULTS.md
```

### Phase 2: 문서 구조화 (다음 단계)

```
1. docs/ 폴더 재구조화
   - architecture/, training/, quantization/, analysis/ 생성
   - 기존 파일 재정렬

2. 각 폴더별 00_INDEX.md 작성
   - 해당 폴더의 문서들을 한눈에 볼 수 있게

3. 루트 README.md 업데이트
   - 전체 구조 설명
```

---

## 📋 파일별 최종 판단

| 파일 | 용도 | 상태 | 조치 |
|------|------|------|------|
| `test_backward_compatibility.py` | 구버전 호환성 | ✅ 완료 | 🗑️ 삭제 |
| `test_upsample_fix.py` | 업샘플 검증 | ✅ 완료 | 🗑️ 삭제 |
| `test_semisup_model_fix.py` | 모델 수정 검증 | ✅ 완료 | 🗑️ 삭제 |
| `test_sparse_ssi_silog.py` | Loss 검증 | ✅ 완료 | 🗑️ 삭제 |
| `test_st2_implementation.py` | ST2 구현 검증 | ⚠️ 참고용 | 📁 이동 |
| `test_integration_training.py` | 통합 테스트 | ⚠️ 참고용 | 📁 이동 |
| `analyze_dual_head_loss.py` | Dual-Head 분석 | ✅ 유용 | 📁 이동 |
| `validate_loss_weight_numerically.py` | 가중치 검증 | ✅ 유용 | 📁 이동 |
| `analyze_training_range_effects.py` | 범위 분석 | ✅ 유용 | 📁 이동 |
| `analyze_min_depth_effects.py` | min_depth 분석 | ✅ 유용 | 📁 이동 |
| `visualize_consistency_and_48.py` | 시각화 생성 | ✅ 유용 | 📁 이동 |
| `experimental_weight_validation.py` | 실험적 검증 | ⚠️ 참고용 | 📁 이동 |
| `analyze_loss_weight_justification.py` | 수학적 증명 | ✅ 유용 | 📁 이동 |

---

## 🚀 정리 명령어 (최종)

```bash
# 1. 폴더 구조 생성
mkdir -p docs/architecture
mkdir -p docs/training
mkdir -p docs/quantization
mkdir -p docs/analysis/reference_scripts
mkdir -p docs/reference/papers
mkdir -p analysis_results

# 2. 안전하지 않은 테스트 파일 (1회성 검증 완료됨)
rm -f test_backward_compatibility.py
rm -f test_upsample_fix.py
rm -f test_semisup_model_fix.py
rm -f test_sparse_ssi_silog.py

# 3. 분석 스크립트 이동 (새로운 reference 폴더)
mv analyze_dual_head_loss.py docs/analysis/reference_scripts/
mv validate_loss_weight_numerically.py docs/analysis/reference_scripts/
mv analyze_training_range_effects.py docs/analysis/reference_scripts/
mv analyze_min_depth_effects.py docs/analysis/reference_scripts/
mv visualize_consistency_and_48.py docs/analysis/reference_scripts/
mv experimental_weight_validation.py docs/analysis/reference_scripts/
mv analyze_loss_weight_justification.py docs/analysis/reference_scripts/
mv test_st2_implementation.py docs/analysis/reference_scripts/
mv test_integration_training.py docs/analysis/reference_scripts/

# 4. 분석 문서 이동
mv ANALYSIS_48_IMPACT_ON_TRAINING.md docs/analysis/04_48_IMPACT.md
mv RANGE_ADJUSTMENT_ANALYSIS_REPORT.md docs/analysis/02_RANGE_EFFECTS.md
mv MIN_DEPTH_EFFECTS_ANALYSIS_REPORT.md docs/analysis/03_MIN_DEPTH_EFFECTS.md
mv CONSISTENCY_WEIGHT_AND_48_LEVELS_EXPLANATION.md docs/analysis/05_CONSISTENCY_WEIGHT.md
```

---

## ✅ 최종 확인 체크리스트

### 삭제 안전성 확인

```
☐ test_backward_compatibility.py
   - import 되는지 확인? NO ✅
   - 학습/추론에 필요? NO ✅
   - git history만 남음 ✅

☐ test_upsample_fix.py
   - import 되는지 확인? NO ✅
   - 학습/추론에 필요? NO ✅
   - git history만 남음 ✅

☐ test_semisup_model_fix.py
   - import 되는지 확인? NO ✅
   - 학습/추론에 필요? NO ✅
   - git history만 남음 ✅

☐ test_sparse_ssi_silog.py
   - import 되는지 확인? NO ✅
   - 학습/추론에 필요? NO ✅
   - git history만 남음 ✅
```

### 이동 안전성 확인

```
☐ analyze_dual_head_loss.py
   - import 되는지? NO ✅
   - 실행할 때만 사용? YES ✅
   - 이동 후 경로 명확? YES ✅

☐ visualize_consistency_and_48.py
   - import 되는지? NO ✅
   - 필요할 때 실행? YES ✅
   - 원본 파일로 복구 가능? YES (git) ✅
```

---

## 📝 최종 상태

### Before
```
/workspace/packnet-sfm/
├── packnet_sfm/          ✅ (유지)
├── scripts/              ✅ (유지)
├── configs/              ✅ (유지)
├── docs/                 ⚠️ (정돈 필요)
├── 13개 .py 파일 (혼재) ❌ (정리 필요)
└── *.md 파일들 (흩어짐) ❌ (정리 필요)
```

### After
```
/workspace/packnet-sfm/
├── packnet_sfm/          ✅ (핵심 코드)
├── scripts/              ✅ (필수 스크립트)
├── configs/              ✅ (학습 설정)
├── docs/                 ✅ (체계적 문서)
│   ├── architecture/
│   ├── training/
│   ├── quantization/
│   ├── analysis/         ← 새로운 분석 섹션
│   └── reference/
├── analysis_results/     ← 새로운 결과 폴더
└── README.md             ✅ (루트 가이드)
```

