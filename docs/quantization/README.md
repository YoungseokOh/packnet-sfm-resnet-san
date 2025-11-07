# 📁 Quantization Documentation

**INT8 양자화 최적화 프로젝트 문서**

---

## 📂 폴더 구조

이 디렉토리는 **INT8 양자화 최적화 프로젝트**의 모든 문서를 체계적으로 정리하고 있습니다.

```
docs/quantization/
├── ST1/                    # Phase 1: Advanced PTQ Calibration
│   ├── README.md
│   ├── ST1_action_plan.md
│   └── ST1_advanced_PTQ_Calibration.md
│
├── ST2/                    # Phase 2: Dual-Head Architecture
│   ├── README.md
│   ├── INDEX.md
│   ├── Quick_Reference.md
│   ├── 01_Overview_Strategy.md
│   ├── 02_Implementation_Guide.md
│   ├── 03_Configuration_Testing.md
│   ├── 04_Training_Evaluation.md
│   ├── 05_Troubleshooting.md
│   └── SPLIT_REPORT.md
│
└── strategy/               # Overall Strategy Documents
    ├── README.md
    ├── INT8_OPTIMIZATION_STRATEGY.md
    ├── INT8_OPTIMIZATION_STRATEGY_backup_v2.md
    └── INT8_OPTIMIZATION_STRATEGY_v2.bak.md
```

---

## 🎯 프로젝트 개요

### 목표
**PackNet-SAN 모델의 INT8 양자화 성능을 FP32 수준으로 향상**

| Metric | FP32 Baseline | INT8 목표 | 현재 상태 |
|--------|---------------|-----------|-----------|
| **abs_rel** | 0.0434 | **< 0.065** | ST2 진행 중 |
| **rmse** | 0.391m | **< 0.55m** | - |
| **δ<1.25** | 0.9759 | **> 0.965** | - |

### 접근 방식
1. **ST1**: 기존 모델 + Advanced Calibration (실패)
2. **ST2**: Dual-Head Architecture (진행 중)

---

## 📖 각 Phase 설명

### 🔍 ST1: Advanced PTQ Calibration
**전략**: 데이터 최적화로 양자화 오차 최소화
- **결과**: 실패 (abs_rel 0.1139, 개선 없음)
- **교훈**: Per-tensor 양자화의 구조적 한계
- **상태**: 완료 (문서화)

### 🚀 ST2: Dual-Head Architecture
**전략**: Integer-Fractional 분리로 정밀도 14배 향상
- **예상**: abs_rel 0.055 (51% 개선)
- **방법**: ±28mm → ±2mm 오차 감소
- **상태**: 구현 중

---

## 📋 문서 이용 가이드

### 처음 방문자
1. **[strategy/README.md](strategy/README.md)** 읽기 - 전체 전략 이해
2. **[ST1/README.md](ST1/README.md)** 읽기 - Phase 1 결과 확인
3. **[ST2/README.md](ST2/README.md)** 읽기 - 현재 진행 상황 파악

### 구현자
1. **[ST2/Quick_Reference.md](ST2/Quick_Reference.md)** - 빠른 참조
2. **[ST2/02_Implementation_Guide.md](ST2/02_Implementation_Guide.md)** - 상세 구현
3. **[ST2/05_Troubleshooting.md](ST2/05_Troubleshooting.md)** - 문제 해결

### 관리자
1. **[ST2/INDEX.md](ST2/INDEX.md)** - 전체 문서 구조
2. **[ST2/SPLIT_REPORT.md](ST2/SPLIT_REPORT.md)** - 문서 관리 내역

---

## 📊 프로젝트 진행 상황

### Phase 1: ST1 (완료)
- ✅ 전략 수립 및 계획
- ✅ Calibration 실험 (300개 이미지)
- ✅ 성능 분석 및 실패 원인 규명
- ✅ ST2 전략으로 전환 결정

### Phase 2: ST2 (진행 중)
- ✅ 전략 설계 및 코드베이스 분석
- ✅ 문서화 및 구현 가이드 작성
- 🔄 코드 구현 시작 예정
- ⏳ 학습 및 평가 진행 예정

---

## 🎯 성공 기준

### 기술적 목표
- ✅ **INT8 abs_rel < 0.065** (FP32의 1.5배 이내)
- ✅ **양자화 오차 < 5mm** (현재 ±28mm → 목표 ±2mm)
- ✅ **Backward Compatibility** (기존 기능 유지)

### 프로젝트 목표
- ✅ **실행 가능한 문서** (Copy-paste 가능한 코드)
- ✅ **체계적인 테스트** (단위/통합 테스트)
- ✅ **문제 해결 가이드** (Troubleshooting 포함)

---

## 🔗 주요 문서 바로가기

### 전략 및 계획
- **[전체 전략](strategy/INT8_OPTIMIZATION_STRATEGY.md)**
- **[ST1 실행 계획](ST1/ST1_action_plan.md)**
- **[ST2 구현 가이드](ST2/02_Implementation_Guide.md)**

### 결과 및 분석
- **[ST1 실험 결과](ST1/ST1_advanced_PTQ_Calibration.md)**
- **[ST2 성능 목표](ST2/04_Training_Evaluation.md)**

### 실용 가이드
- **[빠른 참조](ST2/Quick_Reference.md)**
- **[문제 해결](ST2/05_Troubleshooting.md)**

---

## 📈 다음 단계

### 단기 (1-2주)
- [ ] ST2 Phase 1: DualHeadDepthDecoder 구현
- [ ] ST2 Phase 2: Helper Functions 추가
- [ ] ST2 Phase 3: ResNetSAN01 통합

### 중기 (3-4주)
- [ ] ST2 Phase 4: Loss Function 구현
- [ ] ST2 Phase 5: 학습 및 평가
- [ ] NPU 변환 및 INT8 평가

### 장기 (5-6주)
- [ ] 성능 목표 달성 검증
- [ ] 프로덕션 배포 준비
- [ ] 최종 문서 정리

---

## 👥 담당자

- **전략 수립**: 양자화 최적화 방향성
- **구현**: ST2 Dual-Head 코드 개발
- **평가**: 성능 측정 및 분석
- **문서화**: 구현 가이드 및 Troubleshooting

---

## 💡 팁

- **문서 탐색**: 각 폴더의 `README.md`부터 읽어보세요
- **코드 구현**: `ST2/Quick_Reference.md`를 참고하세요
- **문제 발생**: `ST2/05_Troubleshooting.md`를 먼저 확인하세요

---

**이 프로젝트는 INT8 양자화의 한계를 극복하고, NPU에서 고성능 depth estimation을 실현하는 것을 목표로 합니다.** 🚀