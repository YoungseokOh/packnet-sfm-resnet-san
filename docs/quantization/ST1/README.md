# ST1: Advanced PTQ Calibration

**전략 분류**: 데이터 최적화 (Advanced Post-Training Quantization)  
**난이도**: ⭐⭐⭐ (Medium)  
**예상 소요 시간**: 1-2주  
**결과**: 실패 (abs_rel 0.1139 → 개선 없음)

---

## 📋 ST1 개요

**Phase 1: Advanced PTQ Calibration**은 INT8 양자화의 근본적 한계를 해결하기 위한 첫 번째 시도였습니다.

### 목표
- 기존 FP32 모델을 INT8로 변환
- Calibration 데이터 최적화로 양자화 오차 최소화
- 목표 성능: abs_rel < 0.09

### 결과
- **실패**: 100개 → 300개 calibration 이미지 사용했으나 성능 개선 없음
- **원인**: Per-tensor 양자화의 구조적 한계 (±28mm 오차)
- **결론**: 모델 구조 변경 필요 → **ST2 Dual-Head 전략으로 전환**

---

## 📁 폴더 내용

### ST1_action_plan.md
- ST1 전략의 초기 기획 문서
- Calibration 방법론 설계
- 예상 타임라인 및 리소스 계획

### ST1_advanced_PTQ_Calibration.md
- 실제 Calibration 실험 결과
- 다양한 Calibration 기법 시도
- 성능 분석 및 실패 원인 분석

---

## 🔗 관련 문서

### 다음 단계
- **[ST2](../ST2/)**: Dual-Head Architecture (현재 진행 중)
  - Integer-Fractional 분리로 ±2mm 정밀도 달성 목표

### 전략 문서
- **[strategy/](../strategy/)**: 전체 양자화 전략 문서
  - INT8 최적화 전략 개요
  - Phase별 접근 방식 비교

---

## 📊 성능 비교

| Phase | 전략 | abs_rel | rmse | δ<1.25 | 결과 |
|-------|------|---------|------|--------|------|
| Baseline | FP32 | 0.0434 | 0.391m | 0.9759 | - |
| ST1 | Advanced PTQ | 0.1139 | 0.751m | 0.9061 | ❌ 실패 |
| ST2 | Dual-Head INT8 | 목표: 0.055 | 목표: 0.50m | 목표: 0.970 | 진행 중 |

---

## 💡 교훈

1. **Calibration만으로는 한계**: 데이터 최적화로는 구조적 문제를 해결할 수 없음
2. **Per-tensor의 근본적 한계**: 256 레벨로는 넓은 depth 범위를 표현하기 부족
3. **모델 구조 변경 필요**: 양자화 친화적인 아키텍처 설계가 필수적

---

**ST1은 실패했지만, ST2의 토대를 마련한 중요한 단계였습니다.**