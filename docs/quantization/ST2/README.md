# ST2: Integer-Fractional Dual-Head Architecture

**전략 분류**: 모델 구조 변경 (Parameter-driven Decoder Extension)  
**난이도**: ⭐⭐⭐⭐ (High - 재학습 필요)  
**예상 소요 시간**: 2-3주  
**예상 성능 개선**: abs_rel 0.1139 → **0.055** (51% 개선)  
**날짜**: 2025-11-07  
**문서 버전**: 2.0 (코드베이스 분석 반영)

---

## 🎯 핵심 설계 원칙

**✅ 기존 기능 보존 (Backward Compatibility)**:
- 모든 기존 기능(`use_film`, `use_enhanced_lidar` 등) 100% 유지
- Single-Head 모델과 Dual-Head 모델이 동일 코드베이스에서 YAML만으로 전환 가능
- 기존 checkpoint 호환성 보장

**✅ Parameter-driven 설계**:
- 새 모델 클래스 생성 **없음** (유지보수 악몽 방지)
- Decoder만 조건부 교체 (Factory Pattern)
- YAML config로 모든 동작 제어

---

## 📚 문서 구조

이 문서는 여러 파일로 분리되어 있습니다:

### 1. [개요 및 전략](01_Overview_Strategy.md)
- Phase 1 결과 분석
- 코드베이스 구조 분석
- 설계 결정: 확장 vs 신규 생성
- 기술적 배경 및 아키텍처 설계

### 2. [구현 가이드](02_Implementation_Guide.md)
- Phase 1: DualHeadDepthDecoder 구현
- Phase 2: Helper Functions
- Phase 3: ResNetSAN01 확장
- Phase 4: Loss Function 구현
- Phase 5: Model Wrapper 통합

### 3. [설정 및 테스트](03_Configuration_Testing.md)
- YAML Configuration
- 단위 테스트
- 통합 테스트
- Backward Compatibility 검증

### 4. [학습 및 평가](04_Training_Evaluation.md)
- 학습 실행
- 학습 모니터링
- 평가 프로세스
- 예상 결과

### 5. [Troubleshooting](05_Troubleshooting.md)
- 학습 중 문제
- 코드 통합 문제
- NPU 변환 문제

---

## 🚀 Quick Start

### 최소 변경 요약

| 파일 | 변경 유형 | 줄 수 |
|------|-----------|-------|
| `dual_head_depth_decoder.py` | 🆕 신규 | ~150줄 |
| `layers.py` | ➕ 함수 추가 | +40줄 |
| `ResNetSAN01.py` | ➕ 로직 추가 | +30줄 |
| `dual_head_depth_loss.py` | 🆕 신규 | ~120줄 |
| `SemiSupCompletionModel.py` | ➕ 분기 추가 | +20줄 |
| **Total** | - | **~360줄** |

### 다음 단계

**Week 1** (Day 1-5):
- [ ] Day 1: `DualHeadDepthDecoder` 구현 및 테스트
- [ ] Day 2: Helper functions 및 단위 테스트
- [ ] Day 3: `ResNetSAN01` 통합 및 통합 테스트
- [ ] Day 4: Loss function 구현 및 검증
- [ ] Day 5: YAML config 준비 및 학습 시작

**Week 2-3** (학습 및 평가):
- [ ] Week 2: 모델 학습 (30 epochs)
- [ ] Week 3: FP32 평가, NPU 변환, INT8 평가

### Success Criteria
- ✅ 모든 단위 테스트 통과
- ✅ Backward compatibility 검증
- ✅ FP32 abs_rel < 0.045
- ✅ **INT8 abs_rel < 0.065** (목표)

---

**이 문서는 코드베이스를 깊이 분석한 후 작성되었으며, 기존 기능을 해치지 않고 안전하게 Dual-Head를 통합하는 실무적인 가이드를 제공합니다.**
