# Strategy: INT8 Quantization Optimization

**전략 문서**: 전체 양자화 최적화 로드맵 및 방법론

---

## 📋 전략 문서 개요

이 폴더에는 INT8 양자화 최적화의 **전체 전략과 방법론**에 대한 문서들이 포함되어 있습니다.

### 목적
- 양자화 최적화의 전체적인 방향성 제시
- 각 Phase의 전략 비교 및 선택 근거
- 기술적 의사결정 프로세스 문서화

---

## 📁 폴더 내용

### INT8_OPTIMIZATION_STRATEGY.md
- **메인 전략 문서**: 전체 양자화 최적화 로드맵
- Phase별 접근 방식 비교
- 기술적 의사결정 및 타협점 분석

### INT8_OPTIMIZATION_STRATEGY_backup_v2.md
- 전략 문서의 백업 버전 (v2)
- 이전 버전의 전략 비교용

### INT8_OPTIMIZATION_STRATEGY_v2.bak.md
- 전략 문서의 추가 백업
- 버전 관리 및 변경 이력 추적용

---

## 🔗 전략 단계별 연결

```
전체 전략 (strategy/)
    │
    ├── Phase 1: ST1 (ST1/)
    │   ├── ST1_action_plan.md
    │   └── ST1_advanced_PTQ_Calibration.md
    │       └── 결과: 실패 (Calibration 한계)
    │
    └── Phase 2: ST2 (ST2/)
        ├── README.md
        ├── 01_Overview_Strategy.md
        ├── 02_Implementation_Guide.md
        ├── 03_Configuration_Testing.md
        ├── 04_Training_Evaluation.md
        └── 05_Troubleshooting.md
            └── 목표: Dual-Head로 ±2mm 정밀도
```

---

## 📊 전략 발전 과정

### Phase 1: Advanced PTQ (ST1)
- **접근**: 기존 모델 + Calibration 최적화
- **기대**: 최소한의 변경으로 INT8 성능 향상
- **결과**: 실패 (abs_rel 0.1139, 개선 없음)
- **교훈**: 구조적 문제는 데이터로 해결 불가

### Phase 2: Dual-Head Architecture (ST2)
- **접근**: 양자화 친화적인 모델 구조 변경
- **기대**: Integer-Fractional 분리로 정밀도 14배 향상
- **진행**: 구현 중
- **목표**: abs_rel 0.055 (51% 개선)

---

## 🎯 전략적 의사결정

### 왜 ST1부터 시작했는가?
1. **리스크 최소화**: 기존 모델 유지, 실패 시 즉시 rollback 가능
2. **학습 기회**: 양자화의 실제 한계 체득
3. **베이스라인 확보**: ST2 효과를 명확히 측정하기 위함

### 왜 ST2로 전환했는가?
1. **ST1 실패**: Calibration으로는 근본적 문제 해결 불가
2. **기술적 근거**: Per-tensor 양자화의 수학적 한계
3. **NPU 활용**: Dual-Output 기능으로 추가 비용 없이 해결 가능

---

## 📈 예상 성능 궤적

```
FP32 Baseline: abs_rel 0.0434
                    │
                    │ ST1 시도 (실패)
                    │ abs_rel 0.1139 (2.6배 악화)
                    │
                    └─ ST2 목표
                       abs_rel 0.055 (1.5배 격차, 51% 개선)
```

---

## 🔗 관련 문서

### 구현 문서
- **[ST1](../ST1/)**: Phase 1 실행 결과
- **[ST2](../ST2/)**: Phase 2 구현 가이드

### 참고 자료
- NPU 양자화 가이드라인
- 관련 논문 및 기술 보고서

---

## 💡 전략적 교훈

1. **점진적 접근**: 큰 변경 전에 작은 최적화부터 시도
2. **실험적 검증**: 이론적 근거만으로 결정하지 말고 실제 테스트
3. **Fallback 계획**: 각 단계에서 실패 시 다음 단계로의 명확한 경로
4. **문서화 중요성**: 의사결정 과정과 실패 원인을 철저히 기록

---

**이 전략 문서들은 INT8 양자화 최적화의 전체 여정을 기록하고 있습니다.**