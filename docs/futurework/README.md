# FutureWork Documentation

**목적**: ST2 구현 이후 개선 사항 및 근본 원인 해결 방안 문서화  
**시작일**: 2024-12-19  
**상태**: 📋 Planning & Documentation  

---

## 📋 현재 문서

### 1. CHECKPOINT_MONITORING_ROOT_CAUSE_FIX.md
**담당자**: World-Class Developer  
**주제**: Checkpoint 모니터링 기능 근본 원인 해결  
**상태**: ✅ 완료 (미적용)  

#### 요약
- **문제**: `save_top_k` 파라미터가 메트릭 부재로 작동 불가
- **근본 원인**: `validation_epoch_end()`에서 `'loss'` 메트릭 미반환
- **해결책**: 2단계 근본 원인 해결 (방안 A, B)
- **영향도**: 🔧 Code Quality + 기능성 향상

#### 포함 내용
- [x] 문제 분석 (4단계 추적)
- [x] 2가지 해결 방안 상세 설명
- [x] 방안별 장단점 비교
- [x] 적용 계획 (Phase 1-3)
- [x] 테스트 계획
- [x] 체크리스트

**읽을 시간**: ~15분  
**적용 난이도**: 🟡 Medium

---

### 2. IMPLEMENTATION_GUIDE.md
**담당자**: World-Class Developer  
**주제**: 근본 원인 해결 임플리멘테이션 가이드  
**상태**: ✅ 완료 (즉시 사용 가능)  

#### 요약
- **목표**: Phase 1, 2 구현을 위한 단계별 가이드
- **Phase 1**: YAML 수정 (5분, 즉시 적용)
- **Phase 2**: 코드 수정 (15분, 테스트 후 적용)
- **검증**: 명령어 및 테스트 방법 포함

#### 포함 내용
- [x] Phase 1 YAML 수정 전/후 코드
- [x] Phase 2 코드 수정 상세 위치
- [x] 검증 명령어 (5개)
- [x] 테스트 계획
- [x] 적용 순서표
- [x] 커밋 메시지 템플릿

**읽을 시간**: ~5분 (빠른 참고용)  
**적용 난이도**: 🟢 Easy (Phase 1), 🟡 Medium (Phase 2)

---

## 🗂️ 폴더 구조

```
docs/futurework/
├── README.md (this file)
├── CHECKPOINT_MONITORING_ROOT_CAUSE_FIX.md
│   └── 상세 분석, 해결책, 계획
└── IMPLEMENTATION_GUIDE.md
    └── 단계별 구현 가이드
```

---

## 🚀 빠른 시작

### 상황 1: 문제를 이해하고 싶은 경우
```
→ CHECKPOINT_MONITORING_ROOT_CAUSE_FIX.md 읽기
  섹션: "🔍 문제 분석" + "💡 해결 방안"
  소요 시간: ~10분
```

### 상황 2: 즉시 적용하고 싶은 경우
```
→ IMPLEMENTATION_GUIDE.md 읽기
  섹션: "🚀 Phase 1: YAML 수정"
  소요 시간: ~1분 (실행), ~2분 (검증)
```

### 상황 3: 전체 계획을 알고 싶은 경우
```
→ CHECKPOINT_MONITORING_ROOT_CAUSE_FIX.md 읽기
  섹션: "🎯 권장 적용 계획" + "📝 실행 시 주의사항"
  소요 시간: ~20분
```

---

## 📊 현황 요약

### 현재 상태
| 항목 | 상태 | 비고 |
|------|------|------|
| 문제 분석 | ✅ 완료 | 4단계 근본 원인 추적 |
| 해결책 수립 | ✅ 완료 | 2가지 방안 제시 |
| 문서화 | ✅ 완료 | 상세 가이드 작성 |
| 코드 적용 | ⏳ 미적용 | Phase 1/2 계획됨 |

### 추천 다음 단계

```
[지금]
  ↓
Phase 1 적용 (YAML)
  ├─ 예상 시간: 5분
  ├─ 리스크: 최소
  └─ 효과: save_top_k 작동
      ↓
Phase 1 검증 (간단한 학습 테스트)
  ├─ 예상 시간: 30분
  ├─ 확인 항목: checkpoint 생성
  └─ 성공 신호: 3개 파일만 유지
      ↓
Phase 2 준비 (코드 리뷰)
  ├─ 예상 시간: 2시간
  ├─ 담당자: Lead + Dev
  └─ 결과: Code changes approved
      ↓
Phase 2 적용 (코드 수정)
  ├─ 예상 시간: 2시간
  ├─ 테스트: Integration test
  └─ 결과: 'loss' 메트릭 추가
```

---

## 💡 주요 통찰

### 발견사항
1. **메트릭 선택의 유연성**
   - YAML에서 다양한 메트릭 사용 가능
   - 'depth-abs_rel0' 즉시 사용 가능
   - 'loss' 추가 시 default_config와 호환

2. **근본 원인의 명확성**
   - validation_epoch_end() 체인 4단계 추적 완료
   - 각 단계의 책임 명확
   - 해결책 2가지 모두 타당성 있음

3. **개선 기회**
   - 기존 우회책(save_top_k=-1) 대비 80% 디스크 절약
   - 코드 품질 향상 기회
   - 다른 모델에도 자동 적용 가능

---

## 🔗 관련 문서

**ST2 Dual-Head 구현**:
- [ST2_Integer_Fractional_Dual_Head.md](../quantization/ST2/ST2_Integer_Fractional_Dual_Head.md)
- [PM_FINAL_VALIDATION_REPORT.md](../quantization/ST2/PM_FINAL_VALIDATION_REPORT.md)
- [DEVELOPER_CHECKPOINT_ANALYSIS.md](../quantization/ST2/DEVELOPER_CHECKPOINT_ANALYSIS.md)

**구현 코드**:
- packnet_sfm/models/model_wrapper.py (수정 대상)
- packnet_sfm/models/model_checkpoint.py (참고)
- configs/train_resnet_san_ncdb_dual_head_640x384.yaml (YAML)

---

## ✅ 체크리스트

### 문서 검수
- [x] 근본 원인 분석 완료
- [x] 2가지 해결책 검토됨
- [x] 구현 가이드 작성됨
- [x] 테스트 계획 수립됨
- [x] 커밋 메시지 준비됨

### 적용 준비
- [ ] Phase 1 (YAML) 검증
- [ ] Phase 1 테스트 학습 실행
- [ ] Phase 2 코드 리뷰
- [ ] Phase 2 구현
- [ ] Integration 테스트
- [ ] PR 병합

---

## 📞 문의 및 피드백

**문서 작성**: World-Class Developer  
**검수 필요시**: Lead/PM  
**피드백**: 각 섹션 하단의 주석 참고  

---

## 📝 버전 관리

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| 1.0 | 2024-12-19 | 초기 문서 작성 |
| - | - | - |

---

**Status**: ✅ Documentation Complete  
**Next Review**: After Phase 1 implementation  
**Last Updated**: 2024-12-19

---

## 🎯 FutureWork 개요

이 폴더는 ST2 Dual-Head 구현 완료 후 **근본 원인 해결** 및 **코드 품질 향상**을 위한 우선순위 작업들을 문서화합니다.

**핵심 가치**:
- ✅ 명확한 근본 원인 분석
- ✅ 2가지 선택지 제시
- ✅ 즉시 적용 가능한 가이드
- ✅ 테스트 및 검증 계획
- ✅ Production-ready 문서

**이 문서가 없다면**: 
- ❌ 나중에 같은 문제로 디버깅
- ❌ 해결책 전수 어려움
- ❌ 코드 품질 개선 기회 놓침

**이 문서를 통해**:
- ✅ 언제든 적용 가능
- ✅ 새 팀원도 빠르게 이해
- ✅ 장기적 코드 품질 보증
