# ST2 문서 인덱스

**Integer-Fractional Dual-Head Architecture 구현 가이드**

---

## 📚 문서 구조 한눈에 보기

### 1️⃣ 개요 및 전략
**[01_Overview_Strategy.md](01_Overview_Strategy.md)** - 18KB

**읽어야 할 때**:
- 프로젝트를 처음 시작하는 경우
- 왜 Dual-Head가 필요한지 이해하고 싶을 때
- 코드베이스 구조를 파악하고 싶을 때

**주요 내용**:
- Phase 1 실패 분석 (Calibration 한계)
- INT8 양자화의 근본적 문제 (±28mm 오차)
- Dual-Head로 ±2mm 달성 원리
- 코드베이스 분석 (ResNetSAN01 구조)
- 설계 결정: 확장 vs 신규 생성

---

### 2️⃣ 구현 가이드
**[02_Implementation_Guide.md](02_Implementation_Guide.md)** - 25KB

**읽어야 할 때**:
- 실제 코드를 작성하기 시작할 때
- 각 컴포넌트의 상세 구현을 확인하고 싶을 때

**주요 내용**:
- **Phase 1**: DualHeadDepthDecoder 구현 (~150줄)
- **Phase 2**: Helper Functions (+40줄)
- **Phase 3**: ResNetSAN01 확장 (+30줄)
- **Phase 4**: Loss Function (~120줄)
- **Phase 5**: Model Wrapper 통합 (+20줄)

**모든 코드는 복사-붙여넣기 가능하도록 완전한 형태로 제공됨**

---

### 3️⃣ 설정 및 테스트
**[03_Configuration_Testing.md](03_Configuration_Testing.md)** - 15KB

**읽어야 할 때**:
- YAML 설정 방법을 확인하고 싶을 때
- 구현한 코드를 테스트하고 싶을 때
- Backward compatibility를 검증하고 싶을 때

**주요 내용**:
- YAML 설정 (Single-Head, Dual-Head, 하이브리드)
- 단위 테스트 (Decoder, Helper functions)
- 통합 테스트 (전체 모델)
- pytest 기반 상세 테스트 케이스
- 빠른 CLI 테스트 명령어

---

### 4️⃣ 학습 및 평가
**[04_Training_Evaluation.md](04_Training_Evaluation.md)** - 12KB

**읽어야 할 때**:
- 모델 학습을 시작하려는 경우
- 학습 진행을 모니터링하고 싶을 때
- NPU 변환 및 평가 방법을 확인하고 싶을 때

**주요 내용**:
- 학습 실행 명령어
- Epoch별 예상 Loss 값
- TensorBoard 모니터링
- FP32/INT8 평가 프로세스
- ONNX export 방법
- NPU 변환 및 평가
- 예상 성능 결과

---

### 5️⃣ 문제 해결
**[05_Troubleshooting.md](05_Troubleshooting.md)** - 18KB

**읽어야 할 때**:
- 문제가 발생했을 때 (필수!)
- 학습 중 이상 신호가 보일 때
- NPU 변환이 실패할 때

**주요 내용**:
- **학습 문제**: NaN loss, Integer loss 높음, 학습 속도 느림
- **코드 통합 문제**: ModuleNotFoundError, KeyError, Checkpoint 로딩 실패
- **NPU 변환 문제**: ONNX export 실패, 양자화 오류, 평가 결과 이상
- 각 문제에 대한 구체적 해결 방법 및 검증 코드

---

## 🚀 Quick Access

### 빠른 시작 가이드
**[README.md](README.md)** - 5KB
- 전체 프로젝트 개요
- 핵심 설계 원칙
- Quick Start 체크리스트

### 빠른 참조
**[Quick_Reference.md](Quick_Reference.md)** - 8KB
- 성능 목표 요약
- 구현 체크리스트
- YAML 설정 템플릿
- 빠른 테스트 명령어
- 디버깅 우선순위
- 파일 위치 맵

---

## 📖 읽는 순서 추천

### 🎯 처음 시작하는 경우

```
1. README.md (5분)
   └─> 프로젝트 전체 이해
   
2. 01_Overview_Strategy.md (20분)
   └─> 왜 Dual-Head인지 이해
   
3. 02_Implementation_Guide.md (60분)
   └─> 코드 작성 시작
   
4. 03_Configuration_Testing.md (30분)
   └─> 테스트 및 검증
```

### ⚙️ 구현 중인 경우

```
Quick_Reference.md (항상 옆에 두고 참조)
  ├─> 필요 시 02_Implementation_Guide.md
  └─> 문제 발생 시 05_Troubleshooting.md
```

### 🔧 문제 해결이 필요한 경우

```
05_Troubleshooting.md (즉시)
  └─> 해결 안 되면 관련 섹션으로 이동
      ├─> 학습 문제 → 04_Training_Evaluation.md
      ├─> 코드 문제 → 02_Implementation_Guide.md
      └─> 설정 문제 → 03_Configuration_Testing.md
```

---

## 📊 문서 통계

| 문서 | 크기 | 주요 코드 블록 | 테스트 코드 |
|------|------|----------------|-------------|
| 01_Overview_Strategy.md | 18KB | 10개 | 0개 |
| 02_Implementation_Guide.md | 25KB | 15개 | 5개 |
| 03_Configuration_Testing.md | 15KB | 8개 | 12개 |
| 04_Training_Evaluation.md | 12KB | 5개 | 0개 |
| 05_Troubleshooting.md | 18KB | 20개 | 15개 |
| **Total** | **88KB** | **58개** | **32개** |

**총 코드 라인**: ~1,500줄 (구현 코드 + 테스트 코드)

---

## 🎯 각 Phase별 필수 문서

### Phase 1: 계획 및 이해
- ✅ README.md
- ✅ 01_Overview_Strategy.md
- ✅ Quick_Reference.md

### Phase 2: 구현
- ✅ 02_Implementation_Guide.md (매우 중요!)
- ✅ Quick_Reference.md (항상 참조)

### Phase 3: 테스트
- ✅ 03_Configuration_Testing.md
- ✅ 05_Troubleshooting.md (문제 발생 시)

### Phase 4: 학습 및 평가
- ✅ 04_Training_Evaluation.md
- ✅ 05_Troubleshooting.md (문제 발생 시)

---

## 🔗 상호 참조 맵

```
README.md
├─> 01_Overview_Strategy.md (전체 배경)
├─> 02_Implementation_Guide.md (구현 시작)
└─> Quick_Reference.md (빠른 참조)

01_Overview_Strategy.md
└─> 02_Implementation_Guide.md (구현 방법)

02_Implementation_Guide.md
├─> 03_Configuration_Testing.md (테스트)
└─> 05_Troubleshooting.md (문제 해결)

03_Configuration_Testing.md
├─> 04_Training_Evaluation.md (학습)
└─> 05_Troubleshooting.md (문제 해결)

04_Training_Evaluation.md
└─> 05_Troubleshooting.md (문제 해결)
```

---

## 💡 문서 활용 팁

### 코드 작성 중
→ **Quick_Reference.md**를 화면 한쪽에 띄워두고 작업

### 테스트 실패 시
→ **05_Troubleshooting.md**에서 증상으로 검색

### 학습 모니터링 시
→ **04_Training_Evaluation.md**의 예상 Loss 표와 비교

### 새로운 팀원 온보딩
→ **README.md** → **01_Overview_Strategy.md** 순서로 읽게 함

---

## 📞 도움이 필요한 경우

1. **먼저 확인**: 05_Troubleshooting.md
2. **그래도 안 되면**: 관련 구현 가이드 재확인
3. **여전히 문제**: GitHub Issues에 문의

**문의 시 포함할 정보**:
- 어떤 Phase에서 막혔는지
- 오류 메시지 전체
- 실행한 명령어
- 관련 파일 경로

---

## ✨ 이 문서 세트의 특징

### 1. 완전성
- 모든 코드가 복사-붙여넣기 가능한 완전한 형태
- 모든 단계에 검증 방법 포함
- 예상되는 모든 문제에 대한 해결책 제공

### 2. 실용성
- 이론보다는 "어떻게"에 집중
- 실제 실행 가능한 명령어 제공
- 구체적인 숫자와 메트릭 제시

### 3. 안전성
- Backward compatibility 철저히 보장
- 모든 변경 사항에 대한 롤백 방법 제공
- 기존 기능 파괴하지 않는 설계

### 4. 교육성
- 왜 이렇게 설계했는지 설명
- 코드베이스 패턴 분석 제공
- Best practice 강조

---

**이 문서 세트는 ST2 Dual-Head 전략을 처음부터 끝까지 안전하게 구현할 수 있도록 설계되었습니다.**

**시작하기 전에 반드시 README.md를 읽으세요! 🚀**
