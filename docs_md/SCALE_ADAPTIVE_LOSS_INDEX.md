# 📚 Scale-Adaptive Loss 문서 인덱스

PackNet-SFM 프로젝트에 G2-MonoDepth Scale-Adaptive Loss를 추가하기 위한 완전한 문서 모음입니다.

---

## 📖 문서 구조

### 1. 이론 및 배경 지식

#### [`SCALE_ADAPTIVE_LOSS.md`](./SCALE_ADAPTIVE_LOSS.md) 
**깊이 추정을 위한 Scale-Adaptive Loss 이론**

- 📐 수학적 정식화 및 유도
- 🎯 Scale ambiguity 문제 설명
- 💡 G2-MonoDepth Loss 상세 분석
- 📊 다양한 scaling 방법 비교
- 🔬 PyTorch 구현 예시

**대상:** 이론적 배경을 이해하고 싶은 연구자 및 개발자  
**난이도:** ⭐⭐⭐⭐ (고급)  
**예상 시간:** 30-45분

---

### 2. 구현 가이드

#### [`SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md`](./SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md)
**전체 구현 가이드 (상세)**

- 🏗️ 프로젝트 구조 분석
- 🔧 단계별 구현 방법
- 📝 YAML 설정 예시
- 🧪 테스트 및 검증
- 📈 하이퍼파라미터 튜닝
- 🐛 문제 해결
- 🔄 확장 가이드

**대상:** 실제로 구현하는 개발자  
**난이도:** ⭐⭐⭐ (중상급)  
**예상 시간:** 2-3시간 (구현 포함)

---

### 3. 빠른 시작

#### [`SCALE_ADAPTIVE_LOSS_QUICK_START.md`](./SCALE_ADAPTIVE_LOSS_QUICK_START.md)
**5분 빠른 시작 가이드**

- ⚡ 최소한의 단계로 빠른 구현
- 📋 복사-붙여넣기 코드
- ✅ 빠른 테스트 방법
- 🔥 자주 사용하는 명령어
- 🐛 빠른 문제 해결

**대상:** 빠르게 테스트하고 싶은 개발자  
**난이도:** ⭐⭐ (중급)  
**예상 시간:** 5-10분

---

## 🎯 사용 시나리오별 추천 문서

### 시나리오 1: "이론부터 이해하고 싶어요"

```
1. SCALE_ADAPTIVE_LOSS.md (이론) → 30분
2. SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md (구현 이해) → 1시간
3. SCALE_ADAPTIVE_LOSS_QUICK_START.md (실제 구현) → 10분
```

### 시나리오 2: "빠르게 테스트만 하고 싶어요"

```
1. SCALE_ADAPTIVE_LOSS_QUICK_START.md (바로 구현) → 10분
2. (나중에) SCALE_ADAPTIVE_LOSS.md (이론 학습) → 30분
```

### 시나리오 3: "논문 작성을 위한 구현"

```
1. SCALE_ADAPTIVE_LOSS.md (이론 완전 이해) → 1시간
2. SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md (전체 구현) → 2시간
3. 하이퍼파라미터 튜닝 실험 → 1-2일
4. 성능 비교 및 분석 → 1-2일
```

### 시나리오 4: "프로덕션 배포"

```
1. SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md (완전 구현) → 2시간
2. 테스트 코드 작성 → 1시간
3. 성능 검증 → 1일
4. 문서화 및 코드 리뷰 → 반나절
```

---

## 📊 문서 비교표

| 문서 | 이론 | 구현 | 예제 | 튜닝 | 문제해결 |
|------|------|------|------|------|---------|
| **SCALE_ADAPTIVE_LOSS.md** | ✅✅✅ | ⭐⭐ | ⭐⭐ | ⭐ | ⭐ |
| **IMPLEMENTATION.md** | ⭐⭐ | ✅✅✅ | ✅✅✅ | ✅✅✅ | ✅✅✅ |
| **QUICK_START.md** | ⭐ | ✅✅ | ✅✅✅ | ⭐⭐ | ✅✅ |

**범례:**
- ✅✅✅ : 매우 상세함
- ✅✅ : 충분함
- ⭐⭐ : 간략함
- ⭐ : 최소한

---

## 🗂️ 관련 문서

### 프로젝트 문서

- [`../README.md`](../README.md) - 전체 프로젝트 개요
- [`EVALUATE_NCDB_OBJECT_DEPTH_MAPS.md`](./EVALUATE_NCDB_OBJECT_DEPTH_MAPS.md) - NCDB 평가 가이드
- [`UPDATE_SUMMARY.md`](./UPDATE_SUMMARY.md) - 최근 업데이트 요약

### 코드 위치

```
packnet_sfm/
├── losses/
│   ├── scale_adaptive_loss.py          ← 새로 추가할 파일
│   ├── supervised_loss.py              ← 수정할 파일
│   ├── loss_base.py                    ← 기본 클래스
│   ├── ssi_loss.py                     ← 참고용
│   └── ssi_loss_enhanced.py            ← 참고용
├── models/
│   └── SemiSupModel.py                 ← Loss 사용처
└── utils/
    └── depth.py                        ← inv2depth 유틸
```

---

## 🚀 빠른 네비게이션

### 자주 찾는 섹션

#### 이론 학습
- [Scale Ambiguity 문제](./SCALE_ADAPTIVE_LOSS.md#-문제-scale-ambiguity)
- [수학적 정식화](./SCALE_ADAPTIVE_LOSS.md#-최적-scale-factor-계산)
- [G2-MonoDepth Loss](./SCALE_ADAPTIVE_LOSS.md#-g2-monodepth-loss-고급-구현)

#### 구현 방법
- [파일 생성](./SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md#phase-1-loss-클래스-구현)
- [프로젝트 통합](./SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md#phase-2-프로젝트-통합)
- [YAML 설정](./SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md#phase-3-yaml-설정-파일)

#### 빠른 시작
- [5분 시작](./SCALE_ADAPTIVE_LOSS_QUICK_START.md#-5분-안에-시작하기)
- [테스트](./SCALE_ADAPTIVE_LOSS_QUICK_START.md#-step-3-테스트-2분)
- [문제 해결](./SCALE_ADAPTIVE_LOSS_QUICK_START.md#-빠른-문제-해결)

#### 튜닝 가이드
- [하이퍼파라미터](./SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md#-하이퍼파라미터-튜닝-가이드)
- [성능 비교](./SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md#-성능-비교-예상-결과)
- [모니터링](./SCALE_ADAPTIVE_LOSS_IMPLEMENTATION.md#-모니터링-메트릭)

---

## 📈 학습 로드맵

### Week 1: 이론 및 기초 구현

**Day 1-2: 이론 학습**
- [ ] `SCALE_ADAPTIVE_LOSS.md` 정독
- [ ] 수식 이해 및 노트 정리
- [ ] 기존 SSI/Silog loss와 비교

**Day 3-4: 구현**
- [ ] `SCALE_ADAPTIVE_LOSS_QUICK_START.md` 따라하기
- [ ] 단위 테스트 작성 및 실행
- [ ] 5 에폭 학습 테스트

**Day 5: 검증**
- [ ] Loss 그래프 확인
- [ ] Metrics 분석
- [ ] 첫 결과 시각화

### Week 2: 최적화 및 실험

**Day 1-3: 하이퍼파라미터 튜닝**
- [ ] lambda_sg sweep (0.1, 0.3, 0.5, 0.7, 1.0)
- [ ] num_scales 실험 (2, 3, 4, 5)
- [ ] 최적 조합 찾기

**Day 4-5: 성능 비교**
- [ ] 기존 SSI loss와 비교
- [ ] KITTI/NCDB 평가
- [ ] 결과 정리 및 문서화

### Week 3+: 고급 활용

- [ ] Edge-aware weighting 구현
- [ ] Adaptive lambda 실험
- [ ] 논문 작성 (선택)
- [ ] 프로덕션 배포 (선택)

---

## ✅ 체크리스트

### 초급 개발자

- [ ] Quick Start 문서로 5분 구현 완료
- [ ] 단위 테스트 통과
- [ ] 기본 학습 1회 성공
- [ ] TensorBoard에서 loss 확인

### 중급 개발자

- [ ] 전체 구현 가이드 완독
- [ ] 3가지 이상 하이퍼파라미터 조합 실험
- [ ] 성능 비교 완료
- [ ] 문제 발생 시 스스로 해결

### 고급 개발자/연구자

- [ ] 이론 문서 완전 이해
- [ ] 확장 기능 1개 이상 구현
- [ ] 논문 수준 실험 설계
- [ ] 코드 리뷰 및 기여

---

## 🎓 추가 학습 자료

### 관련 논문

1. **G2-MonoDepth** (원본 논문 찾기)
   - Scale-adaptive loss formulation

2. **Eigen et al. (2014)** - NIPS
   - "Depth Map Prediction from a Single Image"
   - Scale-invariant loss 기초

3. **Godard et al. (2019)** - ICCV
   - "Digging Into Self-Supervised Monocular Depth"
   - Median scaling 평가

### 온라인 리소스

- [PyTorch Tutorial: Custom Loss Functions](https://pytorch.org/tutorials/beginner/examples_nn/polynomial_custom_nn.html)
- [Sobel Operator Explained](https://en.wikipedia.org/wiki/Sobel_operator)
- [Multi-scale Image Processing](https://www.cs.toronto.edu/~fleet/courses/2503/fall11/Handouts/pyramids.pdf)

---

## 💬 지원 및 피드백

### 문제 발생 시

1. **Quick Start 문제 해결 섹션** 확인
2. **Implementation 가이드 문제 해결** 참조
3. **GitHub Issues** 생성
4. **로그 파일** 첨부 (`outputs/logs/`)

### 기여 방법

1. 버그 발견 → Issue 생성
2. 개선 아이디어 → Discussion 시작
3. 코드 기여 → Pull Request
4. 문서 오류 → PR로 수정

---

## 📊 문서 통계

| 항목 | 값 |
|------|-----|
| **총 문서 수** | 3개 |
| **총 페이지** | ~80페이지 |
| **코드 예제** | 50+ 개 |
| **이론 수식** | 30+ 개 |
| **YAML 예시** | 5개 |
| **명령어 예시** | 20+ 개 |

---

## 🔄 문서 업데이트 이력

| 날짜 | 버전 | 변경 사항 |
|------|------|----------|
| 2025-10-17 | 1.0 | 초기 문서 세트 생성 |
| | | - 이론 문서 (한국어 번역) |
| | | - 구현 가이드 작성 |
| | | - 빠른 시작 가이드 작성 |

---

## 📞 연락처

- **프로젝트 메인테이너:** PackNet-SFM Team
- **원본 저장소:** [TRI-ML/packnet-sfm](https://github.com/TRI-ML/packnet-sfm)
- **현재 저장소:** packnet-sfm-resnet-san
- **라이센스:** MIT (Toyota Research Institute)

---

**마지막 업데이트:** 2025년 10월 17일  
**문서 버전:** 1.0  
**언어:** 한국어 (Korean)
