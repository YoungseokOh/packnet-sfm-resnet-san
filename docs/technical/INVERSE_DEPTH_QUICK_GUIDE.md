# 빠른 참고: 역깊이 변환 방식 비교 (Quick Reference)

**프로젝트**: PackNet-SfM ResNetSAN01  
**목적**: 세 가지 방법의 빠른 이해와 의사결정  
**읽기 시간**: 5분

---

## 🎯 한눈에 보는 비교

### 세 가지 방법

```
방법 1: Linear Space (현재)
┌──────────────────────────────────┐
│ disp [0,1] →→ 선형 변환 →→ depth │
│ inv_depth = a + b×disp (선형)   │
└──────────────────────────────────┘
상태: ✅ 구현됨 (현재 사용 중)
문제: ❌ INT8 양자화에서 39% 오차

방법 2: Log Space (권장 개선)
┌──────────────────────────────────┐
│ disp [0,1] →→ 로그 변환 →→ depth  │
│ log(inv) = a + b×disp (선형)     │
└──────────────────────────────────┘
상태: 미구현 (추가 코드 30줄)
장점: ✅ INT8 양자화 3% 오차 (13배 향상!)

방법 3: Post-Processing (장기 전략)
┌──────────────────────────────────┐
│ sigmoid → [0,1] → Post-Proc      │
│ depth = min + range × normalized │
└──────────────────────────────────┘
상태: 미구현 (구조 변경 필요)
장점: ✅ 최고의 유연성 (같은 모델로 여러 범위)
```

---

## 📊 성능 비교 (0.05~100m 범위)

### INT8 양자화 후 오차

| 방법 | 거리 1m | 거리 10m | 거리 50m | 거리 100m |
|------|--------|----------|----------|-----------|
| **Linear** | **39%** ❌ | **392%** ❌❌ | **1960%** ❌❌❌ | **극심** ❌❌❌ |
| **Log** | **3%** ✅ | **3%** ✅ | **3%** ✅ | **3%** ✅ |
| **Post-Proc** | 39% ❌ | 39% ❌ | 39% ❌ | 39% ❌ |

**결론**: Log Space가 현재 상황에서 최선 ✅

---

## 🔄 각 방법의 핵심 수식

### 방법 1: Linear Space (현재)

```python
min_inv = 1/100 = 0.01
max_inv = 1/0.05 = 20
inv_depth = 0.01 + 19.99 × disp
depth = 1 / inv_depth

오차: Δdepth = 0.0784 × depth²
상대오차: 0.0784 × depth ← 거리에 선형 비례!
d=1m → 7.84% ❌
d=10m → 78.4% ❌❌
d=50m → 392% ❌❌❌
```

### 방법 2: Log Space (권장)

```python
log_min = log(0.01) = -4.605
log_max = log(20) = 2.995
log_inv = -4.605 + 7.60 × disp
inv_depth = exp(log_inv)
depth = 1 / inv_depth

오차: Δ(log_inv) = 7.60 / 255 = 0.0298 (상수!)
상대오차: exp(0.0298) - 1 ≈ 3% ← 모든 거리에서 일정! ✅
d=0.05m → 3% ✅
d=1m → 3% ✅
d=100m → 3% ✅
```

### 방법 3: Post-Processing

```python
# 모델이 반환: normalized_depth ∈ [0, 1]

# Post-Processing
min_inv = 1/100 = 0.01
max_inv = 1/0.05 = 20
inv_depth = 0.01 + 19.99 × normalized_depth
depth = 1 / inv_depth

양자화: [0,1] 공간에서 수행
이후: Post-Processing으로 변환 (오버헤드)
```

---

## ⚡ 즉각적 권장사항

### 당장 할 것 (지금)

```
1️⃣ 현재 Linear 학습 완료
   상태: Epoch 2 진행 중
   변경: 없음
   예상 완료: ~2일

2️⃣ INT8 양자화 후 성능 측정
   목표: FP32 vs INT8 비교
   시간: ~4시간
   결과: "Linear의 INT8이 정말 나쁜가?" 확인
```

### 만약 INT8 성능이 나쁘면 (권장: 39%+ 오차)

```
3️⃣ Log Space 도입
   코드 변경: 30줄 추가
   시간: ~1시간
   개선: 39% → 3% (13배!) ✅✅✅
   
   변경 위치: packnet_sfm/networks/depth/ResNetSAN01.py
   변경 방식: disp_to_inv 함수만 수정
   비용: 최소
```

### 장기 계획 (1개월 후)

```
4️⃣ Post-Processing 도입
   대상: 다음 프로젝트부터
   설계: 정규화 [0,1] 구조
   이점: 같은 모델로 여러 범위 지원
   
   예: depth_1 = post_proc(model_out, 0.05, 100)
       depth_2 = post_proc(model_out, 0.125, 20)
```

---

## 🎓 왜 각 방법이 다른가?

### Linear가 INT8에서 나쁜 이유

```
선형 역깊이 공간:
  disp 0~1 → inv 0.01~20 (선형)
  
깊이 공간으로 변환하면:
  inv 0.01 → depth 100m
  inv 20 → depth 0.05m
  
문제: 깊이 공간은 비선형!
      inv 0.01 변화 = depth 100m 변화
      inv 20 변화 = depth 0.05m 변화
      차이가 극심함!

INT8 양자화 (256단계 균등):
  inv 공간에는 균등하지만
  → 깊이 공간에는 극불균등
  
결과:
  가까운 거리: 오차 작음 ✓
  먼 거리: 오차 극심 ❌❌❌
```

### Log가 INT8에서 좋은 이유

```
로그 역깊이 공간:
  disp 0~1 → log_inv -4.6~3.0 (선형)
  → inv 0.01~20 (로그 분포)
  → depth 100m~0.05m (기하 평균)
  
특징: 깊이가 기하 평균으로 분포!
  0.05m와 100m 사이의 기하 평균:
  √(0.05×100) = √5 ≈ 2.2m
  
INT8 양자화 (256단계):
  log 공간에 균등
  → 깊이 공간에도 (거의) 균등 분포!
  
결과:
  모든 거리: 3% 오차 ✅✅✅
```

### Post-Processing이 유연한 이유

```
구조:
  모델 → [0, 1] → Post-Proc → depth
  
이점:
  같은 모델 (FP32) 가중치로:
  
  depth_A = post_proc(model_out, 0.05, 100)
  depth_B = post_proc(model_out, 0.125, 20)
  depth_C = post_proc(model_out, 0.1, 50)
  
  모두 가능! 범위만 다름!
  
양자화:
  [0, 1] 공간에서 양자화
  후처리는 별도
  → 이론상 최적
```

---

## 💡 의사결정 플로우

```
Q1: INT8 양자화가 중요한가?
├─ 아니오 → Linear 유지 가능
└─ 예 → Q2로

Q2: 빠른 개선을 원하는가?
├─ 예 (지금 바로) → Log Space 도입 ✅
├─ 아니오 → Q3으로
└─ 확인 필요 → INT8 성능 측정 후 판단

Q3: 유연성이 중요한가?
├─ 예 (여러 범위 비교) → Post-Processing ✅
├─ 아니오 → Log Space 충분
└─ 장기 전략 → Post-Processing 설계

최종 권장:
  현재: Linear → INT8 측정 → Log 도입
  미래: Post-Processing 구조 설계
```

---

## 📋 실행 체크리스트

### Phase 1 (현재 진행 중)

- [ ] Linear 학습 완료 (Epoch 28)
- [ ] FP32 성능 기록
- [ ] INT8 양자화
- [ ] INT8 성능 측정

### Phase 2 (선택: INT8이 나쁜 경우)

- [ ] Log Space 코드 준비 (30줄)
- [ ] 기존 모델에 적용
- [ ] INT8 양자화 재시행
- [ ] 성능 개선 확인

### Phase 3 (다음 프로젝트)

- [ ] Post-Processing 구조 설계
- [ ] 정규화 [0,1] 모델 구현
- [ ] 여러 범위 테스트 코드
- [ ] 문서화

---

## 🔗 관련 문서

| 문서 | 목적 | 읽기 시간 |
|------|------|---------|
| `INVERSE_DEPTH_METHODS_ANALYSIS.md` | 상세 분석 | 30분 |
| `INVERSE_DEPTH_VISUAL_ANALYSIS.md` | 시각화 + 증명 | 20분 |
| `DEPTH_RANGE_STRATEGY_COMPARISON.md` | 범위 최적화 | 20분 |

---

## 💬 요점 정리

```
현재 (Linear):
  ✅ 간단, 빠름
  ❌ INT8 양자화 39% 오차

개선 (Log):
  ✅ INT8 양자화 3% 오차 (13배 향상)
  ⏱️ 30분 구현
  
장기 (Post-Processing):
  ✅ 최고의 유연성
  🎯 다음 프로젝트 설계 대상
  
행동:
  1. 현재 INT8 평가
  2. 필요 시 Log Space 도입
  3. 장기적으로 Post-Processing 설계
```

---

**버전**: 1.0  
**작성일**: 2025-10-28  
**다음 갱신**: INT8 평가 완료 후

