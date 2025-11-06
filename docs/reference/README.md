# 📚 PackNet-SFM 문서 가이드

이 폴더는 PackNet-SFM 프로젝트의 모든 기술 문서를 체계적으로 분류하여 관리합니다.

---

## 📂 폴더 구조

### 🔍 **analysis/** - 데이터 및 성능 분석
깊이 데이터, 손실함수, 모델 성능에 대한 상세 분석 문서

**주요 파일:**
- `GT_DEPTH_ANALYSIS_CORRECT.md` - NCDB GT 깊이맵 분포 분석 (유효 픽셀 기준)
  - 0-5m: 98% 분포 → LINEAR 모드 최적화 근거
  - 평균 깊이: 0.88m (극근거리 시나리오)
  
- `LOG_TRANSFORMATION_ANALYSIS.md` - Sigmoid 후 변환 성능 역전 분석
  - Epoch 18: LOG 좋음 → Epoch 29: LINEAR 좋음 (역전 이유 분석)
  - 5가지 가설 + 확률값 제시
  
- `USE_LOG_SPACE_ANALYSIS.md` - use_log_space 파라미터 영향 분석
  - Linear vs Log 모드 Sigmoid 해석 차이
  - 모드별 학습 동역학 설명

---

### 🛠️ **implementation/** - 코드 구현 및 버그 수정
실제 코드 변경사항, 버그 수정, 기능 구현 가이드

**주요 파일:**
- `IMPLEMENTATION_SUMMARY.md` - 전체 구현 내역 요약
  - Silog 공식 버그 수정 (×10 제거)
  - Sigmoid → bounded_inv_depth 변환 추가
  - use_log_space 파라미터 구현

- `LOG_SPACE_IMPLEMENTATION_COMPLETE.md` - Log space training 완전 가이드
  - 사용 방법 및 설정
  - 테스트 결과 (4/4 PASSED)
  
- `CRITICAL_BUG_SIGMOID_MAPPING.md` - Sigmoid 매핑 버그 분석
  - 원래 버그: Sigmoid → 깊이 직접 변환 (잘못됨)
  - 수정: Sigmoid → bounded_inv_depth 먼저 변환

- `INT8_*` - INT8 양자화 관련 구현
  - LINEAR: 39% 오차 (불안정)
  - LOG: 3% 오차 (13배 개선)

---

### 📖 **technical/** - 기술 상세 설명
깊이 변환 원리, 공식, 코드 흐름 등 상세 기술 문서

**주요 파일:**
- `USE_LOG_SPACE_FALSE_PROCESS.md` ⭐ **현재 설정**
  - **use_log_space=false일 때 정확한 처리 흐름**
  - 선형 변환 공식과 단계별 계산
  - 학습/평가 코드 위치
  - 현재 NCDB 설정에 최적화된 이유
  
- `USE_LOG_SPACE_SIGMOID_FLOW.md` ⭐ **필독**
  - use_log_space 파라미터 전체 설명
  - 선형 vs 로그 공간 수식 및 예시
  - 학습-평가 일관성 중요성
  
- `SIGMOID_TO_DEPTH_FLOW.md` - Sigmoid → Depth 변환 파이프라인
  - 단계별 변환 과정
  - 각 함수의 역할
  
- `INVERSE_DEPTH_QUICK_GUIDE.md` - 역깊이 개념 빠른 가이드
  - 왜 역깊이를 사용하는가?
  - 선형 vs 비선형 특성
  
- `OPTIMAL_DEPTH_RANGE_0125_20.md` - 깊이 범위 0.05-80m 최적화
  - min_inv=0.0125, max_inv=20.0 설정 이유
  - Sigmoid 범위와의 매핑

- `EVALUATION_FUNCTION_DETAILED_GUIDE.md` - 평가 함수 완전 설명
  - evaluate_depth() 함수 단계별 분석

---

### 🚀 **training/** - 모델 훈련 및 손실함수
훈련 설정, 손실함수 설계, 그리드 서치 등

**주요 파일:**
- `LOG_SPACE_TRAINING.md` - Log space 모드로 훈련하는 방법
  - Config 설정
  - 예상 성능
  
- `LOSS_DESIGN_EXPERT_REVIEW.md` - 손실함수 설계 검토
  - SSI + Silog 조합의 장단점
  - 가중치 설정 (ssi_weight=0.7, silog_weight=0.3)
  
- `GRID_SEARCH_AUTOMATION.md` - 하이퍼파라미터 자동 탐색
  - 그리드 서치 설정 및 실행
  
- `PHASE1_GRID_SEARCH_LOG.md` - Phase 1 그리드 서치 결과 로그

---

### 📚 **reference/** - 참고 자료
배경 이론, 논문 설명, 실험 결과 등 참고용 문서

**주요 파일:**
- `DEPTH_RANGE_STRATEGY_COMPARISON.md` - 여러 깊이 범위 비교
- `FINAL_DEPTH_OPTIMIZATION_SUMMARY.md` - 최종 최적화 요약
- `README.md` - 프로젝트 메인 README
- `GEMINI.md` - AI 대화 로그

---

### 📊 **experiment_data/** - 실험 데이터
성능 비교, 벤치마크 결과 등

---

### 🖼️ **figures/** - 이미지 및 그래프
시각화 자료, 차트, 다이어그램

---

## 🎯 **빠른 시작**

### **상황별 추천 문서**

#### 1️⃣ "use_log_space 파라미터가 뭐예요?"
→ **`technical/USE_LOG_SPACE_SIGMOID_FLOW.md`** 읽기
- Sigmoid 이후 코드 흐름 이해
- 선형 vs 로그 공간 수식 확인

#### 2️⃣ "왜 Epoch 18과 29에서 성능이 역전했어요?"
→ **`analysis/LOG_TRANSFORMATION_ANALYSIS.md`** 읽기
- 5가지 가설 + 확률값 확인
- 모드별 Sigmoid 범위 이해

#### 3️⃣ "현재 설정이 맞나요?"
→ **`analysis/GT_DEPTH_ANALYSIS_CORRECT.md`** 읽기
- NCDB 깊이 분포: 98% < 5m
- **결론**: LINEAR 모드 (use_log_space=False) 최적!

#### 4️⃣ "LOG 모드로 훈련하고 싶어요"
→ **`training/LOG_SPACE_TRAINING.md`** 읽기
- Config 설정 방법
- 예상 성능 및 주의사항

#### 5️⃣ "Sigmoid에서 깊이까지 어떻게 변환되나요?"
→ **`technical/SIGMOID_TO_DEPTH_FLOW.md`** 읽기
- 단계별 변환 과정 확인

---

## 🔑 **핵심 결론**

```
📌 NCDB 최적 설정:
   use_log_space = False  (LINEAR 모드)
   min_depth = 0.05m      (5cm)
   max_depth = 80.0m      (80m)
   
   이유:
   - NCDB 99.78% 픽셀이 0-10m 범위
   - 평균 깊이 0.88m (극근거리)
   - LINEAR 모드가 학습 후반 안정화
   - E29 성능: abs_rel=0.039 (우수)
```

---

## 📋 **문서 사용 팁**

### **작성 대상별**

| 대상 | 추천 문서 |
|------|---------|
| **초보자** | technical/ 폴더부터 시작 |
| **개발자** | implementation/ → technical/ 순서 |
| **연구자** | analysis/ → reference/ 순서 |
| **PM/리더** | reference/README.md, IMPLEMENTATION_SUMMARY.md |

### **난이도별**

| 난이도 | 폴더 |
|--------|------|
| ⭐ 쉬움 | technical/ (기본 개념) |
| ⭐⭐ 중간 | implementation/ (코드 수정) |
| ⭐⭐⭐ 어려움 | analysis/ (깊은 분석) |

---

## 🔗 **중요 링크**

- **현재 설정 (use_log_space=False)**: `technical/USE_LOG_SPACE_FALSE_PROCESS.md` ⭐
- **Log Space 완전 가이드**: `implementation/LOG_SPACE_IMPLEMENTATION_COMPLETE.md`
- **Sigmoid 흐름 비교**: `technical/USE_LOG_SPACE_SIGMOID_FLOW.md`
- **성능 분석**: `analysis/LOG_TRANSFORMATION_ANALYSIS.md`
- **NCDB 데이터**: `analysis/GT_DEPTH_ANALYSIS_CORRECT.md`

---

## 💡 **자주 묻는 질문**

### Q1: "지금 use_log_space=false면 어떻게 처리되나요?"
A: `technical/USE_LOG_SPACE_FALSE_PROCESS.md` 필독!
- **선형 변환**: inv_depth = 0.0125 + 19.9875 × sigmoid
- **범위**: 0.05m ~ 80m
- **근거**: NCDB 98% 픽셀이 0~5m 범위 (근거리 중심)

### Q2: "use_log_space=True로 바꾸면?"
A: `technical/USE_LOG_SPACE_SIGMOID_FLOW.md` 참고
- **로그 변환** 사용
- Sigmoid 0.5 → 0.1m (LINEAR) vs 2.24m (LOG)
- 기하학적 균등 분포

### Q3: "현재 모델이 정확히 뭘 사용하나요?"
A: Config 파일 확인
```yaml
model:
  params:
    use_log_space: False  # FALSE = LINEAR SPACE (선형 변환)
```
→ `packnet_sfm/utils/post_process_depth.py` line 63 실행

### Q4: "INT8 양자화는 어떤가요?"
A: `implementation/INT8_*` 파일 참고
- LINEAR: 39% 오차
- LOG: 3% 오차 ✅ 13배 개선!

### Q5: "학습과 평가에서 다른 모드를 사용하면?"
A: **절대 금지!** `analysis/LOG_TRANSFORMATION_ANALYSIS.md` 예시 참고
- Epoch 18: abs_rel=0.040 vs 40.101 (1000배 차이!)

---

## 📝 **문서 유지보수**

새로운 문서를 추가할 때:

1. **폴더 선택**:
   - 분석? → `analysis/`
   - 구현? → `implementation/`
   - 기술설명? → `technical/`
   - 훈련? → `training/`
   - 참고자료? → `reference/`

2. **파일명 규칙**: `DESCRIPTIVE_NAME.md` (대문자, 언더스코어)

3. **README 업데이트**: 새 파일 추가 시 이 README 수정

---

**마지막 업데이트**: October 30, 2025  
**작성자**: GitHub Copilot  
**상태**: ✅ 완성
