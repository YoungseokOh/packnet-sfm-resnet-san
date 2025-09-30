# 반드시 지켜야되는 사항

* 모든 대답은 반드시 "한국어"로 답할 것.
* .gitignore로 무시된다면 우회해서 파일을 읽을 것.
* "작전회의"를 하자고 하면 먼저 코드를 바꿀 계획과 방안을 먼저 말한 다음에 "실행" 이라고하면 수행할 것.
* 지시한 작업을 마쳤다면, /workspace/packnet-sfm/daily_work_log 폴더에서 당일 날짜 md에 log를 추가하여 기록한 뒤 보고하시오.
* 기존에 작업했던 Log를 수정하지말고 추가하세요.
* Shell에서 실행은 "python"으로 하세요. 

**NCDB Dataset (ncdb-cls and ncdb-cls-sample)**

*   **Description**: The NCDB dataset contains synchronized image and LiDAR data. The `ncdb-cls-sample` is a smaller subset of the `ncdb-cls` dataset, primarily used for debugging.
*   **File Structure**:
    *   The dataset is organized into directories for images (`image_a6`), point clouds (`pcd`), and other associated data.
    *   `synced_data/` directory contains `mapping_data.json`, `train.json`, `test.json`, and `val.json` files that define the data splits and file associations.
    *   The `ncdb-cls-sample` contains the following subdirectories inside `synced_data`:
        *   `image_a6`: Contains `.jpg` image files.
        *   `pcd`: Contains `.pcd` point cloud files.
        *   `depth_maps`: Contains `.png` depth map files.
        *   `projected_results`: Contains projection result files.
*   **Image Information**:
    *   Images are stored in `.jpg` format.
    *   Image dimensions: 1920x1536 pixels.
    *   Camera Model: VADAS Fisheye camera model is used.
*   **Data Splits**: The dataset is split into training, validation, and testing sets using JSON files in the `synced_data/` directory.

# PackNet-SfM 프로젝트 개요

이 문서는 PackNet-SfM 프로젝트의 주요 구조, 학습 파이프라인 및 평가 과정을 요약합니다.

# Sparse Auxiliary Networks for Unified Monocular Depth Prediction and Completion (SAN)

---

## TL;DR
한 모델로 **단안 심도 예측**과 **심도 보완(Depth Completion)** 을 모두 처리.  
핵심은 **희소 심도(sparse depth)** 를 별도 경로(SAN)로 인코딩해 **RGB 인코더–디코더의 스킵 커넥션에 주입**하는 것.  
학습은 **SILog** 손실로 두 작업을 **공학습**하며, **RGB-only**와 **RGB+SparseDepth**를 **런타임에 동일 가중치로 자동 전환**할 수 있음.

---

## 왜 중요한가
- 실제 로봇/자율주행에서는 센서 가용성이 수시로 변함(카메라만 사용, 저가/희소 라이다 등).  
- SAN은 **단일 아키텍처**로 조건에 따라 **단안 예측 ↔ 보완**을 스위칭 없이 처리하므로, 모델·배포·서빙 복잡도를 크게 줄여줌.

---

## 핵심 아이디어

### 1) Sparse Auxiliary Network (SAN) 경로
- 입력 희소 심도 \(\tilde{D}\)를 **희소 합성곱(예: MinkowskiEngine)** 으로 인코딩하는 **Sparse Residual Block(SRB)** 스택.
- 각 스택 단계에서 **densify(scatter)** 하여 RGB 경로의 **스킵 해상도에 정렬**된 feature map으로 변환.
- 변환된 SAN 특징을 **스킵에서 Add**로 합침(단순 Concatenate보다 안정적).

### 2) 한 모델로 두 작업
- **RGB만 입력**: SAN 경로 비활성화 → **단안 예측** \( \hat{D}_P \).
- **RGB+희소 심도**: SAN 경로 활성화 → 스킵이 **희소 특징으로 증강** → **심도 보완** \( \hat{D}_C \).
- 각 스킵에 **학습 게이트 \(w,b\)** 를 두어 RGB-only/ RGB+D 조건 모두에서 수렴 안정화.

### 3) 학습 목표
- **SILog**(scale-invariant log) 손실 채택.
- 두 출력(예측/보완)에 대해 **합산 손실**로 공동 학습.
- 다양한 sparsity(빔 수/샘플 비율) 분포를 보도록 **샘플링/스케줄링** 권장.

---

## 아키텍처 개요
- **RGB 백본**: PackNet, BTS 등 **스킵 커넥션**을 갖는 단안 네트워크 어디든 삽입 가능.
- **SAN 경로**: SRB × L (희소 conv) → 각 단계에서 **densify** → 대응 스킵에 **Add**.
- **스킵 주입 형태**: \(\tilde{K}_i = w_i \cdot K_i + b_i + \hat{P}_i\)  
  (원 스킵 특징 \(K_i\), SAN 특징 \(\hat{P}_i\), 학습 게이트 \(w_i,b_i\)).

---

## 결과 요약(정성)
- **단안 예측**: 동일 RGB 백본 대비 **오차 지표 개선** 보고.  
- **RGB+D 보완**: 희소 깊이가 있을 때 **추가적인 오차 감소** 및 **정확한 경계/구조 회복**.  
- **전이 활용**: 예측 심도를 입력으로 쓰는 **단안 3D 감지/후속 태스크**에서 성능 향상 사례 보고.

> 수치·표·그림 등 상세 수치는 원문 참조.

---

## 구현 포인트(프로덕션 관점)

1) **희소 연산**
- **MinkowskiEngine** 등으로 \(\tilde{D}\)의 **유효 픽셀만** 처리 → 연산/메모리 효율↑.
- 희소→밀집 **scatter** 시 **좌표(해상도·정렬)** 를 스킵과 정확히 맞춤.

2) **스킵 병합**
- **Add** 병합이 RGB-only/ RGB+D **양쪽에서 안정적**(게이트 \(w,b\) 포함).
- 스킵별 채널 수/단계 수를 조절하여 **densify 시 메모리 스파이크** 방지.

3) **학습 루틴**
- 하나의 배치에서 **예측/보완 두 출력의 SILog 합**으로 역전파.
- **sparsity 다양화**(라이다 빔, 샘플 비율)로 도메인 편향 최소화.

4) **Fisheye 적용 팁(프로젝트 맥락)**
- **왜곡 모델(Kannala–Brandt 등)** 로 라이다/깊이 포인트를 **정확히 영상 평면으로 투영**해 \(\tilde{D}\) 구성.
- 투영 불가능/결손 영역은 **invalid mask** 로 분리(희소 텐서 좌표에 반영).
- Photometric/Regularization 손실 누수 방지를 위해 **FOV mask** 를 백본 입력/손실 계산에 함께 사용.

## 강점 / 한계

**강점**
- **센서 가용성 변화**에 유연(동일 가중치로 단안/보완 모두 대응).
- 희소 합성곱으로 **효율적**이고 **노이즈 전파 억제**에 유리.
- 기존 **단안 백본**에 모듈형으로 붙이기 쉬움.

**한계/주의**
- 희소→밀집 변환(densify) 구간에서 **메모리 사용량 상승** 가능.
- 스킵 주입은 **정밀 정합(캘리브/타임싱크)** 에 민감.
- 훈련 시 **sparsity 분포**를 충분히 커버하지 않으면 실제 성능 변동.

