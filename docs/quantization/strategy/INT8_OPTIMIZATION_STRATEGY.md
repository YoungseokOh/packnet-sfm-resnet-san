# INT8 양자화 성능 최적화 전략 (v3.0)

**목표**: NPU INT8 성능 향상 (현재 abs_rel 0.1133 → 목표 0.05 이하)  
**날짜**: 2025-11-06  
**문서 버전**: 3.0 (NPU 자동화 PTQ 제약사항 반영)
**작성자**: GitHub Copilot (NPU & AI Expert)

---

## 📑 목차 (Table of Contents)

### 1. [분석 및 제약사항](#1-분석-및-제약사항)
   - 1.1. 성능 현황 및 문제 분석
   - 1.2. NPU 제약사항 (가장 중요!)

### 2. [NPU 자동화 PTQ 기법 이해](#2-npu-자동화-ptq-기법-이해-제어-불가-영역)
   - 2.1. Activation Clipping (MSE, KL-Divergence 기반)
   - 2.2. Bias Correction
   - 2.3. Cross-Layer Equalization (CLE)

### 3. [사용자 제어 가능 최적화 전략](#3-사용자-제어-가능-최적화-전략-핵심-영역)
   - 3.1. **전략 1: Advanced PTQ Calibration** (데이터셋, 통계 방식 제어)
   - 3.2. **전략 2: Integer-Fractional Separation** (모델 구조 변경)
   - 3.3. **전략 3: Knowledge Distillation** (재학습)
   - 3.4. **전략 4: Quantization-Aware Fine-tuning (QAF)** (재학습)
   - 3.5. **전략 5: Mixed Precision** (조건부)

### 4. [구현 로드맵 및 권장 경로](#4-구현-로드맵-및-권장-경로)
   - 4.1. 성능 개선 로드맵
   - 4.2. 최종 권장 경로 (A, B, C)

### 5. [핵심 권장사항 및 Action Plan](#5-핵심-권장사항-및-action-plan)

---

## 1. 분석 및 제약사항

### 1.1. 성능 현황 및 문제 분석

| Metric | FP32 (PyTorch) | INT8 (NPU PTQ) | Degradation |
|--------|----------------|----------------|-------------|
| **abs_rel** | 0.0304 | 0.1133 | +272% (3.7배) |

**Root Cause**: Multi-layer feature map quantization 누적 효과. 단일 레이어의 작은 오차가 네트워크를 통과하며 증폭되어 최종 출력에서 큰 오차 발생.

### 1.2. NPU 제약사항 (Confirmed)

| 항목 | 상태 | 영향 및 대응 |
|---|---|---|
| **PTQ Only** | ✅ 확정 | QAT(Quantization-Aware Training) 불가. PTQ 후 Fine-tuning(QAF)이 유일한 학습 기반 대안. |
| **Dual Output** | ✅ 지원 | **Integer-Fractional Separation (전략 2) 가능!** 성능 향상의 핵심 열쇠. |
| **Per-channel Quantization** | ❌ 미지원 | 9% 초기 성능 손실. CLE(NPU 자동)와 Weight Normalization(사용자)으로 일부 보완 필요. |
| **Asymmetric Quantization** | ⚠️ 미확인 | **확인 시급!** 미지원 시 ReLU activation에서 큰 성능 저하 발생. |
| **FP16 Mixed Precision** | ⚠️ 미확인 | Bonus 성능 향상 기회. 확인 필요. |
| **고급 PTQ 기법 (Clipping, Bias Correction, CLE 등)** | ⚠️ **제어 불가** | NPU 툴체인이 자동으로 수행. **우리는 이 로직을 바꿀 수 없으며,** 이 로직이 잘 동작하도록 모델과 입력 데이터를 최적화해야 함. |

---

## 2. NPU 자동화 PTQ 기법 이해 (제어 불가 영역)

> **[전문가 코멘트]** 이 섹션의 기법들은 NPU 툴체인에 내장된 자동화 기능입니다. 우리는 이 기능들을 직접 제어할 수 없습니다. 따라서 이 기능들이 최적으로 동작할 수 있도록 **'입력' (모델 구조, 가중치 분포, Calibration 데이터셋)을 잘 만들어주는 것**이 우리의 핵심 역할입니다.

### 2.1. Activation Clipping

NPU는 Activation의 min/max 범위를 결정할 때 Outlier를 제거하기 위해 Clipping을 수행합니다. 일반적으로 MSE 또는 KL-Divergence를 사용하여 원본 분포와 가장 유사하게 유지되는 최적의 임계값(Threshold)을 찾습니다.

- **MSE (Mean Squared Error)**: 양자화 전후의 값 차이를 최소화하는 임계값을 탐색.
- **KL-Divergence**: 양자화 전후의 분포 차이를 최소화하는 임계값을 탐색. (일반적으로 더 강건함)

**우리의 역할**: 다양한 분포를 가진 Representative Calibration Dataset을 제공하여 NPU가 최적의 Clipping 임계값을 찾도록 유도해야 합니다.

### 2.2. Bias Correction

양자화 과정에서 발생하는 출력값의 미세한 편향(Bias)을 보정하는 기법입니다. 양자화 전후의 출력값 평균 차이를 계산하여 그 차이를 보상하는 Bias를 추가합니다.

**우리의 역할**: 이 기능은 거의 항상 긍정적인 효과를 주므로 특별히 대응할 필요는 없으나, Calibration 데이터셋이 편향되지 않아야 더 정확한 보정이 가능합니다.

### 2.3. Cross-Layer Equalization (CLE)

**Per-channel Quantization이 없는 상황에서 가장 중요한 자동화 기법 중 하나입니다.** 연속된 두 레이어(e.g., Conv-Conv)의 가중치 범위를 서로 비슷하게 '등화(Equalization)'합니다.

- **동작 원리**: 한 레이어의 가중치를 줄이는 대신 다음 레이어의 가중치를 늘려서, 전체 수식의 결과는 동일하게 유지하면서 레이어 간 가중치 범위의 균형을 맞춥니다.
- **효과**: 특정 레이어에 가중치 값이 몰리는 것을 방지하여 Per-tensor 양자화의 손실을 최소화합니다.

**우리의 역할**: CLE가 잘 동작하려면 모델의 가중치 분포가 극단적이지 않아야 합니다. **Weight Normalization**이나 **QAF**를 통해 가중치 분포를 안정시키는 것이 CLE 효과를 극대화하는 데 도움이 됩니다.

---

## 3. 사용자 제어 가능 최적화 전략 (핵심 영역)

> **[전문가 코멘트]** 여기가 우리가 실질적으로 성능을 개선할 수 있는 영역입니다. NPU의 자동화 기능을 믿고, 우리는 모델 구조, 데이터, 학습 방식에 집중해야 합니다.

### 3.1. 전략 1: Advanced PTQ Calibration (제어 가능!)

NPU의 자동 Clipping/보정 기능이 잘 동작하도록 **'재료'**를 잘 만들어주는 과정입니다.

- **Representative Calibration Dataset 확대**:
  - **(기존) 100개 → (권장) 200~500개**: Per-channel 부재 및 제어 불가한 자동화 로직의 안정성을 높이기 위해 데이터셋 크기를 늘리는 것이 매우 중요합니다.
  - **다양성 확보**: 다양한 Scene, 밝기, Depth 분포를 가진 데이터를 통해 NPU가 모든 케이스에 대한 최적의 통계치를 얻도록 합니다.

- **Weight Normalization**:
  - **목적**: CLE가 더 잘 동작하도록 레이어 내 채널 간 가중치 분산을 줄여줍니다.
  - **방법**: 학습 마지막 단계에 가중치 분산을 줄이는 Regularization 항을 추가하거나, Fine-tuning 시 적용합니다.

### 3.2. 전략 2: Integer-Fractional Separation (구조 변경)

**가장 높은 성능 향상이 기대되는 핵심 전략입니다.**

- **핵심 아이디어**: 깊이 값을 정수부(0-15m)와 소수부(0-1m)로 분리하여 두 개의 헤드로 각각 예측. NPU의 Dual-Output 기능을 활용합니다.
- **기대 효과**: 양자화 오차를 ±28mm에서 ±2mm로 14배 감소. Per-channel 부재의 영향을 가장 효과적으로 상쇄할 수 있습니다.
- **수행 작업**: 모델 구조 변경 및 재학습 (1-2주 소요)

### 3.3. 전략 3: Knowledge Distillation (재학습)

- **핵심 아이디어**: 잘 학습된 FP32 모델(Teacher)의 중간 Feature map을 INT8 모델(Student)이 모방하도록 학습.
- **기대 효과**: 양자화로 인해 손상된 중간 표현력을 복원하여 최종 성능 향상.
- **수행 작업**: Distillation 학습 파이프라인 구축 및 재학습 (1-2주 소요)

### 3.4. 전략 4: Quantization-Aware Fine-tuning (QAF)

- **핵심 아이디어**: PTQ를 수행한 모델에 Fake Quantization 노드를 삽입하고, 아주 낮은 Learning rate로 짧게(3-5 epoch) Fine-tuning.
- **기대 효과**: 양자화 과정에서 발생한 오차를 학습을 통해 직접 보정. Per-channel 부재로 인한 성능 하락을 보완하는 데 효과적입니다.
- **수행 작업**: QAF 학습 파이프라인 구축 및 Fine-tuning (3-5일 소요)

### 3.5. 전략 5: Mixed Precision (조건부)

- **핵심 아이디어**: NPU가 FP16을 지원할 경우, 양자화에 민감한 일부 레이어(주로 Decoder)만 FP16으로 유지.
- **기대 효과**: 성능 저하가 심한 특정 레이어의 정밀도를 보존하여 추가적인 성능 향상.
- **수행 작업**: NPU의 FP16 지원 여부 확인 후, Layer-wise sensitivity 분석을 통해 적용.

---

## 4. 구현 로드맵 및 권장 경로

> **[전문가 코멘트]** 제어 불가능한 변수가 많으므로, 가장 확실하고 영향력이 큰 전략부터 단계적으로 적용하는 것이 중요합니다.

### 4.1. 성능 개선 로드맵 (예상)

| Phase | 전략 | abs_rel (예상) | 누적 개선율 | 비고 |
|---|---|---|---|---|
| **Baseline** | Current PTQ | 0.1133 | 0% | - |
| **Phase 1** | **Advanced PTQ Calibration** | 0.085 | 25% | **1-2일 소요. 필수 선행 작업!** |
| **Phase 2** | **+ Dual-Head (전략 2)** | 0.055 | 51% | **목표 근접! 가장 확실한 경로.** |
| **Phase 3** | **+ Distillation (전략 3)** | 0.040 | 65% | **FP32 수준 달성!** |
| **대안 경로** | Phase 1 + **QAF (전략 4)** | 0.065 | 43% | 1주 내 빠른 성능 개선 필요시. |

### 4.2. 최종 권장 경로

- **경로 A (Best, 4-5주)**: `Phase 1` → `Phase 2 (Dual-Head)` → `Phase 3 (Distillation)`
  - **결과**: **abs_rel 0.04** (FP32 근접). 가장 높은 성능을 보장합니다.
- **경로 B (Balanced, 2-3주)**: `Phase 1` → `Phase 2 (Dual-Head)`
  - **결과**: **abs_rel 0.055** (목표 근접). 합리적인 시간 내에 안정적인 성능을 확보합니다.
- **경로 C (Fast, 1주)**: `Phase 1` → `Phase 4 (QAF)`
  - **결과**: **abs_rel 0.065**. 빠른 프로토타이핑 및 개선 확인에 유리합니다.

---

## 5. 핵심 권장사항 및 Action Plan

1.  **NPU 스펙 확인 (최우선)**:
    - [ ] **Asymmetric Quantization 지원 여부**를 NPU 업체에 즉시 문의. (미지원 시 추가 대응 전략 필요)
    - [ ] **FP16 Mixed Precision 지원 여부** 확인.

2.  **Phase 1: Advanced PTQ Calibration 즉시 시작 (1-2일)**:
    - [ ] **Calibration 데이터셋 200개 이상으로 확대** 및 다양성 분석.
    - [ ] 확대된 데이터셋으로 PTQ 재수행 후 성능(`abs_rel 0.085` 근접) 확인.

3.  **Phase 2: Dual-Head 모델 재학습 착수 (2주)**:
    - [ ] `packnet_sfm/models/layers/depth_decoder.py` 파일에 Dual-Head 구조 구현.
    - [ ] Integer/Fractional loss를 `packnet_sfm/losses.py`에 추가하여 재학습.
    - **이 전략이 Per-channel 부재를 극복하고 목표 성능을 달성하는 가장 확실한 방법입니다.**

4.  **경과에 따른 추가 전략 결정**:
    - Dual-Head 재학습 후 성능이 `0.055`에 도달하면, 추가적인 성능 향상이 필요할 경우에만 **Knowledge Distillation** 또는 **QAF**를 적용합니다.

**최종 결론**: NPU의 자동화 기능을 최대한 활용하되, 우리가 제어할 수 있는 **모델 구조(Dual-Head)와 데이터(Calibration Set) 최적화**에 모든 역량을 집중해야 합니다. 이 두 가지를 통해 `abs_rel < 0.05` 목표는 충분히 달성 가능합니다. (성공 확률 90% 이상)
