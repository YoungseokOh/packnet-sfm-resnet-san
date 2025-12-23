# G2-MonoDepth 전체 개요

## 1. 프로젝트 정보

| 항목 | 내용 |
|------|------|
| **논문명** | RGB+X: Depth Completion and Monocular Depth Estimation |
| **학술지** | IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2024 |
| **저자** | Wang et al. (Xi'an Jiaotong University) |
| **GitHub** | https://github.com/Wang-xjtu/G2-MonoDepth |
| **핵심 기여** | RGB + Sparse Depth를 통합하여 다양한 센서에 일반화된 depth inference |

---

## 2. 연구 동기 및 핵심 아이디어

### 2.1 문제 정의
기존 depth completion 방법들은 특정 센서(LiDAR 등)에 최적화되어 있어, 다른 센서로의 일반화(generalization)가 어려움.

### 2.2 핵심 접근법
**RGB + X 프레임워크**: X는 "any sparse depth source"를 의미
- LiDAR (정확한 sparse point)
- Stereo matching (dense but noisy)
- SfM (Structure from Motion)
- ToF (Time of Flight)
- 심지어 **No X** (Monocular Depth Estimation)

### 2.3 핵심 contribution
1. **General Framework**: 다양한 sparsity level (0%~100%)에서 학습
2. **Robust to Artifacts**: noise, holes, blur 등 다양한 센서 특성 시뮬레이션
3. **Single Model**: depth completion과 monocular depth estimation을 하나의 모델로 통합

---

## 3. 프로젝트 구조

```
G2-MonoDepth/
├── config.py              # 설정 파일 (경로, 하이퍼파라미터)
├── train.py               # 학습 진입점 (DDP multi-GPU)
├── val.py                 # 검증 스크립트
├── infer.py               # 추론 스크립트
│
├── src/
│   ├── src_main.py        # 메인 Trainer 클래스
│   ├── networks.py        # UNet 네트워크 정의
│   ├── modules.py         # UNet 모듈 블록
│   ├── custom_blocks.py   # BottleNeck (ReZero)
│   ├── losses.py          # Loss 함수들
│   ├── data_tools.py      # 데이터 로더 및 augmentation
│   └── utils.py           # 유틸리티 함수
│
├── pretrained/            # 사전학습 모델
└── depth_selection/       # NYUv2 depth selection 도구
```

---

## 4. 네트워크 아키텍처 요약

### 4.1 전체 구조: 7-Layer UNet

```
Input (5ch: RGB + point + hole_point)
    ↓
[Encoder: FirstModule + 5 × UNetModule (stride=2)]
    ↓
[Decoder: 5 × UNetModule (upsample=2)]
    ↓
Output (1ch: depth)
```

### 4.2 입력 채널 구성 (5 channels)
| Channel | 설명 |
|---------|------|
| 0-2 | RGB 이미지 |
| 3 | `point`: sparse depth 값 |
| 4 | `hole_point`: depth가 있는 위치 마스크 (binary) |

### 4.3 특징
- **ReZero 기법**: 각 residual block에 학습 가능한 scaling factor (alpha)
- **No BatchNorm**: Layer Normalization 사용
- **Symmetric UNet**: Encoder-Decoder 대칭 구조

---

## 5. Loss 함수 구성

### 5.1 총 Loss
```python
loss = loss_adepth + loss_rdepth + 0.5 * loss_rgrad
```

| Loss | 설명 | 역할 |
|------|------|------|
| `loss_adepth` | Absolute Depth Loss | 절대 depth 값 정확도 |
| `loss_rdepth` | Relative Depth Loss | 상대적 depth 분포 정확도 |
| `loss_rgrad` | Gradient Loss | Edge/구조 보존 |

### 5.2 Relative Domain 변환
```python
# Standardization: mean=0, std 기반 정규화
sta_depth = (depth - mean) / std
sta_gt = (gt - mean_gt) / std_gt
```

- **Robust Standardization**: MAD (Median Absolute Deviation) 사용

### 5.3 Gradient Loss (Multi-Scale Sobel)
- Scale: 1, 2, 4, 8배
- X, Y 방향 Sobel 필터 적용
- Edge 보존 및 전체 구조 일관성 향상

---

## 6. Data Augmentation 전략

### 6.1 기본 Augmentation
| 종류 | 확률 | 설명 |
|------|------|------|
| Horizontal Flip | 50% | 좌우 반전 |
| Color Jitter | 50% | brightness, contrast, saturation, hue |
| Point Hole | 50% | sparse depth에 구멍 생성 |
| Point Noise | 50% | sparse depth에 노이즈 추가 |
| Point Blur | 50% | sparse depth에 blur 적용 |

### 6.2 Sparsity 전략
```python
self.random_point_percentages = list(range(0, 101))  # 0%, 1%, ..., 100%
```
- 매 샘플마다 랜덤한 sparsity 적용
- **0%**: Monocular Depth Estimation 모드
- **100%**: Dense Depth Completion 모드

### 6.3 센서 시뮬레이션
| 센서 | Sparsity | 특성 |
|------|----------|------|
| LiDAR | 5% | sparse, accurate |
| Stereo | 50-70% | dense, noisy |
| SfM | 1-5% | very sparse |
| ToF | 80-100% | dense, blur |

---

## 7. 학습 설정

### 7.1 하이퍼파라미터
| 항목 | 값 |
|------|-----|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 8 (per GPU) |
| Input Size | 640 × 480 |
| Epochs | - (scheduler 기반) |

### 7.2 학습 전략
- **DDP (Distributed Data Parallel)**: Multi-GPU 학습
- **Mixed Precision**: 지원 가능 (별도 설정)
- **Scheduler**: StepLR 또는 CosineAnnealing

---

## 8. 우리 프로젝트에 적용 가능한 요소

### 8.1 직접 적용 가능
1. **Gradient Loss**: Edge 보존을 위한 multi-scale Sobel loss
2. **Relative Domain Loss**: Scale-invariant 학습 효과
3. **ReZero 기법**: Residual connection 안정화

### 8.2 참고 가능
1. **Robust Standardization**: Outlier에 강건한 정규화
2. **Multi-sparsity Training**: 다양한 데이터 조건에 대한 일반화
3. **Sensor Artifact Simulation**: Noise, hole, blur augmentation

### 8.3 적용 시 고려사항
- 우리 프로젝트는 RGB-only이므로 sparse depth 입력 부분은 제외
- Gradient Loss는 Edge-Aware Smoothness Loss와 유사한 효과
- Relative Loss는 SSI Loss와 유사한 개념

---

## 9. 다음 문서

- [02_network_architecture.md](02_network_architecture.md) - 네트워크 구조 상세
- [03_loss_functions.md](03_loss_functions.md) - Loss 함수 상세
- [04_data_processing.md](04_data_processing.md) - 데이터 처리 상세
- [05_training_process.md](05_training_process.md) - 학습 프로세스 상세
