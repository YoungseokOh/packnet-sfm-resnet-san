# G2-MonoDepth 분석 요약 및 적용 제안

## 1. G2-MonoDepth 핵심 요약

### 1.1 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목적** | RGB + 다양한 sparse depth 소스를 통합한 depth inference |
| **핵심 혁신** | 0~100% sparsity에서 학습하여 어떤 센서에도 일반화 |
| **네트워크** | 7-Layer UNet with ReZero |
| **Loss** | Absolute + Relative + Gradient Loss |

### 1.2 핵심 기술 요소

```
┌─────────────────────────────────────────────────────────────────┐
│                    G2-MonoDepth 핵심 요소                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🔷 ReZero BottleNeck                                           │
│     - 학습 가능한 residual scaling (alpha = 0 초기화)            │
│     - 깊은 네트워크의 안정적 학습                                 │
│                                                                 │
│  🔷 3-Term Loss                                                 │
│     - Absolute: 절대 depth 정확도                                │
│     - Relative: scale-invariant 분포 학습                        │
│     - Gradient: edge/구조 보존                                   │
│                                                                 │
│  🔷 Robust Standardization                                      │
│     - MAD (Mean Absolute Deviation) 사용                        │
│     - Outlier에 강건한 정규화                                    │
│                                                                 │
│  🔷 Multi-Sparsity Training                                     │
│     - 0% (RGB-only) ~ 100% (Dense) 전 범위 학습                  │
│     - 다양한 센서 artifact 시뮬레이션                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 우리 프로젝트와의 비교

### 2.1 아키텍처 비교

| 항목 | G2-MonoDepth | PackNet-SfM (우리) |
|------|--------------|-------------------|
| Encoder | Custom UNet (7-layer) | ResNet18 (pretrained) |
| Decoder | Symmetric UNet | Custom Decoder |
| Skip Connection | Addition | Addition |
| Normalization | LayerNorm | BatchNorm |
| Activation | GELU | ELU |
| Residual 기법 | ReZero | Standard |
| 입력 | 5ch (RGB + sparse + mask) | 3ch (RGB) |
| 출력 | Direct depth | Sigmoid × max_depth |

### 2.2 Loss 비교

| 항목 | G2-MonoDepth | PackNet-SfM (우리) |
|------|--------------|-------------------|
| Absolute Loss | L1 | Silog Loss |
| Relative Loss | Standardized L1 | SSI Loss |
| Gradient Loss | Multi-scale Sobel | ❌ 없음 |
| 총 Loss | A + R + 0.5G | 0.5×SSI + 0.5×Silog |

### 2.3 데이터 처리 비교

| 항목 | G2-MonoDepth | PackNet-SfM (우리) |
|------|--------------|-------------------|
| Augmentation | Heavy (sparsity, artifacts) | Basic (flip, color) |
| Normalization | Robust (MAD) | Standard |
| Mask 처리 | hole_point 채널 | GT 유효성 mask |

---

## 3. 적용 가능한 요소

### 3.1 🟢 즉시 적용 가능 (High Priority)

#### (1) Gradient Loss 추가

**효과**: Edge 보존, 전체 맵 일관성 향상

```python
# 제안 구현
class MultiScaleGradientLoss(nn.Module):
    def __init__(self, scales=[1, 2, 4, 8]):
        super().__init__()
        self.scales = scales
        self.sobel_x = torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ]).float().view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ]).float().view(1, 1, 3, 3)
    
    def forward(self, pred, gt, mask):
        total_loss = 0
        for scale in self.scales:
            # Downsample
            pred_s = F.avg_pool2d(pred, scale) if scale > 1 else pred
            gt_s = F.avg_pool2d(gt, scale) if scale > 1 else gt
            mask_s = (F.avg_pool2d(mask.float(), scale) > 0.5).float() if scale > 1 else mask
            
            # Gradient
            grad_pred = self.compute_gradient(pred_s)
            grad_gt = self.compute_gradient(gt_s)
            
            # Loss
            total_loss += self.masked_l1(grad_pred, grad_gt, mask_s)
        
        return total_loss / len(self.scales)
```

**Config 변경**:
```yaml
loss:
  ssi_weight: 0.4
  silog_weight: 0.4
  gradient_weight: 0.2  # 새로 추가
```

#### (2) Loss Logging 세분화

**효과**: 학습 분석 용이, 튜닝 가이드

```python
# supervised_loss.py 수정
def compute_loss(pred, gt, mask):
    ssi_loss = compute_ssi_loss(pred, gt, mask)
    silog_loss = compute_silog_loss(pred, gt, mask)
    gradient_loss = compute_gradient_loss(pred, gt, mask)
    
    total = ssi_weight * ssi_loss + silog_weight * silog_loss + grad_weight * gradient_loss
    
    return total, {
        'ssi_loss': ssi_loss.item(),
        'silog_loss': silog_loss.item(),
        'gradient_loss': gradient_loss.item(),
        'total_loss': total.item()
    }
```

### 3.2 🟡 검토 후 적용 (Medium Priority)

#### (1) ReZero 기법

**효과**: 학습 안정성 향상, 더 깊은 네트워크 가능

```python
class ReZeroResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = ...  # 기존 conv block
        self.alpha = nn.Parameter(torch.zeros(1))  # ReZero
    
    def forward(self, x):
        return x + self.alpha * self.conv_block(x)
```

**적용 위치**: ResNetSAN의 SpatialAttention 모듈

#### (2) GELU Activation

**효과**: 더 부드러운 activation, 학습 안정성

```python
# 현재: ELU
self.act = nn.ELU(inplace=True)

# 변경: GELU
self.act = nn.GELU()
```

**주의**: 기존 pretrained weights와의 호환성 확인 필요

### 3.3 🔵 참고만 (Low Priority)

#### (1) LayerNorm 전환

- BatchNorm → LayerNorm은 큰 변경
- ResNet pretrained weights 호환성 문제
- 새로 학습할 경우에만 고려

#### (2) Multi-Sparsity Training

- 우리는 RGB-only이므로 직접 적용 불가
- 하지만 data augmentation 강화는 참고 가능

#### (3) Robust Standardization (MAD)

- 현재 SSI Loss가 유사한 역할
- 필요 시 SSI Loss 내부에 적용 가능

---

## 4. 구현 우선순위

### 4.1 Phase 1: Gradient Loss 추가 (권장)

```
목표: 전체 맵 일관성 향상

구현 사항:
1. MultiScaleGradientLoss 클래스 구현
2. supervised_loss.py에 통합
3. YAML config에 gradient_weight 추가
4. 학습 및 평가

예상 효과:
- Edge 선명도 향상
- Depth 경계 보존
- 전체 구조 일관성 개선
```

### 4.2 Phase 2: Loss Logging 세분화

```
목표: 학습 분석 및 튜닝 용이성

구현 사항:
1. 각 loss term 별도 logging
2. TensorBoard/WandB에 시각화
3. Loss term 별 추이 분석

예상 효과:
- 어떤 loss가 학습에 기여하는지 파악
- 최적의 weight 조합 탐색 용이
```

### 4.3 Phase 3: ReZero (선택적)

```
목표: 학습 안정성 향상

구현 사항:
1. Attention 모듈에 ReZero 적용
2. 기존 weights 호환성 테스트
3. 학습 속도 및 수렴 비교

예상 효과:
- 더 안정적인 학습
- 잠재적으로 더 깊은 네트워크 가능
```

---

## 5. 실험 계획 제안

### 5.1 Baseline 확립

```
실험명: Baseline (현재 구현)
Config: ssi_weight=0.5, silog_weight=0.5
메트릭: RMSE, MAE, δ1, δ2, δ3
```

### 5.2 Gradient Loss 실험

```
실험명: +Gradient Loss
Config 변형:
  A) ssi=0.4, silog=0.4, grad=0.2
  B) ssi=0.35, silog=0.35, grad=0.3
  C) ssi=0.5, silog=0.5, grad=0.1 (보수적)

평가:
  - Edge 영역에서의 성능 변화
  - 전체 메트릭 변화
  - 학습 속도 변화
```

### 5.3 Ablation Study

```
실험명: Loss Term Ablation
Variants:
  1) SSI only
  2) Silog only
  3) SSI + Silog
  4) SSI + Silog + Gradient

분석:
  - 각 loss term의 기여도
  - 최적 조합 탐색
```

---

## 6. 결론

### 6.1 핵심 takeaway

1. **Gradient Loss는 즉시 적용 가치가 있음**
   - 전체 맵 일관성 개선에 직접적으로 기여
   - 구현 난이도 낮음, 리스크 낮음

2. **Loss 세분화 logging은 필수**
   - 학습 분석 및 디버깅에 필수
   - 향후 튜닝에 필요

3. **ReZero는 선택적**
   - 학습 안정성 이슈가 있을 때 고려
   - 기존 weights 호환성 주의

### 6.2 다음 단계

```
1. Gradient Loss 구현 및 통합
2. 실험 진행 (Baseline vs +Gradient)
3. 결과 분석 및 weight 튜닝
4. 필요 시 추가 기법 적용
```

---

## 문서 목록

| 파일 | 내용 |
|------|------|
| [01_overview.md](01_overview.md) | 프로젝트 전체 개요 |
| [02_network_architecture.md](02_network_architecture.md) | 네트워크 구조 상세 |
| [03_loss_functions.md](03_loss_functions.md) | Loss 함수 상세 |
| [04_data_processing.md](04_data_processing.md) | 데이터 처리 상세 |
| [05_training_process.md](05_training_process.md) | 학습 프로세스 상세 |
| [06_summary_and_recommendations.md](06_summary_and_recommendations.md) | 요약 및 적용 제안 (현재 문서) |
