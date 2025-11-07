# ST2 Quick Reference Guide

**빠른 참조를 위한 핵심 정보 요약**

---

## 📊 성능 목표

| Metric | 현재 (INT8) | 목표 | 개선율 |
|--------|-------------|------|--------|
| abs_rel | 0.1139 | **0.055** | **51%** |
| rmse | 0.751m | **0.50m** | **33%** |
| δ<1.25 | 0.9061 | **0.970** | **7%** |

---

## 🔧 구현 체크리스트

### Phase 1: Decoder (1일)
- [ ] `dual_head_depth_decoder.py` 생성 (~150줄)
- [ ] Unit test 통과
- [ ] Output keys 확인: `("integer", 0)`, `("fractional", 0)`

### Phase 2: Helper Functions (1일)
- [ ] `layers.py`에 함수 추가 (+40줄)
  - `dual_head_to_depth`
  - `decompose_depth`
  - `dual_head_to_inv_depth`
- [ ] Decompose → Reconstruct 오차 < 1e-5

### Phase 3: ResNetSAN01 통합 (1일)
- [ ] `ResNetSAN01.py` 수정 (+30줄)
- [ ] `use_dual_head` 파라미터 추가
- [ ] Factory pattern 구현
- [ ] `is_dual_head` 플래그 확인

### Phase 4: Loss Function (1일)
- [ ] `dual_head_depth_loss.py` 생성 (~120줄)
- [ ] Weights: integer=1.0, fractional=10.0, consistency=0.5
- [ ] NaN 체크 추가

### Phase 5: Model Wrapper (1일)
- [ ] `SemiSupCompletionModel.py` 수정 (+20줄)
- [ ] Dual-Head 자동 감지
- [ ] Backward compatibility 유지

---

## ⚙️ YAML 설정 (완전한 예제 - 복사 후 사용)

```yaml
# configs/train_resnet_san_ncdb_dual_head_640x384.yaml
# ⚠️ 완전한 설정 예제 - 실제 사용 가능

model:
    name: 'SemiSupCompletionModel'
    
    # Loss 설정
    loss:
        supervised_method: 'sparse-l1'
        supervised_num_scales: 1
        supervised_loss_weight: 1.0
    
    # Depth Network 설정
    depth_net:
        name: 'ResNetSAN01'
        version: '18A'
        use_dual_head: true   # ⭐ 핵심 파라미터 (Dual-Head 활성화)
        use_film: false       # FiLM 비활성화 (선택)
        use_enhanced_lidar: false  # Enhanced LiDAR 비활성화 (선택)
    
    # Depth 범위 설정
    params:
        min_depth: 0.5
        max_depth: 15.0       # ⭐ 데이터에 맞춰 조정 (NCDB: 15.0)

# 데이터셋 설정
datasets:
    train:
        split: 'train'
        path: '/data/ncdb/'   # 실제 경로로 변경
        batch_size: 4
        num_workers: 8
    validation:
        split: 'val'
        path: '/data/ncdb/'
        batch_size: 4
        num_workers: 4

# Optimizer 설정
optimizer:
    name: 'Adam'
    learning_rate: 2.0e-4     # ⭐ Dual-Head 권장값
    weight_decay: 0.0

# Scheduler 설정
scheduler:
    name: 'StepLR'
    step_size: 15             # 15 epoch마다 LR 감소
    gamma: 0.1                # LR × 0.1

# Checkpoint 설정
checkpoint:
    save_top_k: 3             # 상위 3개 checkpoint 저장
    monitor: 'abs_rel'        # abs_rel 기준으로 선택
    mode: 'min'               # 낮을수록 좋음

# Trainer 설정
trainer:
    max_epochs: 30
    gradient_clip_val: 1.0
    check_val_every_n_epoch: 1
    log_every_n_steps: 50

# 기타 설정
arch:
    seed: 42                  # 재현성을 위한 seed
```

**🔑 핵심 파라미터 설명**:

| 파라미터 | 값 | 설명 |
|---------|---|------|
| `use_dual_head` | `true` | **필수!** Dual-Head 활성화 |
| `max_depth` | `15.0` | 데이터셋 depth 범위 (NCDB: 15.0, KITTI: 80.0) |
| `learning_rate` | `2.0e-4` | Dual-Head 권장 학습률 |
| `batch_size` | `4` | GPU 메모리에 맞춰 조정 |
| `max_epochs` | `30` | 충분한 수렴 시간 |

**⚠️ 데이터셋별 설정**:

```yaml
# NCDB (Near-field, 0.5~15m)
params:
    min_depth: 0.5
    max_depth: 15.0

# KITTI (Far-field, 1~80m)
params:
    min_depth: 1.0
    max_depth: 80.0
```

---

## 🧪 빠른 테스트 명령어

```bash
cd /workspace/packnet-sfm

# 1. Decoder 테스트
python -c "
from packnet_sfm.networks.layers.resnet.dual_head_depth_decoder import DualHeadDepthDecoder
import torch
decoder = DualHeadDepthDecoder([64, 64, 128, 256, 512], max_depth=15.0)
features = [torch.randn(1, c, 96//(2**i), 160//(2**i)) for i, c in enumerate([64, 64, 128, 256, 512])]
outputs = decoder(features)
assert ('integer', 0) in outputs and ('fractional', 0) in outputs
print('✅ Decoder test passed')
"

# 2. Helper functions 테스트
python -c "
from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth, decompose_depth
import torch
depth = torch.tensor([[[[5.7]]]])
integer_gt, frac_gt = decompose_depth(depth, 15.0)
depth_recon = dual_head_to_depth(integer_gt, frac_gt, 15.0)
assert torch.allclose(depth, depth_recon)
print('✅ Helper test passed')
"

# 3. 전체 모델 테스트
python -c "
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01
import torch
model = ResNetSAN01(version='18A', use_dual_head=True, max_depth=15.0)
print(f'✅ is_dual_head: {model.is_dual_head}')
rgb = torch.randn(1, 3, 384, 640)
model.eval()
with torch.no_grad():
    output, _ = model.run_network(rgb)
print(f'✅ Output shape: {output.shape}')
"
```

---

## 🚀 학습 실행

```bash
# 학습 시작
python scripts/train.py configs/train_resnet_san_ncdb_dual_head_640x384.yaml

# 로그 확인
tail -f checkpoints/resnetsan01_dual_head_640x384/training.log

# TensorBoard
tensorboard --logdir checkpoints/resnetsan01_dual_head_640x384
```

---

## 📈 학습 진행 확인

| Epoch | Integer Loss | Fractional Loss | Val abs_rel |
|-------|--------------|-----------------|-------------|
| 5 | 0.010 | 0.040 | ~0.120 |
| 10 | 0.005 | 0.020 | ~0.090 |
| 20 | 0.002 | 0.010 | ~0.060 |
| **30** | **0.001** | **0.005** | **~0.055** |

**정상 학습 신호**:
- ✅ Integer loss: 빠르게 수렴 (Epoch 5에 0.01 이하)
- ✅ Fractional loss: 천천히 감소 (정밀도 향상)
- ✅ Val abs_rel: 지속적으로 감소

**비정상 신호**:
- ❌ NaN loss
- ❌ Integer loss > 0.05 after Epoch 10
- ❌ Fractional loss > 0.05 after Epoch 30

---

## 🔍 디버깅 우선순위

### 문제 1: NaN Loss
→ [05_Troubleshooting.md#문제-3-nan-loss](05_Troubleshooting.md)

### 문제 2: Integer Loss 높음
→ `max_depth` 값 확인 (YAML vs 데이터)

### 문제 3: Fractional Loss 높음
→ `fractional_weight` 증가 (10.0 → 15.0)

### 문제 4: ModuleNotFoundError
→ `__init__.py` 파일 확인

### 문제 5: ONNX Export 실패
→ Wrapper 클래스 사용 ([05_Troubleshooting.md#문제-8](05_Troubleshooting.md))

---

## 📁 파일 위치 맵

```
packnet_sfm/
├── networks/
│   ├── depth/
│   │   └── ResNetSAN01.py                    # ✏️ 수정 (+30줄)
│   └── layers/
│       └── resnet/
│           ├── dual_head_depth_decoder.py    # 🆕 신규 (~150줄)
│           └── layers.py                     # ✏️ 수정 (+40줄)
├── losses/
│   └── dual_head_depth_loss.py               # 🆕 신규 (~120줄)
└── models/
    └── SemiSupCompletionModel.py             # ✏️ 수정 (+20줄)

configs/
└── train_resnet_san_ncdb_dual_head_640x384.yaml  # 🆕 신규

docs/
└── quantization/
    └── ST2/                                  # 📚 문서
        ├── README.md
        ├── 01_Overview_Strategy.md
        ├── 02_Implementation_Guide.md
        ├── 03_Configuration_Testing.md
        ├── 04_Training_Evaluation.md
        └── 05_Troubleshooting.md
```

---

## 💡 핵심 개념 복습

### Dual-Head 아키텍처

```python
# Integer Head: [0, 15] 범위
integer_sigmoid = 0.333  # → 0.333 * 15 = 5.0m

# Fractional Head: [0, 1] 범위
fractional_sigmoid = 0.7  # → 0.7m

# 최종 깊이
depth = 5.0 + 0.7 = 5.7m
```

### 양자화 정밀도

| 방식 | 양자화 간격 | 오차 |
|------|-------------|------|
| Single-Head | 56.9mm | ±28mm |
| Dual-Head (Fractional) | **3.92mm** | **±2mm** |

**개선율**: **14배**

---

## ✅ Success Criteria

### 필수
- [ ] FP32 abs_rel < 0.045
- [ ] INT8 abs_rel < 0.065
- [ ] 모든 테스트 통과

### 선택
- [ ] FP32 abs_rel < 0.040 (초과 달성)
- [ ] INT8 abs_rel < 0.060 (초과 달성)

---

**이 Quick Reference는 구현 중 자주 참조할 핵심 정보만 담고 있습니다.**  
**상세 내용은 각 문서를 참조하세요.**
