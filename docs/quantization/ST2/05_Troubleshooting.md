# 5. Troubleshooting

## 학습 중 문제

### 문제 1: Integer Loss가 감소하지 않음

**증상**:
```
Epoch 10: integer_loss=0.05, fractional_loss=0.02, consistency_loss=0.03
Epoch 20: integer_loss=0.05, fractional_loss=0.01, consistency_loss=0.015
```
Integer loss가 0.05 이상에서 멈춤

**원인**:
- `max_depth` 설정이 실제 데이터 범위와 불일치
- Integer head가 잘못된 범위로 정규화됨

**해결 방법**:

```bash
# 1. 데이터셋의 실제 depth 범위 확인
python -c "
import numpy as np
from packnet_sfm.datasets.ncdb_dataset import NCDBDataset

dataset = NCDBDataset(...)
depths = [sample['depth'].numpy() for sample in dataset[:100]]
print(f'Min depth: {np.min(depths):.2f}m')
print(f'Max depth: {np.max(depths):.2f}m')
print(f'Mean depth: {np.mean(depths):.2f}m')
"

# 2. YAML의 max_depth 수정
# configs/train_resnet_san_ncdb_dual_head_640x384.yaml
params:
    max_depth: 15.0  # 실제 데이터 범위에 맞춰 조정
```

**검증**:
```python
# Integer GT 분포 확인
from packnet_sfm.networks.layers.resnet.layers import decompose_depth
import torch

depth_samples = torch.randn(100, 1, 384, 640) * 10  # 예시
integer_gt, frac_gt = decompose_depth(depth_samples, max_depth=15.0)

print(f"Integer GT range: [{integer_gt.min():.3f}, {integer_gt.max():.3f}]")
# 예상: [0.0, 1.0] 범위에 균등 분포
```

---

### 문제 2: Fractional Loss가 너무 높음

**증상**:
```
Epoch 30: fractional_loss=0.08 (너무 높음, 목표: 0.005)
```

**원인**:
1. Fractional weight가 너무 낮음 (모델이 소수부를 무시)
2. Learning rate가 너무 높음 (overshooting)

**해결 방법 1: Weight 조정**

```python
# packnet_sfm/losses/dual_head_depth_loss.py
class DualHeadDepthLoss(LossBase):
    def __init__(self, ..., fractional_weight=10.0, ...):
        # 기존: 10.0
        # 시도: 15.0 또는 20.0으로 증가
        self.fractional_weight = fractional_weight
```

**해결 방법 2: Learning Rate 감소**

```yaml
# configs/train_resnet_san_ncdb_dual_head_640x384.yaml
optimizer:
    learning_rate: 1.0e-4  # 기존: 2.0e-4에서 절반으로 감소
```

**검증**:
```bash
# Loss 비율 확인
python -c "
# Training log에서 loss 비율 확인
# 이상적인 비율: integer:fractional:consistency = 1:10:5
# 실제가 1:2:5라면 fractional weight 증가 필요
"
```

---

### 문제 3: NaN Loss

**증상**:
```
Epoch 5: loss=NaN, integer_loss=NaN
RuntimeError: Found NaN in loss
```

**원인**:
1. Ground truth depth에 무한대 또는 0 값 포함
2. Division by zero in inverse depth conversion
3. Gradient explosion

**해결 방법 1: GT 데이터 검증**

```python
# packnet_sfm/datasets/ncdb_dataset.py
def __getitem__(self, idx):
    # ... 기존 코드 ...
    
    # 🆕 Depth 유효성 검사
    depth = sample['depth']
    
    # 무한대 제거
    depth = torch.where(torch.isinf(depth), torch.zeros_like(depth), depth)
    
    # 유효 범위 클리핑
    depth = torch.clamp(depth, min=0.5, max=15.0)
    
    # NaN 제거
    depth = torch.where(torch.isnan(depth), torch.zeros_like(depth), depth)
    
    sample['depth'] = depth
    return sample
```

**해결 방법 2: Loss 함수에 안전장치 추가**

```python
# packnet_sfm/losses/dual_head_depth_loss.py
def forward(self, outputs, depth_gt, ...):
    # ... 기존 코드 ...
    
    # 🆕 NaN 체크
    if torch.isnan(depth_gt).any() or torch.isinf(depth_gt).any():
        print("⚠️ Warning: NaN or Inf in GT depth, skipping batch")
        return {
            'loss': torch.tensor(0.0, device=depth_gt.device, requires_grad=True),
            'integer_loss': torch.tensor(0.0),
            'fractional_loss': torch.tensor(0.0),
            'consistency_loss': torch.tensor(0.0)
        }
    
    # Valid mask 강화
    mask = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
    mask = mask & (~torch.isnan(depth_gt)) & (~torch.isinf(depth_gt))
    
    if mask.sum() == 0:
        # No valid pixels
        return {...}
    
    # ... 나머지 코드 ...
```

**해결 방법 3: Gradient Clipping**

```yaml
# configs/train_resnet_san_ncdb_dual_head_640x384.yaml
trainer:
    gradient_clip_val: 1.0  # 🆕 Gradient norm 제한
    gradient_clip_algorithm: 'norm'
```

---

### 문제 4: 학습이 너무 느림

**증상**:
- Single-Head: 5 min/epoch
- Dual-Head: 12 min/epoch (2.4배 느림)

**원인**:
- Dual-Head는 2배의 출력 헤드를 계산
- 추가 loss 계산 (integer + fractional + consistency)

**해결 방법 1: Batch Size 증가 (GPU 메모리 허용 시)**

```yaml
datasets:
    train:
        batch_size: 8  # 기존: 4에서 2배 증가
```

**해결 방법 2: Multi-Scale Loss 비활성화**

```yaml
model:
    loss:
        supervised_num_scales: 1  # 기존: 4에서 1로 감소 (scale 0만 사용)
```

**해결 방법 3: Mixed Precision Training**

```yaml
trainer:
    precision: 16  # 🆕 FP16 혼합 정밀도 학습
```

---

## 코드 통합 문제

### 문제 5: ModuleNotFoundError

**증상**:
```python
ModuleNotFoundError: No module named 'packnet_sfm.networks.layers.resnet.dual_head_depth_decoder'
```

**원인**:
- 파일이 생성되지 않았거나 경로 오류

**해결 방법**:

```bash
# 1. 파일 존재 확인
ls -la packnet_sfm/networks/layers/resnet/dual_head_depth_decoder.py

# 2. __init__.py 확인
cat packnet_sfm/networks/layers/resnet/__init__.py

# 3. __init__.py가 없으면 생성
touch packnet_sfm/networks/layers/resnet/__init__.py

# 4. Python path 확인
python -c "import sys; print('\n'.join(sys.path))"

# 5. 프로젝트 루트에서 실행하는지 확인
pwd  # /workspace/packnet-sfm이어야 함
```

---

### 문제 6: KeyError in outputs

**증상**:
```python
KeyError: ("integer", 0)
```

**원인**:
- 모델이 여전히 Single-Head로 로딩됨
- YAML의 `use_dual_head` 파라미터가 전달되지 않음

**해결 방법**:

```bash
# 1. 모델 초기화 확인
python -c "
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01

model = ResNetSAN01(version='18A', use_dual_head=True, max_depth=15.0)
print(f'is_dual_head: {model.is_dual_head}')  # True여야 함
print(f'Decoder type: {type(model.decoder).__name__}')  # DualHeadDepthDecoder여야 함
"

# 2. YAML config 확인
cat configs/train_resnet_san_ncdb_dual_head_640x384.yaml | grep use_dual_head
# 출력: use_dual_head: true

# 3. Config 로딩 확인
python -c "
from packnet_sfm.utils.config import parse_train_file

config = parse_train_file('configs/train_resnet_san_ncdb_dual_head_640x384.yaml')
print(config['model']['depth_net'])
"
```

**디버깅 코드 추가**:

```python
# packnet_sfm/networks/depth/ResNetSAN01.py
def __init__(self, ..., use_dual_head=False, **kwargs):
    print(f"🔍 ResNetSAN01 init: use_dual_head={use_dual_head}")  # 🆕 디버깅
    
    if use_dual_head:
        print("✅ Creating DualHeadDepthDecoder")  # 🆕
        self.decoder = DualHeadDepthDecoder(...)
        self.is_dual_head = True
    else:
        print("✅ Creating standard DepthDecoder")  # 🆕
        self.decoder = DepthDecoder(...)
        self.is_dual_head = False
```

---

### 문제 7: Checkpoint 로딩 실패

**증상**:
```python
RuntimeError: Error(s) in loading state_dict:
    size mismatch for decoder.convs.("integer_conv", 0).conv.weight
```

**원인**:
- Single-Head checkpoint를 Dual-Head 모델에 로딩하려 함
- 또는 그 반대

**해결 방법 1: Strict Loading 비활성화**

```python
# scripts/train.py 또는 eval.py
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['state_dict'], strict=False)  # 🆕 strict=False
```

**해결 방법 2: Checkpoint 변환 스크립트**

```python
# scripts/convert_checkpoint.py
import torch

# Single-Head → Dual-Head 변환
checkpoint = torch.load('single_head.ckpt')
state_dict = checkpoint['state_dict']

# Decoder weights만 제거 (나머지는 재사용)
new_state_dict = {k: v for k, v in state_dict.items() if 'decoder' not in k}

# 새 checkpoint 저장
torch.save({'state_dict': new_state_dict}, 'dual_head_init.ckpt')
```

**해결 방법 3: From Scratch 학습**

```yaml
# configs/train_resnet_san_ncdb_dual_head_640x384.yaml
checkpoint:
    resume: null  # Checkpoint 없이 처음부터 학습
```

---

## NPU 변환 문제

### 문제 8: ONNX Export 실패

**증상**:
```python
RuntimeError: ONNX export failed: Dual output is not exported
```

**원인**:
- PyTorch → ONNX 변환 시 output_names 명시 필요

**해결 방법**:

```python
# scripts/export_to_onnx.py 수정
import torch
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01

# 모델 로딩
model = ResNetSAN01(version='18A', use_dual_head=True, max_depth=15.0)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

# Wrapper 클래스로 출력 형식 명시
class DualHeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, rgb):
        self.model.eval()
        outputs = self.model.decoder(self.model.encoder(rgb))
        
        # 명시적으로 두 출력 반환
        integer_sigmoid = outputs[("integer", 0)]
        fractional_sigmoid = outputs[("fractional", 0)]
        
        return integer_sigmoid, fractional_sigmoid

# Wrapper로 export
wrapper = DualHeadWrapper(model)
dummy_input = torch.randn(1, 3, 384, 640)

torch.onnx.export(
    wrapper,
    dummy_input,
    args.output,
    input_names=['rgb'],
    output_names=['integer_sigmoid', 'fractional_sigmoid'],  # 🆕 명시
    dynamic_axes={
        'rgb': {0: 'batch_size'},
        'integer_sigmoid': {0: 'batch_size'},
        'fractional_sigmoid': {0: 'batch_size'}
    },
    opset_version=11,
    do_constant_folding=True,
    verbose=True
)

print(f"✅ ONNX export complete: {args.output}")
```

---

### 문제 9: NPU 양자화 오류

**증상**:
```
Pulsar2 error: Calibration failed for dual outputs
```

**원인**:
- NPU가 두 출력을 독립적으로 calibration 필요

**해결 방법**:

```json
// configs/npu_config_dual_head.json
{
  "model_type": "ONNX",
  "npu_mode": "NPU3",
  "quant": {
    "input_configs": [
      {
        "tensor_name": "rgb",
        "calibration_dataset": "calibration_data_300/",
        "calibration_size": 300,
        "calibration_mean": [0.485, 0.456, 0.406],
        "calibration_std": [0.229, 0.224, 0.225]
      }
    ],
    "output_configs": [
      {
        "tensor_name": "integer_sigmoid",
        "calibration_method": "MinMax",
        "quantize_method": "PerTensor"
      },
      {
        "tensor_name": "fractional_sigmoid",
        "calibration_method": "MinMax",
        "quantize_method": "PerTensor"
      }
    ],
    "calibration_method": "MinMax",
    "precision_analysis": true
  }
}
```

---

### 문제 10: NPU 평가 결과 이상

**증상**:
```
NPU INT8: abs_rel=0.15 (예상: 0.055보다 훨씬 높음)
```

**원인**:
1. Depth 복원 로직 오류
2. Integer/Fractional 출력 순서 바뀜
3. max_depth 값 불일치

**해결 방법 1: 출력 검증**

```python
# scripts/evaluate_npu_dual_head.py
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(args.npu_model)

# 출력 이름 확인
output_names = [output.name for output in session.get_outputs()]
print(f"NPU output names: {output_names}")
# 예상: ['integer_sigmoid', 'fractional_sigmoid']

# 단일 이미지 테스트
rgb_test = np.random.randn(1, 3, 384, 640).astype(np.float32)
outputs = session.run(None, {'rgb': rgb_test})

print(f"Output 0 shape: {outputs[0].shape}, range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
print(f"Output 1 shape: {outputs[1].shape}, range: [{outputs[1].min():.3f}, {outputs[1].max():.3f}]")

# Sigmoid 범위 [0, 1] 확인
assert 0.0 <= outputs[0].min() and outputs[0].max() <= 1.0, "Integer sigmoid out of range"
assert 0.0 <= outputs[1].min() and outputs[1].max() <= 1.0, "Fractional sigmoid out of range"
```

**해결 방법 2: Depth 복원 검증**

```python
# scripts/evaluate_npu_dual_head.py
from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth
import torch

# NPU 출력
integer_sigmoid = torch.from_numpy(outputs[0])
fractional_sigmoid = torch.from_numpy(outputs[1])

# Depth 복원
depth_pred = dual_head_to_depth(integer_sigmoid, fractional_sigmoid, max_depth=15.0)

print(f"Predicted depth range: [{depth_pred.min():.2f}, {depth_pred.max():.2f}]m")
# 예상: [0.0, 16.0]m (max_depth + 1)

# GT와 비교
print(f"GT depth range: [{depth_gt.min():.2f}, {depth_gt.max():.2f}]m")

# Sanity check
assert depth_pred.min() >= 0.0, "Negative depth"
assert depth_pred.max() <= 16.0, "Depth exceeds max_depth + 1"
```

---

## 일반적인 디버깅 체크리스트

### 학습 시작 전

- [ ] 모든 파일이 올바른 위치에 생성됨
- [ ] `use_dual_head=True` 확인 (YAML, 모델 초기화)
- [ ] 단위 테스트 통과
- [ ] 통합 테스트 통과
- [ ] Dummy data로 1 step forward/backward 성공

### 학습 중

- [ ] Loss 값이 NaN이 아님
- [ ] Loss가 감소 추세
- [ ] Gradient norm이 폭발하지 않음 (< 10.0)
- [ ] Validation metrics 개선
- [ ] GPU 메모리 사용량 정상 (< 11GB for V100)

### 학습 후

- [ ] FP32 abs_rel < 0.045
- [ ] Checkpoint 저장 정상
- [ ] ONNX export 성공
- [ ] NPU 변환 성공
- [ ] NPU INT8 abs_rel < 0.065

### NPU 변환 후

- [ ] ONNX 출력 개수 = 2
- [ ] 출력 이름 확인 (integer_sigmoid, fractional_sigmoid)
- [ ] 출력 값 범위 [0, 1]
- [ ] Depth 복원 로직 정상
- [ ] max_depth 값 일치 (PyTorch vs NPU)

---

## 추가 리소스

### 로그 분석 도구

```bash
# Loss 추이 그래프
python scripts/plot_training_logs.py \
    --log checkpoints/resnetsan01_dual_head_640x384/training.log

# Tensorboard
tensorboard --logdir checkpoints/resnetsan01_dual_head_640x384

# Metric 비교
python scripts/compare_metrics.py \
    --baseline outputs/single_head_fp32_results/metrics.json \
    --experiment outputs/dual_head_fp32_results/metrics.json
```

### 문의 채널

- 코드베이스 이슈: GitHub Issues
- 학습 문제: Training logs 첨부
- NPU 변환 문제: ONNX 파일 및 config 첨부

---

**이 Troubleshooting 가이드는 실제 구현 과정에서 발생할 수 있는 대부분의 문제를 다루고 있습니다. 각 문제에 대한 구체적인 해결 방법과 검증 코드가 포함되어 있어 빠른 디버깅이 가능합니다.**
