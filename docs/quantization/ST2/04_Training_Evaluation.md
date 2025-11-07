# 4. 학습 및 평가

## 학습 실행

### 4.1. 학습 명령어

```bash
cd /workspace/packnet-sfm

# Dual-Head 모델 학습
python scripts/train.py \
    configs/train_resnet_san_ncdb_dual_head_640x384.yaml

# 학습 진행 확인
tail -f checkpoints/resnetsan01_dual_head_640x384/training.log
```

### 4.2. 학습 파라미터

**권장 설정**:

```yaml
# configs/train_resnet_san_ncdb_dual_head_640x384.yaml
datasets:
    train:
        batch_size: 4  # 기존과 동일
    validation:
        batch_size: 4

optimizer:
    name: 'Adam'
    learning_rate: 2.0e-4  # 기존과 동일
    weight_decay: 0.0

scheduler:
    name: 'StepLR'
    step_size: 15
    gamma: 0.1

checkpoint:
    save_top_k: 3
    monitor: 'abs_rel'
    mode: 'min'

trainer:
    max_epochs: 30
    gradient_clip_val: 1.0
    check_val_every_n_epoch: 1
```

---

## 학습 모니터링

### 4.3. 주요 메트릭

| Epoch | Integer Loss | Fractional Loss | Consistency Loss | Val abs_rel |
|-------|--------------|-----------------|------------------|-------------|
| 1 | 0.050 | 0.080 | 0.120 | ~0.150 |
| 5 | 0.010 | 0.040 | 0.060 | ~0.120 |
| 10 | 0.005 | 0.020 | 0.030 | ~0.090 |
| 15 | 0.003 | 0.015 | 0.020 | ~0.070 |
| 20 | 0.002 | 0.010 | 0.015 | ~0.060 |
| 25 | 0.001 | 0.007 | 0.012 | ~0.057 |
| **30** | **0.001** | **0.005** | **0.010** | **~0.055** |

**기대 사항**:
- **Integer Loss**: 빠르게 수렴 (Epoch 5에 0.01 이하)
- **Fractional Loss**: 천천히 감소 (핵심 정밀도)
- **Consistency Loss**: 안정적으로 감소
- **Val abs_rel**: 30 epoch에 0.055 달성 목표

### 4.3.1. 학습 이상 탐지 기준 (Health Check)

**🟢 Epoch 5 체크포인트**:

| 메트릭 | 정상 (✅) | 경고 (⚠️) | 비정상 (❌) |
|--------|----------|----------|-----------|
| Integer Loss | < 0.012 | 0.012~0.020 | > 0.020 |
| Fractional Loss | < 0.045 | 0.045~0.060 | > 0.060 |
| Consistency Loss | < 0.065 | 0.065~0.080 | > 0.080 |
| Val abs_rel | < 0.125 | 0.125~0.140 | > 0.140 |

**조치 사항**:
- ✅ **정상**: 계속 학습
- ⚠️ **경고**: 로그 확인, 다음 체크포인트 주의 깊게 관찰
- ❌ **비정상**: 학습 중단, Troubleshooting 섹션 참조

**🟡 Epoch 10 체크포인트**:

| 메트릭 | 정상 (✅) | 경고 (⚠️) | 비정상 (❌) |
|--------|----------|----------|-----------|
| Integer Loss | < 0.007 | 0.007~0.015 | > 0.015 |
| Fractional Loss | < 0.025 | 0.025~0.035 | > 0.035 |
| Consistency Loss | < 0.035 | 0.035~0.045 | > 0.045 |
| Val abs_rel | < 0.095 | 0.095~0.110 | > 0.110 |

**🔵 Epoch 20 체크포인트** (최종 수렴 확인):

| 메트릭 | 정상 (✅) | 경고 (⚠️) | 비정상 (❌) |
|--------|----------|----------|-----------|
| Integer Loss | < 0.003 | 0.003~0.005 | > 0.005 |
| Fractional Loss | < 0.012 | 0.012~0.018 | > 0.018 |
| Consistency Loss | < 0.018 | 0.018~0.025 | > 0.025 |
| Val abs_rel | < 0.065 | 0.065~0.075 | > 0.075 |

**비정상 상황 대응**:

1. **Integer Loss가 높음** (> 임계값):
   - 원인: Learning rate가 너무 낮음, 또는 max_depth 설정 오류
   - 조치: LR 증가 (2e-4 → 3e-4), max_depth 확인

2. **Fractional Loss가 높음** (> 임계값):
   - 원인: `fractional_weight`가 낮음 (기본값 10.0 미만)
   - 조치: `fractional_weight`를 15.0~20.0으로 증가

3. **Val abs_rel이 정체** (20 epoch 이후도 > 0.075):
   - 원인: 과적합 또는 데이터 품질 문제
   - 조치: Early stopping, 데이터셋 검증

### 4.4. TensorBoard 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir checkpoints/resnetsan01_dual_head_640x384

# 주요 확인 사항:
# 1. Loss curves: Integer/Fractional/Consistency 모두 감소 추세
# 2. Validation metrics: abs_rel, rmse, δ<1.25
# 3. Learning rate schedule: Step decay 확인
# 4. Gradient norms: 폭발하지 않는지 확인
```

### 4.5. 학습 중 체크포인트

```bash
# 최고 성능 모델 확인
ls -lh checkpoints/resnetsan01_dual_head_640x384/*.ckpt

# 중간 평가 (Epoch 15)
python scripts/eval.py \
    --checkpoint checkpoints/resnetsan01_dual_head_640x384/epoch_15.ckpt \
    --config configs/train_resnet_san_ncdb_dual_head_640x384.yaml

# 최종 평가 (Epoch 30)
python scripts/eval.py \
    --checkpoint checkpoints/resnetsan01_dual_head_640x384/epoch_30.ckpt \
    --config configs/train_resnet_san_ncdb_dual_head_640x384.yaml
```

---

## 평가 프로세스

### 4.6. FP32 평가 (PyTorch)

**⚠️ 중요: 공식 평가 스크립트 사용**

Dual-Head 모델의 FP32 성능을 평가할 때는 **`scripts/eval_official.py`를 수정**하여 사용해야 합니다.

#### 방법 1: Validation Set 평가 (권장)

```bash
# eval_official.py를 사용하여 validation set 평가
python scripts/eval_official.py \
    --checkpoint checkpoints/resnetsan01_dual_head_640x384/epoch_30.ckpt \
    --config configs/train_resnet_san_ncdb_dual_head_640x384.yaml \
    --split val
```

**`eval_official.py` 수정 사항**:

기존 파일은 `val`/`test` split을 지원합니다. Dual-Head 모델에 대해서는 수정 불필요하지만, 
`use_dual_head=true` 설정이 YAML에 포함되어 있는지 확인하세요.

```python
# scripts/eval_official.py (기존 파일 사용 가능)
#!/usr/bin/env python3
"""
Official evaluation script modified to support validation set evaluation
Based on scripts/eval.py
"""

import argparse
import torch

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.horovod import hvd_init


def parse_args():
    """Parse arguments for evaluation script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM evaluation script (with val/test support)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint (.ckpt)')
    parser.add_argument('--config', type=str, default=None, help='Configuration (.yaml)')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                       help='Dataset split to evaluate (val or test)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    args = parser.parse_args()
    return args


def evaluate(ckpt_file, cfg_file, split, half):
    """Evaluation function"""
    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(ckpt_file, cfg_file)

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_state_dict(state_dict, strict=False)

    # Change to half precision if requested
    config.arch["dtype"] = torch.float16 if half else None

    # Create trainer
    trainer = HorovodTrainer(**config.arch)

    # Choose evaluation method based on split
    if split == 'val':
        print("\n" + "="*80)
        print(f"📊 VALIDATION SET EVALUATION")
        print("="*80)
        
        # Send module to GPU
        model_wrapper = model_wrapper.to('cuda', dtype=trainer.dtype)
        # Get validation dataloaders
        val_dataloaders = model_wrapper.val_dataloader()
        # Run validation
        trainer.validate(val_dataloaders, model_wrapper)
        
    else:  # test
        print("\n" + "="*80)
        print(f"📊 TEST SET EVALUATION")
        print("="*80)
        
        # Use standard test method
        trainer.test(model_wrapper)


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.checkpoint, args.config, args.split, args.half)
```

#### 방법 2: PyTorch 예측 생성 후 별도 평가

```bash
# Step 1: PyTorch 모델로 예측 생성 (.npy 파일)
python scripts/generate_pytorch_predictions.py \
    --checkpoint checkpoints/resnetsan01_dual_head_640x384/epoch_30.ckpt \
    --config configs/train_resnet_san_ncdb_dual_head_640x384.yaml \
    --output_dir outputs/pytorch_fp32_predictions

# Step 2: 생성된 예측을 평가
python scripts/evaluate_predictions.py \
    --pred_dir outputs/pytorch_fp32_predictions \
    --test_json /workspace/data/ncdb-cls-640x384/splits/combined_test.json
```

**`generate_pytorch_predictions.py` 사용 방법**:

이 스크립트는 **공식 평가 파이프라인과 동일한 방식**으로 예측을 생성합니다:

```python
# scripts/generate_pytorch_predictions.py (기존 파일 활용)
"""
Generate PyTorch FP32 predictions using the same pipeline as official eval.
This ensures predictions match exactly what the official evaluation uses.
"""
# (기존 파일 참조 - 수정 불필요)
```

**사용 예시**:

```bash
# Dual-Head 모델로 예측 생성
python scripts/generate_pytorch_predictions.py \
    --checkpoint checkpoints/resnetsan01_dual_head_640x384/epoch_30.ckpt \
    --config configs/train_resnet_san_ncdb_dual_head_640x384.yaml \
    --output_dir outputs/dual_head_fp32_predictions

# 출력: outputs/dual_head_fp32_predictions/*.npy (각 이미지별 depth map)
```

#### 예상 FP32 결과

```json
{
    "abs_rel": 0.038,
    "sq_rel": 0.045,
    "rmse": 0.350,
    "rmse_log": 0.055,
    "a1": 0.982,
    "a2": 0.996,
    "a3": 0.999
}
```

**🔑 핵심 포인트**:

1. **`eval_official.py` 사용**:
   - Validation set 평가에 최적화
   - 공식 평가 파이프라인과 동일
   - `--split val` 또는 `--split test` 선택 가능

2. **`generate_pytorch_predictions.py` 사용**:
   - NPU 결과와 직접 비교 가능한 .npy 파일 생성
   - 동일한 후처리 적용 보장
   - 디버깅 및 분석에 유용

3. **YAML 설정 확인**:
   ```yaml
   depth_net:
       name: 'ResNetSAN01'
       use_dual_head: true  # ✅ 필수!
   ```

### 4.7. ONNX Export

```bash
# Dual-Head 모델을 ONNX로 변환
python scripts/export_to_onnx.py \
    --checkpoint checkpoints/resnetsan01_dual_head_640x384/epoch_30.ckpt \
    --output onnx/resnetsan_dual_head.onnx \
    --dual_head  # 🆕 Dual output 플래그
```

**Export 스크립트 수정 필요**:

```python
# scripts/export_to_onnx.py (수정 필요)
import torch
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01

# Load model
model = ResNetSAN01(version='18A', use_dual_head=True, max_depth=15.0)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 384, 640)

# Export with dual outputs
torch.onnx.export(
    model,
    dummy_input,
    args.output,
    input_names=['rgb'],
    output_names=['integer_sigmoid', 'fractional_sigmoid'],  # 🆕 명시
    dynamic_axes={'rgb': {0: 'batch_size'}},
    opset_version=11
)
```

### 4.8. NPU 변환 및 평가

```bash
# ONNX → NPU 변환 (Pulsar2 사용)
pulsar2 build \
    --input onnx/resnetsan_dual_head.onnx \
    --output npu/resnetsan_dual_head.joint \
    --config configs/npu_config_dual_head.json \
    --calibration_data calibration_data_300/

# NPU INT8 평가
python scripts/evaluate_npu_dual_head.py \
    --npu_model npu/resnetsan_dual_head.joint \
    --output_dir outputs/dual_head_npu_results
```

**⚠️ NPU 평가 스크립트 작성 방법**:

기존 `scripts/evaluate_npu_direct_depth_official.py`를 **수정**하여 사용해야 합니다:

```python
# scripts/evaluate_npu_dual_head.py
"""
NPU Dual-Head 모델 평가
Integer/Fractional 두 출력을 받아서 depth 복원

📝 NOTE: 
기존 evaluate_npu_direct_depth_official.py를 참고하되, 
Dual-Head 출력 처리 로직으로 수정 필요
"""

import numpy as np
import json
import torch
from pathlib import Path
from PIL import Image
from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth


def load_gt_depth(new_filename, test_json_path):
    """GT depth 로드 (combined_test.json 기반)"""
    with open(test_json_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        if entry['new_filename'] == new_filename:
            dataset_root = entry['dataset_root']
            depth_path = Path(dataset_root) / 'newest_depth_maps' / f'{new_filename}.png'
            
            if not depth_path.exists():
                raise FileNotFoundError(f"GT depth not found: {depth_path}")
            
            # PNG 16-bit 로드 → meters
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img, dtype=np.float32) / 256.0
            
            return depth
    
    raise ValueError(f"new_filename {new_filename} not found in {test_json_path}")


def compute_depth_metrics(gt, pred, min_depth=0.5, max_depth=15.0):
    """공식 eval.py의 compute_depth_metrics() 재현"""
    valid_mask = (gt > min_depth) & (gt < max_depth)
    
    if valid_mask.sum() == 0:
        return None
    
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    
    # Metrics
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'valid_pixels': int(valid_mask.sum())
    }


def main():
    import argparse
    import onnxruntime as ort
    
    parser = argparse.ArgumentParser(description='Evaluate NPU Dual-Head model')
    parser.add_argument('--npu_dir', type=str, required=True,
                       help='NPU output directory with .npy files')
    parser.add_argument('--test_json', type=str, 
                       default='/workspace/data/ncdb-cls-640x384/splits/combined_test.json',
                       help='Test JSON path')
    parser.add_argument('--min_depth', type=float, default=0.5, help='Min depth')
    parser.add_argument('--max_depth', type=float, default=15.0, help='Max depth')
    args = parser.parse_args()
    
    # Configuration
    npu_output_dir = Path(args.npu_dir)
    test_json = args.test_json
    min_depth = args.min_depth
    max_depth = args.max_depth
    
    # NPU 출력 파일 로드 (integer_*.npy, fractional_*.npy 형식)
    integer_files = sorted(npu_output_dir.glob('integer_*.npy'))
    fractional_files = sorted(npu_output_dir.glob('fractional_*.npy'))
    
    print("="*80)
    print("🚀 NPU Dual-Head 평가")
    print("="*80)
    print(f"📁 NPU output dir: {npu_output_dir}")
    print(f"📊 Depth range: [{min_depth}, {max_depth}]m")
    print(f"📊 Integer outputs: {len(integer_files)}")
    print(f"📊 Fractional outputs: {len(fractional_files)}")
    print()
    
    all_metrics = []
    
    for int_file, frac_file in zip(integer_files, fractional_files):
        # 파일명에서 new_filename 추출
        new_filename = int_file.stem.replace('integer_', '')
        
        # Load NPU Dual-Head outputs
        integer_sigmoid = np.load(int_file)
        fractional_sigmoid = np.load(frac_file)
        
        # Shape normalization
        while integer_sigmoid.ndim > 2:
            integer_sigmoid = integer_sigmoid.squeeze(0)
        while fractional_sigmoid.ndim > 2:
            fractional_sigmoid = fractional_sigmoid.squeeze(0)
        
        # Convert to torch tensors
        integer_sigmoid = torch.from_numpy(integer_sigmoid).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        fractional_sigmoid = torch.from_numpy(fractional_sigmoid).unsqueeze(0).unsqueeze(0)
        
        # Depth 복원 (dual_head_to_depth 사용)
        depth_pred = dual_head_to_depth(
            integer_sigmoid, fractional_sigmoid, max_depth=max_depth
        )
        depth_pred = depth_pred.squeeze().numpy()
        
        # Load GT depth
        try:
            gt_depth = load_gt_depth(new_filename, test_json)
        except Exception as e:
            print(f"⚠️  SKIP: {new_filename} - GT loading failed: {e}")
            continue
        
        # Compute metrics
        metrics = compute_depth_metrics(gt_depth, depth_pred, min_depth, max_depth)
        
        if metrics is None:
            print(f"⚠️  SKIP: {new_filename} - No valid pixels")
            continue
        
        all_metrics.append(metrics)
        
        print(f"✅ {new_filename}")
        print(f"   abs_rel: {metrics['abs_rel']:.4f}, "
              f"rmse: {metrics['rmse']:.4f}m, δ<1.25: {metrics['a1']:.4f}")
        print()
    
    if not all_metrics:
        print("❌ No valid results")
        return
    
    # Average metrics
    print("="*80)
    print("📊 AVERAGE METRICS (NPU Dual-Head INT8)")
    print("="*80)
    
    for key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
        avg_val = np.mean([m[key] for m in all_metrics])
        print(f"   {key:12s}: {avg_val:.4f}")
    
    print(f"\n✅ Total evaluated: {len(all_metrics)} images")
    print("="*80)


if __name__ == '__main__':
    main()
```

**🔑 핵심 차이점**:

1. **기존 Direct Depth 스크립트**:
   - 단일 출력 (.npy 파일)
   - Depth 변환 없음 (이미 Linear depth)

2. **Dual-Head 스크립트** (수정 필요):
   - 두 개 출력 (integer_*.npy, fractional_*.npy)
   - `dual_head_to_depth()` 함수로 depth 복원 필요
   
**사용 방법**:

```bash
# 1. NPU 추론 실행 (두 출력 저장)
python scripts/run_npu_inference_dual_head.py \
    --npu_model npu/resnetsan_dual_head.joint \
    --output_dir outputs/dual_head_npu_outputs

# 2. 평가 실행
python scripts/evaluate_npu_dual_head.py \
    --npu_dir outputs/dual_head_npu_outputs \
    --test_json /workspace/data/ncdb-cls-640x384/splits/combined_test.json
```

---

## 예상 결과

### 4.9. FP32 성능 (PyTorch)

| Metric | Single-Head (Baseline) | Dual-Head (Expected) | Improvement |
|--------|------------------------|----------------------|-------------|
| **abs_rel** | 0.0434 | **0.038~0.042** | **10-15%** |
| **rmse** | 0.391m | **0.35~0.38m** | **10-15%** |
| **δ<1.25** | 0.9759 | **0.980~0.985** | **+0.5%** |

**분석**:
- Dual-Head는 FP32에서도 약간의 성능 향상 예상
- 이유: 더 명시적인 표현 (정수부 + 소수부)
- 특히 중거리(5-10m)에서 정밀도 향상

### 4.10. INT8 성능 (NPU)

| Metric | Phase 1 (300 cal) | Dual-Head INT8 | Improvement |
|--------|-------------------|----------------|-------------|
| **abs_rel** | 0.1139 | **0.055~0.065** | **47-52%** |
| **rmse** | 0.751m | **0.45~0.55m** | **33-40%** |
| **δ<1.25** | 0.9061 | **0.965~0.975** | **6-7%** |

**목표 달성 평가**:

| 목표 | 현재 | 예상 | 달성 가능성 |
|------|------|------|-------------|
| abs_rel < 0.09 | 0.1139 | **0.055~0.065** | ✅ **높음** |
| FP32 대비 격차 축소 | 2.6x | **1.5x** | ✅ **달성** |
| 양자화 오차 감소 | ±28mm | **±2mm** | ✅ **14배 개선** |

### 4.11. 정밀도 분석

**양자화 간격 비교**:

| 방식 | Integer Head | Fractional Head | 전체 정밀도 |
|------|--------------|-----------------|-------------|
| **Single-Head** | N/A | N/A | 56.9mm |
| **Dual-Head** | 58.8mm (15/255) | **3.92mm** (1/255) | **±2mm** |

**거리별 오차 분석**:

| 거리 범위 | Single-Head 오차 | Dual-Head 오차 | 개선율 |
|-----------|------------------|----------------|--------|
| 0-1m | ±28mm | ±2mm | **14배** |
| 1-5m | ±28mm | ±2mm | **14배** |
| 5-10m | ±28mm | ±2mm | **14배** |
| 10-15m | ±28mm | ±2mm | **14배** |

**핵심 인사이트**:
- Fractional Head가 전체 정밀도 결정 (3.92mm)
- 모든 거리 범위에서 균일한 정밀도 (거리 독립적)
- Integer Head는 정확한 미터 단위 선택 담당

---

## 학습 스케줄

### Week 1: 구현 및 테스트 (Day 1-5)

- **Day 1**: DualHeadDepthDecoder 구현 및 테스트
- **Day 2**: Helper functions 및 단위 테스트
- **Day 3**: ResNetSAN01 통합 및 통합 테스트
- **Day 4**: Loss function 구현 및 검증
- **Day 5**: YAML config 준비 및 학습 시작

### Week 2: 학습 (Day 6-12)

- **Day 6-8**: 초기 학습 (Epoch 1-10)
  - Integer loss 수렴 확인
  - Fractional loss 감소 추세 확인
  
- **Day 9-10**: 중기 학습 (Epoch 11-20)
  - Validation metrics 모니터링
  - Learning rate schedule 확인
  
- **Day 11-12**: 후기 학습 (Epoch 21-30)
  - 최종 수렴 확인
  - Best checkpoint 선정

### Week 3: 평가 및 배포 (Day 13-15)

- **Day 13**: FP32 평가
  - Validation set 평가
  - Test set 평가
  - 메트릭 비교 (vs Single-Head)
  
- **Day 14**: NPU 변환
  - ONNX export
  - NPU quantization
  - INT8 정확도 검증
  
- **Day 15**: 최종 평가 및 분석
  - NPU test set 평가
  - 성능 리포트 작성
  - 목표 달성 여부 확인

---

## Success Criteria

### 필수 조건

- ✅ **FP32 abs_rel < 0.045**: Dual-Head 모델이 baseline 유지 또는 개선
- ✅ **INT8 abs_rel < 0.065**: 목표 달성 (현재 0.1139 대비 47% 개선)
- ✅ **양자화 오차 < 5mm**: Fractional head 정밀도 (목표 3.92mm)
- ✅ **Backward compatibility**: 기존 코드 정상 동작

### 선택 조건

- 🎯 **FP32 abs_rel < 0.040**: 초과 달성
- 🎯 **INT8 abs_rel < 0.060**: 초과 달성
- 🎯 **FP32 대비 격차 < 1.5배**: 현재 2.6배 대비 대폭 개선

### 실패 기준

- ❌ **FP32 abs_rel > 0.050**: Baseline 대비 성능 저하
- ❌ **INT8 abs_rel > 0.090**: 목표 미달성
- ❌ **학습 불안정**: NaN loss, gradient explosion 등

→ **다음**: [Troubleshooting](05_Troubleshooting.md)
