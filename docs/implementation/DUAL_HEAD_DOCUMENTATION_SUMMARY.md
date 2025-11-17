# Dual-Head Implementation Documentation Summary

## 📚 Available Documentation

### 1. **DUAL_HEAD_OUTPUT_STRUCTURE.md** (Primary Technical Reference)
- **목적**: Complete technical specification of Dual-Head architecture
- **내용**:
  - PyTorch checkpoint output format (tuple keys, multi-scale)
  - Integer and Fractional head design
  - Loss function implementation
  - Training pipeline integration
  - Backward compatibility with Single-Head

### 2. **DUAL_HEAD_OUTPUT_SUMMARY.md** (Quick Reference)
- **목적**: Quick lookup guide for developers
- **내용**:
  - One-page cheat sheet
  - Key formulas and ranges
  - Common usage patterns
  - Troubleshooting guide

### 3. **DUAL_HEAD_ONNX_CONVERSION.md** (Deployment Guide)
- **목적**: ONNX conversion and deployment reference
- **내용**:
  - ⭐ **PyTorch Checkpoint vs ONNX output comparison**
  - Two ONNX models (composed vs separate outputs)
  - Validation results (FP32 accuracy)
  - ⭐ **INT8 quantization workflow for NPU**
  - Usage examples and conversion scripts

### 4. **DUAL_HEAD_SAVE_REPORT.md** (Data Archival)
- **목적**: NPZ file saving and verification report
- **내용**:
  - 91 test samples saved as NPZ files
  - Integer/Fractional/Composed depth archival
  - Reconstruction accuracy verification

---

## 🎯 핵심 내용 요약

### Checkpoint Output Structure (Training)

```python
# PyTorch checkpoint inference
outputs = depth_net(rgb)

# Output format: Dict with tuple keys
{
    ('integer', 0): Tensor[1, 1, 384, 640],    # Full resolution
    ('fractional', 0): Tensor[1, 1, 384, 640],
    ('integer', 1): Tensor[1, 1, 192, 320],    # Half resolution
    ('fractional', 1): Tensor[1, 1, 192, 320],
    ('integer', 2): Tensor[1, 1, 96, 160],     # Quarter resolution
    ('fractional', 2): Tensor[1, 1, 96, 160],
    ('integer', 3): Tensor[1, 1, 48, 80],      # 1/8 resolution
    ('fractional', 3): Tensor[1, 1, 48, 80],
}

# Depth composition (manual)
integer_sig = outputs[('integer', 0)]
fractional_sig = outputs[('fractional', 0)]
depth = integer_sig * 15.0 + fractional_sig
```

**특징**:
- ✅ Multi-scale outputs (4 scales)
- ✅ Tuple keys for type safety
- ❌ No pre-composed depth
- 🎯 Purpose: Multi-scale supervision during training

---

### ONNX Output Structure (Inference)

#### Option 1: Separate Outputs (Recommended for NPU)

```python
# ONNX separate outputs inference
outputs = onnx_session.run(None, {'rgb': image})

integer_sigmoid = outputs[0]      # [1, 1, 384, 640]
fractional_sigmoid = outputs[1]   # [1, 1, 384, 640]
depth_composed = outputs[2]       # [1, 1, 384, 640] - pre-calculated ✅
```

**특징**:
- ✅ Single scale (full resolution only)
- ✅ Pre-composed depth provided
- ✅ Separate outputs for quantization analysis
- 🎯 Purpose: NPU deployment with INT8 quantization

#### Option 2: Composed Output (Simple Deployment)

```python
# ONNX composed output inference
outputs = onnx_session.run(None, {'rgb': image})

depth = outputs[0]  # [1, 1, 384, 640] - final depth only
```

**특징**:
- ✅ Single output (simplest)
- ✅ Pre-composed depth
- ❌ No access to integer/fractional components
- 🎯 Purpose: Simple production deployment

---

## 🔧 INT8 Quantization Workflow

### Why Use Separate Outputs ONNX?

1. **Per-head error analysis**
   - Integer head: High sensitivity (15× amplification)
   - Fractional head: Low sensitivity (1× amplification)

2. **Quantization impact tracking**
   ```
   Δ_depth = Δ_integer × 15.0 + Δ_fractional
              ↑                 ↑
         Major contributor   Minor contributor
   ```

3. **Independent optimization**
   - Different calibration methods per head
   - Per-head quantization range tuning

### Recommended Workflow

```bash
# Step 1: Convert to separate outputs ONNX
python scripts/convert_dual_head_to_onnx.py \
    --checkpoint checkpoints/.../epoch=28_..._val-loss=0.000.ckpt \
    --separate_outputs \
    --input_shape 384 640 \
    --max_depth 15.0

# Step 2: Save FP32 reference (all 3 outputs per image)
python scripts/save_fp32_references.py \
    --onnx onnx/dual_head_..._separate_zero.onnx \
    --output_dir outputs/fp32_reference

# Step 3: Convert to INT8 using NPU toolkit
your_npu_converter --input onnx/dual_head_..._separate_zero.onnx

# Step 4: Run INT8 inference (save all 3 outputs per image)
your_npu_runner --model model_int8.bin

# Step 5: Compare FP32 vs INT8
python scripts/compare_fp32_int8.py \
    --fp32_dir outputs/fp32_reference \
    --int8_dir outputs/int8_inference
```

### What to Save for Comparison

```python
# For each test image:

# FP32 ONNX
fp32_outputs = onnx_session.run(None, {'rgb': image})
np.savez(f'{sample_id}_fp32.npz',
         integer_sigmoid=fp32_outputs[0],
         fractional_sigmoid=fp32_outputs[1],
         depth_composed=fp32_outputs[2])

# INT8 NPU
int8_outputs = npu_inference(image)
np.savez(f'{sample_id}_int8.npz',
         integer_sigmoid=int8_outputs[0],
         fractional_sigmoid=int8_outputs[1],
         depth_composed=int8_outputs[2])
```

### Analysis Metrics

```python
fp32 = np.load(f'{sample_id}_fp32.npz')
int8 = np.load(f'{sample_id}_int8.npz')

# Per-head errors
int_error = np.abs(fp32['integer_sigmoid'] - int8['integer_sigmoid'])
frac_error = np.abs(fp32['fractional_sigmoid'] - int8['fractional_sigmoid'])
depth_error = np.abs(fp32['depth_composed'] - int8['depth_composed'])

# Error contribution
int_contribution = int_error.mean() * 15.0
frac_contribution = frac_error.mean()

print(f"Integer contribution: {int_contribution/depth_error.mean()*100:.1f}%")
print(f"Fractional contribution: {frac_contribution/depth_error.mean()*100:.1f}%")
```

---

## 📊 Validation Results

### FP32 Accuracy (ONNX vs PyTorch)

**Separate outputs model**:
```
Integer sigmoid:    error < 1e-6  ✅
Fractional sigmoid: error < 1e-5  ✅
Composed depth:     error < 1e-5  ✅
Composition check:  error = 0     ✅
```

**Composed output model**:
```
Composed depth:     error < 1e-5  ✅
```

### Real KITTI Images Tested

- Image 1 (0000000147.png): ✅ Perfect match
- Image 2 (0000000655.png): ✅ Perfect match
- Random samples (10 tests): ✅ All passed

---

## 🚀 Quick Start Guide

### For Training
```python
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01

model = ResNetSAN01(use_dual_head=True, version='18A')
outputs = model(rgb)

# Access full resolution outputs
int_sig = outputs[('integer', 0)]
frac_sig = outputs[('fractional', 0)]
depth = int_sig * 15.0 + frac_sig
```

### For Inference (PyTorch)
```python
checkpoint = torch.load('epoch=28_..._val-loss=0.000.ckpt')
# ... load model ...
outputs = model(rgb)
depth = outputs[('integer', 0)] * 15.0 + outputs[('fractional', 0)]
```

### For Inference (ONNX)
```python
import onnxruntime as ort

session = ort.InferenceSession('dual_head_..._separate_zero.onnx')
outputs = session.run(None, {'rgb': image})

integer_sig = outputs[0]
fractional_sig = outputs[1]
depth = outputs[2]  # Pre-calculated ✅
```

### For INT8 NPU Deployment
```python
# Use separate outputs ONNX
# Save all 3 outputs from both FP32 and INT8
# Compare per-head errors to optimize quantization
```

---

## ✅ Checklist for Documentation Review

- [x] Checkpoint output structure explained (tuple keys, multi-scale)
- [x] ONNX output structure explained (array outputs, single scale)
- [x] Difference between checkpoint and ONNX clearly documented
- [x] INT8 quantization workflow provided
- [x] Separate outputs recommendation for NPU deployment
- [x] Per-head error analysis methodology explained
- [x] Code examples for all use cases provided
- [x] Validation results documented
- [x] Real image testing completed

---

## 📞 문서 활용 가이드

### NPU 업체에 전달할 문서
1. **DUAL_HEAD_ONNX_CONVERSION.md** - 주요 문서
   - PyTorch vs ONNX 차이점
   - INT8 양자화 워크플로우
   - 3개 출력 저장 방법
   - 오차 분석 방법

2. **DUAL_HEAD_OUTPUT_SUMMARY.md** - Quick reference
   - 빠른 참조용

### 내부 개발팀 참고 문서
1. **DUAL_HEAD_OUTPUT_STRUCTURE.md** - 전체 기술 사양
2. **DUAL_HEAD_SAVE_REPORT.md** - 데이터 아카이브 리포트

---

## 🎯 결론

✅ **모든 핵심 내용이 문서화되어 있습니다**:
- Checkpoint의 tuple key 구조와 multi-scale 출력
- ONNX의 array 구조와 single-scale 출력
- INT8 양자화를 위한 separate outputs 사용법
- FP32 vs INT8 비교를 위한 3개 출력 저장 방법
- Per-head 오차 분석 방법론

NPU 업체에 **DUAL_HEAD_ONNX_CONVERSION.md**를 전달하시면 됩니다!
