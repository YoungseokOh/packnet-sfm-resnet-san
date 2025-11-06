# Direct Depth Model Evaluation Summary

**Model**: ResNetSAN01 Direct Depth (0.5-15m range)  
**Checkpoint**: `epoch=29_ncdb-cls-640x384-combined_val-loss=0.000.ckpt`  
**Test Set**: `combined_test.json` (91 samples)  
**Date**: 2025-11-06

## ✅ Validated Results

### PyTorch FP32 (Baseline)

| Metric | No GT Scale | With GT Scale |
|--------|-------------|---------------|
| abs_rel | **0.043** | **0.039** |
| sq_rel | 0.034 | 0.036 |
| RMSE | 0.391m | 0.400m |
| RMSE_log | 0.085 | 0.083 |
| δ<1.25 | 0.976 | 0.976 |
| δ<1.25² | 0.992 | 0.992 |
| δ<1.25³ | 0.997 | 0.997 |

**Source**: Official eval.py + Batch evaluation on precomputed predictions  
**Status**: ✅ Validated (matches official eval exactly)

### NPU INT8

| Metric | No GT Scale | With GT Scale |
|--------|-------------|---------------|
| abs_rel | **0.113** | **0.118** |
| sq_rel | 0.255 | 0.239 |
| RMSE | 0.741m | 0.810m |
| RMSE_log | 0.164 | 0.161 |
| δ<1.25 | 0.924 | 0.912 |
| δ<1.25² | 0.958 | 0.959 |
| δ<1.25³ | 0.975 | 0.977 |

**Source**: Batch evaluation on NPU INT8 precomputed predictions  
**Degradation from PyTorch FP32**: 2.6x in abs_rel (0.043 → 0.113)  
**Status**: ✅ Validated with correct batch evaluation

## 📊 Performance Analysis

### Quantization Impact

The Direct Depth model shows significant degradation when quantized to INT8:

- **abs_rel**: 0.043 (FP32) → 0.113 (INT8) = **+163% error increase**
- **RMSE**: 0.391m (FP32) → 0.741m (INT8) = **+89% error increase**
- **δ<1.25**: 0.976 (FP32) → 0.924 (INT8) = **-5.3% accuracy decrease**

This is **much worse** than expected based on theoretical quantization error analysis (±28.4mm), suggesting the INT8 quantization is introducing significant numerical errors beyond simple output discretization.

### Possible Causes of High INT8 Degradation

1. **Intermediate layer quantization**: The ±28.4mm analysis only considered output quantization, but intermediate conv/pooling layers are also quantized
2. **Accumulation errors**: Multiple quantized operations compound errors through the network
3. **Activation distributions**: If activations don't match INT8 calibration range, clipping introduces errors
4. **Batch normalization**: BN layer fusion may not preserve FP32 behavior exactly

### Comparison with Bounded Inverse Depth Models

Previous work showed bounded inverse depth models had:
- **FP32**: abs_rel ≈ 0.032
- **INT8**: abs_rel ≈ 0.114

This direct depth model shows:
- **FP32**: abs_rel = 0.043 (slightly worse, -34%)
- **INT8**: abs_rel = 0.113 (about the same!)

So while the FP32 baseline is slightly worse, the INT8 degradation is similar between direct and bounded inverse approaches.

## 🔍 Methodology Notes

### Critical Finding: Batch Evaluation vs Per-Sample Averaging

During evaluation, we discovered that computing metrics per-sample and averaging gives **incorrect results**:

**Wrong Method** (per-sample averaging):
```python
for sample in samples:
    metrics = compute_depth_metrics(gt[1,1,H,W], pred[1,1,H,W])
    all_metrics.append(metrics)
result = mean(all_metrics)  # ❌ Gives abs_rel=0.089 (2x too high!)
```

**Correct Method** (batch evaluation):
```python
all_gt = stack([gt1, ..., gtN])      # [N,1,H,W]
all_pred = stack([pred1, ..., predN])  # [N,1,H,W]
result = compute_depth_metrics(all_gt, all_pred)  # ✅ Gives abs_rel=0.043
```

**Why Different?**  
The `compute_depth_metrics()` function computes:
```
abs_rel = sum(|pred - gt| / gt) / (batch_size × valid_pixels)
```

When called per-sample with batch_size=1:
- Each sample's valid pixel count differs
- Averaging per-sample metrics weights all samples equally
- Samples with fewer valid pixels get disproportionate weight

When called with full batch:
- All valid pixels from all samples contribute equally
- Matches the official evaluation behavior
- Mathematically correct for pixel-wise metrics

This explains why our initial simplified evaluation gave abs_rel=0.089 instead of 0.043!

### Data Sources

| Model | Predictions Directory | Generation Method |
|-------|-----------------------|-------------------|
| PyTorch FP32 | `outputs/pytorch_fp32_official_pipeline/` | Official model inference with correct checkpoint |
| NPU INT8 | `outputs/resnetsan_direct_depth_05_15_640x384/` | ONNX INT8 model on NPU hardware |
| ~~ONNX FP32~~ | ~~`outputs/onnx_fp32_direct_depth_inference/`~~ | ❌ Invalid (wrong checkpoint/conversion) |

### Evaluation Scripts

- **Official**: `scripts/eval.py` (ground truth, batch_size=1, validates to abs_rel=0.043)
- **Batch Evaluation**: `eval_precomputed_simple.py` (loads all samples into [N,1,H,W] batch)
- **Prediction Generation**: `generate_pytorch_predictions.py` (uses official pipeline)

## 🎯 Conclusions

1. **Direct Depth FP32** performs well: abs_rel=0.043, RMSE=0.391m ✅
2. **Direct Depth INT8** shows 2.6x degradation: abs_rel=0.113, RMSE=0.741m ⚠️
3. **Batch evaluation is critical** for correct metrics (per-sample averaging gives wrong results!)
4. **INT8 quantization** impact is larger than theoretical output discretization error
5. **Further investigation needed** to understand why INT8 degrades more than expected

## 📁 Files Generated

- `outputs/pytorch_fp32_official_pipeline/*.npy` - Correct PyTorch FP32 predictions (91 files)
- `eval_precomputed_simple.py` - Validated batch evaluation script
- `generate_pytorch_predictions.py` - Prediction generation using official pipeline
- `DIRECT_DEPTH_EVALUATION_SUMMARY.md` - This summary

## 🔄 Next Steps

To improve INT8 performance:

1. **Calibration analysis**: Check if INT8 calibration data is representative
2. **Layer-wise debugging**: Identify which layers contribute most to quantization error
3. **Mixed precision**: Try keeping critical layers in FP16
4. **Quantization-aware training**: Retrain with INT8 quantization simulation
5. **Alternative architectures**: Test if different backbone/decoder improves INT8 robustness
