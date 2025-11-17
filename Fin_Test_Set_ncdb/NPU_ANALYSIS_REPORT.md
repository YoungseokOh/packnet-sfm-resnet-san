# NPU Performance Analysis Report
================================================================================

## Executive Summary

- **Total Samples**: 91
- **NPU Mean abs_rel**: 0.1024
- **FP32 Mean abs_rel**: 0.0415
- **Mean Performance Gap**: 0.0610

- **NPU Worse than FP32**: 91 samples (100.0%)
- **Similar Performance**: 5 samples (5.5%)
- **NPU Better than FP32**: 0 samples (0.0%)

## Key Findings

### 🔴 Critical Issues (Worst NPU Performance)

**Top 15 samples with highest NPU abs_rel:**
```
0000000168
0000000531
0000000533
0000000120
0000000112
0000000567
0000000080
0000000873
0000000106
0000000322
0000000450
0000000147
0000000219
0000000637
0000000398
```

**Characteristics:**
- abs_rel ranges from 0.75 to 0.15
- Significantly worse than FP32 (delta > 0.10)
- Possible causes: GT-RGB mismatch, difficult scenes, quantization artifacts

### 🟡 Large FP32-NPU Gap

**Top 15 samples with largest performance degradation from FP32 to NPU:**
```
0000000168
0000000531
0000000533
0000000120
0000000112
0000000567
0000000873
0000000106
0000000080
0000000398
0000000450
0000000147
0000000940
0000000219
0000000929
```

**Characteristics:**
- Delta (NPU - FP32) ranges from 0.71 to 0.10
- FP32 performs well (abs_rel < 0.07) but NPU struggles
- Priority for quantization optimization

### 🟢 Best NPU Performance

**Top 15 samples with lowest NPU abs_rel:**
```
0000001308
0000000172
0000000279
0000002405
0000001230
0000000246
0000002462
0000002417
0000000137
0000002348
0000002234
0000000547
0000002293
0000002304
0000000676
```

**Characteristics:**
- abs_rel ranges from 0.04 to 0.05
- Good depth estimation quality
- Reference samples for successful NPU inference

### 🔵 Similar to FP32 (Small Gap)

**Top 15 samples where NPU performance is closest to FP32:**
```
0000002507
0000000715
0000001792
0000002129
0000000779
0000001423
0000000796
0000002417
0000000547
0000000094
0000000854
0000002068
0000000067
0000001243
0000002509
```

**Characteristics:**
- Delta < 0.013 (very small performance gap)
- Both FP32 and NPU perform reasonably well
- Good quantization preservation

## Recommendations

1. **Visual Inspection Priority**:
   - Start with worst 15 samples to identify GT-RGB mismatch
   - Check if depth range or scene complexity causes issues

2. **Dataset Refinement**:
   - Remove samples with GT-RGB mismatch from test set
   - Re-evaluate NPU after dataset cleanup

3. **Quantization Analysis**:
   - Focus on samples with large FP32-NPU gap
   - Analyze integer/fractional component differences

4. **Visualization Locations**:
   - FP32 only: `Fin_Test_Set_ncdb/viz/fp32/index.html`
   - FP32 vs NPU: `Fin_Test_Set_ncdb/viz_npu/npu/index.html`
