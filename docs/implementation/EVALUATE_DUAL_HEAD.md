# Dual-Head Evaluation Script

## Overview

Simple evaluation script for Dual-Head depth predictions. Supports:
- ✅ Pre-computed depth files (FP32 ONNX or INT8 NPU outputs)
- ✅ FP32 vs INT8 comparison
- ✅ Dual-head outputs (integer + fractional) or composed depth
- ✅ Standard KITTI metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)

## Quick Start

### 1. Evaluate FP32 ONNX Inference

```bash
python scripts/evaluate_dual_head_simple.py \
    --precomputed_dir outputs/fp32_inference \
    --dataset_root /path/to/kitti \
    --test_file data_splits/kitti_eigen_test.txt \
    --dual_head_max_depth 15.0
```

### 2. Evaluate INT8 NPU Inference

```bash
python scripts/evaluate_dual_head_simple.py \
    --precomputed_dir outputs/int8_npu_inference \
    --dataset_root /path/to/kitti \
    --test_file data_splits/kitti_eigen_test.txt \
    --dual_head_max_depth 15.0
```

### 3. Compare FP32 vs INT8

```bash
python scripts/evaluate_dual_head_simple.py \
    --precomputed_dir outputs/fp32_inference \
    --compare_dir outputs/int8_npu_inference \
    --dataset_root /path/to/kitti \
    --test_file data_splits/kitti_eigen_test.txt \
    --dual_head_max_depth 15.0 \
    --output_json comparison_results.json
```

## Input File Formats

The script supports multiple input formats:

### Format 1: Composed Depth Only (`.npy`)

```
outputs/fp32_inference/
├── 0000000147.npy  # Shape: [384, 640] or [1, 1, 384, 640]
├── 0000000655.npy
└── ...
```

Each `.npy` file contains the final composed depth map.

### Format 2: Dual-Head Outputs (`.npz`)

```
outputs/fp32_inference/
├── 0000000147.npz  # Contains: integer_sigmoid, fractional_sigmoid
├── 0000000655.npz
└── ...
```

Each `.npz` file contains:
- `integer_sigmoid`: [384, 640] - Coarse depth component
- `fractional_sigmoid`: [384, 640] - Fine depth component

The script will automatically compose: `depth = integer * 15.0 + fractional`

### Format 3: Pre-composed with Components (`.npz`)

```
outputs/fp32_inference/
├── 0000000147.npz  # Contains: depth, integer_sigmoid, fractional_sigmoid
├── 0000000655.npz
└── ...
```

Each `.npz` file contains:
- `depth` or `depth_composed`: [384, 640] - Pre-computed depth
- `integer_sigmoid` (optional): For analysis
- `fractional_sigmoid` (optional): For analysis

## Command-Line Options

### Required Arguments

- `--dataset_root`: Path to KITTI dataset root
  - Must contain `newest_depth_maps/` or `depth_selection/val_selection_cropped/groundtruth_depth/`
  
- `--test_file`: Path to test file list
  - Example: `data_splits/kitti_eigen_test.txt`

### Input Options

- `--precomputed_dir`: Directory with pre-computed depth files (`.npy` or `.npz`)
  - FP32 ONNX outputs or INT8 NPU outputs
  
- `--compare_dir`: Directory with comparison depth files
  - For FP32 vs INT8 comparison

- `--checkpoint`: Path to PyTorch checkpoint (not yet implemented)

### Depth Range Options

- `--min_depth`: Minimum valid depth for evaluation (default: 1e-3)
- `--max_depth`: Maximum valid depth in dataset (default: 80.0)
- `--eval_max_depth`: Maximum depth for evaluation metrics (default: 80.0)
- `--dual_head_max_depth`: Max depth for dual-head composition (default: 15.0)
  - ⚠️ **Important**: Must match the value used during training/inference

### Output Options

- `--output_json`: Path to save results as JSON
  - Includes metrics and comparison results

## Example Workflows

### Workflow 1: NPU INT8 Validation

```bash
# Step 1: Run FP32 ONNX inference (reference)
python scripts/infer_dual_head_onnx.py \
    --onnx onnx/dual_head_..._separate_zero_static.onnx \
    --dataset_root /path/to/kitti \
    --test_file data_splits/kitti_eigen_test.txt \
    --output_dir outputs/fp32_reference \
    --max_depth 15.0

# Step 2: Run INT8 NPU inference (your NPU toolkit)
your_npu_runner \
    --model model_int8.bin \
    --dataset /path/to/kitti \
    --test_file data_splits/kitti_eigen_test.txt \
    --output_dir outputs/int8_npu

# Step 3: Compare FP32 vs INT8
python scripts/evaluate_dual_head_simple.py \
    --precomputed_dir outputs/fp32_reference \
    --compare_dir outputs/int8_npu \
    --dataset_root /path/to/kitti \
    --test_file data_splits/kitti_eigen_test.txt \
    --dual_head_max_depth 15.0 \
    --output_json fp32_vs_int8_results.json
```

Expected output:
```
📊 Dual-Head Depth Evaluation
================================================================================

📂 Loading test file: data_splits/kitti_eigen_test.txt
   Found 697 test samples

⚙️  Evaluation setup:
   Dataset root:    /path/to/kitti
   Min depth:       0.001m
   Max depth:       80.0m
   Eval max depth:  80.0m
   Dual-head max:   15.0m
   Precomputed dir: outputs/fp32_reference
   Compare dir:     outputs/int8_npu

🔍 Evaluating 697 samples...
100%|████████████████████████████████████████| 697/697 [00:05<00:00, 134.21it/s]

✅ Evaluated 697 valid samples

================================================================================
📊 Evaluation Results
================================================================================

Model: outputs/fp32_reference
Samples: 697

Metrics:
  abs_rel:  0.0420
  sq_rel:   0.236
  rmse:     2.489
  rmse_log: 0.1050
  a1:       96.80%
  a2:       99.40%
  a3:       99.80%

--------------------------------------------------------------------------------

Comparison Model: outputs/int8_npu
Samples: 697

Metrics:
  abs_rel:  0.0425 (Δ +0.0005)
  sq_rel:   0.238 (Δ +0.002)
  rmse:     2.495 (Δ +0.006)
  rmse_log: 0.1052 (Δ +0.0002)
  a1:       96.75% (Δ -0.05%)
  a2:       99.38% (Δ -0.02%)
  a3:       99.79% (Δ -0.01%)

💾 Results saved to: fp32_vs_int8_results.json

================================================================================
```

### Workflow 2: Quick Accuracy Check

```bash
# Evaluate just the FP32 model
python scripts/evaluate_dual_head_simple.py \
    --precomputed_dir outputs/fp32_inference \
    --dataset_root /path/to/kitti \
    --test_file data_splits/kitti_eigen_test.txt \
    --dual_head_max_depth 15.0
```

## Output JSON Format

When using `--output_json`, the script saves results in the following format:

```json
{
  "model": "outputs/fp32_reference",
  "num_samples": 697,
  "metrics": {
    "abs_rel": 0.0420,
    "sq_rel": 0.236,
    "rmse": 2.489,
    "rmse_log": 0.1050,
    "a1": 0.9680,
    "a2": 0.9940,
    "a3": 0.9980
  },
  "comparison": {
    "model": "outputs/int8_npu",
    "num_samples": 697,
    "metrics": {
      "abs_rel": 0.0425,
      "sq_rel": 0.238,
      "rmse": 2.495,
      "rmse_log": 0.1052,
      "a1": 0.9675,
      "a2": 0.9938,
      "a3": 0.9979
    }
  }
}
```

## NPU Inference Output Format

When saving NPU inference results, use one of these formats:

### Recommended: Dual-Head NPZ Format

```python
# After NPU inference
integer_sigmoid, fractional_sigmoid = npu_model.infer(rgb)

# Save for evaluation
np.savez(f'outputs/int8_npu/{filename}.npz',
         integer_sigmoid=integer_sigmoid,    # [1, 1, 384, 640]
         fractional_sigmoid=fractional_sigmoid)  # [1, 1, 384, 640]

# The evaluation script will compose: depth = integer * 15.0 + fractional
```

### Alternative: Pre-composed NPY Format

```python
# After NPU inference
integer_sigmoid, fractional_sigmoid = npu_model.infer(rgb)

# Compose depth
depth = integer_sigmoid * 15.0 + fractional_sigmoid

# Save for evaluation
np.save(f'outputs/int8_npu/{filename}.npy', depth)  # [384, 640]
```

## Metrics Explanation

- **abs_rel**: Absolute relative error - Lower is better (typical: 0.04 - 0.15)
- **sq_rel**: Squared relative error - Lower is better (typical: 0.2 - 1.0)
- **rmse**: Root mean squared error (meters) - Lower is better (typical: 2 - 5)
- **rmse_log**: RMSE in log space - Lower is better (typical: 0.1 - 0.3)
- **a1**: % pixels with error < 1.25 - Higher is better (typical: 85% - 98%)
- **a2**: % pixels with error < 1.25² - Higher is better (typical: 95% - 99%)
- **a3**: % pixels with error < 1.25³ - Higher is better (typical: 98% - 100%)

## Expected INT8 Degradation

Based on typical INT8 quantization:

- **abs_rel**: +0.0005 to +0.002 degradation
- **rmse**: +0.01m to +0.05m degradation
- **a1**: -0.1% to -0.5% degradation

If degradation is larger, consider:
1. Better calibration data
2. Per-channel quantization
3. Mixed precision (keep sensitive layers in FP16)

## Troubleshooting

### Error: "GT depth not found"

- Ensure `--dataset_root` points to KITTI dataset root
- Check that `newest_depth_maps/` or `depth_selection/val_selection_cropped/groundtruth_depth/` exists
- Verify test file paths match GT depth filenames

### Error: "Predicted depth not found"

- Ensure `--precomputed_dir` contains `.npy` or `.npz` files
- Filenames must match test file entries (without `.png` extension)
- Example: If test file has `0000000147.png`, need `0000000147.npy` or `0000000147.npz`

### Error: "No valid samples found"

- Check that test file format is correct
- Verify GT depth and predicted depth paths
- Try with a single sample first for debugging

## Related Scripts

- `scripts/infer_dual_head_onnx.py` - Run FP32 ONNX inference
- `scripts/convert_dual_head_to_onnx.py` - Convert checkpoint to ONNX
- `scripts/validate_dual_head_onnx.py` - Validate ONNX conversion accuracy

## Notes

- ⚠️ PyTorch checkpoint evaluation is not yet implemented (only pre-computed depths)
- ✅ Supports multiple depth file formats (`.npy`, `.npz`)
- ✅ Automatically handles dual-head composition if needed
- ✅ Compatible with NPU inference outputs
