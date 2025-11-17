# Dual-Head Output Structure Documentation

## 📋 Overview

ST2 (Structured Target 2) Dual-Head architecture는 depth estimation을 두 개의 head로 분리하여 정밀도를 향상시킵니다:
- **Integer Head**: 전체 depth 범위 [0, max_depth]m
- **Fractional Head**: 세밀한 fractional 값 [0, 1]m

---

## 🏗️ Architecture Components

### 1. DualHeadDepthDecoder

**Location**: `packnet_sfm/networks/depth/DualHeadDepthDecoder.py`

**Purpose**: ResNet encoder features를 integer와 fractional depth로 디코딩

**Output Format**:
```python
{
    ('integer', 0): Tensor[B, 1, H, W],      # Scale 0 (full resolution)
    ('fractional', 0): Tensor[B, 1, H, W],
    ('integer', 1): Tensor[B, 1, H/2, W/2],  # Scale 1
    ('fractional', 1): Tensor[B, 1, H/2, W/2],
    ('integer', 2): Tensor[B, 1, H/4, W/4],  # Scale 2
    ('fractional', 2): Tensor[B, 1, H/4, W/4],
    ('integer', 3): Tensor[B, 1, H/8, W/8],  # Scale 3
    ('fractional', 3): Tensor[B, 1, H/8, W/8],
}
```

**Key Properties**:
- 각 head는 sigmoid activation을 통과하여 [0, 1] 범위
- Multi-scale outputs (4 scales: 0, 1, 2, 3)
- Tuple keys를 사용: `(head_type, scale_index)`

---

## 📊 Output Value Ranges

### Integer Head Output
```
Raw Output (after sigmoid): [0, 1]
Interpretation: 
  - 0.0 → 0m
  - 1.0 → max_depth (e.g., 15.0m)
  
Quantization Interval: max_depth / 255 = 0.0588m (58.82mm for max_depth=15m)
```

### Fractional Head Output
```
Raw Output (after sigmoid): [0, 1]
Interpretation:
  - 0.0 → 0m (fractional part)
  - 1.0 → 1.0m (fractional part)
  
Quantization Interval: 1.0 / 255 = 0.0039m (3.92mm)
```

---

## 🔢 Depth Reconstruction Formula

### Helper Function: `dual_head_to_depth()`

**Location**: `packnet_sfm/networks/layers/resnet/layers.py`

**Formula**:
```python
depth = integer_sigmoid * max_depth + fractional_sigmoid
```

**Example** (max_depth = 15.0m):
```python
integer_sigmoid = 0.5    # → 7.5m
fractional_sigmoid = 0.3  # → 0.3m
depth = 0.5 * 15.0 + 0.3 = 7.8m
```

**Code**:
```python
from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth

integer_output = model_output[('integer', 0)]      # [B, 1, H, W]
fractional_output = model_output[('fractional', 0)] # [B, 1, H, W]
max_depth = 15.0

depth_reconstructed = dual_head_to_depth(
    integer_output, 
    fractional_output, 
    max_depth
)  # [B, 1, H, W] in meters
```

---

## 🔄 Data Flow

### Training Forward Pass

```
Input RGB [B, 3, H, W]
    ↓
ResNet Encoder (ResNetSAN01)
    ↓
DualHeadDepthDecoder
    ↓
{
  ('integer', 0): [B, 1, H, W],
  ('fractional', 0): [B, 1, H, W],
  ...
}
    ↓
SemiSupCompletionModel.forward()
    ↓
DualHeadDepthLoss
    ↓
Loss (3 components):
  - Integer loss
  - Fractional loss (weighted 10x)
  - Consistency loss
```

### Evaluation Forward Pass

```
Input RGB [B, 3, H, W]
    ↓
ResNet Encoder (ResNetSAN01)
    ↓
DualHeadDepthDecoder
    ↓
{('integer', 0): [B,1,H,W], ('fractional', 0): [B,1,H,W]}
    ↓
model_wrapper.evaluate_depth()
    ↓
dual_head_to_depth() → depth [B, 1, H, W]
    ↓
compute_depth_metrics() → abs_rel, rmse, a1, etc.
```

---

## 💾 Output Dictionary Structure

### Full Output Example

```python
# After forward pass
outputs = model.depth_net(batch)

# Type: dict
# Keys: tuple (head_type: str, scale: int)
# Values: torch.Tensor [B, 1, H_scale, W_scale]

print(outputs.keys())
# Output:
# dict_keys([
#   ('integer', 0), ('fractional', 0),
#   ('integer', 1), ('fractional', 1),
#   ('integer', 2), ('fractional', 2),
#   ('integer', 3), ('fractional', 3)
# ])

# Accessing outputs
integer_full_res = outputs[('integer', 0)]    # [B, 1, 384, 640]
fractional_full_res = outputs[('fractional', 0)]  # [B, 1, 384, 640]

# Check value ranges
print(f"Integer range: [{integer_full_res.min():.4f}, {integer_full_res.max():.4f}]")
# Output: Integer range: [0.0234, 0.9876]

print(f"Fractional range: [{fractional_full_res.min():.4f}, {fractional_full_res.max():.4f}]")
# Output: Fractional range: [0.0123, 0.9543]
```

---

## 🔍 Key Differences from Single-Head

| Aspect | Single-Head | Dual-Head |
|--------|------------|-----------|
| **Output Keys** | `'inv_depths'` (string) | `('integer', scale)`, `('fractional', scale)` (tuple) |
| **Output Type** | List of tensors | Dict with tuple keys |
| **Value Range** | Sigmoid [0, 1] → inverse depth | Two sigmoids [0, 1] → actual depth |
| **Reconstruction** | `sigmoid_to_inv_depth()` → `inv2depth()` | `dual_head_to_depth()` directly |
| **Loss Function** | `SupervisedLoss` (e.g., SSI-Silog) | `DualHeadDepthLoss` (3-component) |

---

## 📝 Code Examples

### Example 1: Checking Model Output Type

```python
# Detect if model is Dual-Head
if 'inv_depths' in outputs:
    print("Single-Head model")
    sigmoid_outputs = outputs['inv_depths']
else:
    print("Dual-Head model")
    has_integer = ('integer', 0) in outputs
    has_fractional = ('fractional', 0) in outputs
    print(f"Has integer: {has_integer}, Has fractional: {has_fractional}")
```

### Example 2: Extracting and Visualizing Outputs

```python
import numpy as np
import matplotlib.pyplot as plt

# Extract outputs
integer_sig = outputs[('integer', 0)][0, 0].cpu().numpy()  # [H, W]
fractional_sig = outputs[('fractional', 0)][0, 0].cpu().numpy()  # [H, W]

# Reconstruct depth
depth = integer_sig * 15.0 + fractional_sig

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(integer_sig, cmap='viridis')
axes[0].set_title('Integer Head (sigmoid [0,1])')
axes[1].imshow(fractional_sig, cmap='viridis')
axes[1].set_title('Fractional Head (sigmoid [0,1])')
axes[2].imshow(depth, cmap='magma', vmin=0, vmax=15)
axes[2].set_title('Reconstructed Depth (m)')
plt.show()
```

### Example 3: Saving Outputs to NPY

```python
import torch
import numpy as np
from pathlib import Path

def save_dual_head_outputs(outputs, save_dir, sample_idx, max_depth=15.0):
    """Save Dual-Head outputs to NPY files"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract scale 0 outputs
    integer_sig = outputs[('integer', 0)][0, 0].cpu().numpy()
    fractional_sig = outputs[('fractional', 0)][0, 0].cpu().numpy()
    
    # Reconstruct depth
    from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth
    depth = dual_head_to_depth(
        outputs[('integer', 0)],
        outputs[('fractional', 0)],
        max_depth
    )[0, 0].cpu().numpy()
    
    # Save
    np.save(save_dir / f"sample_{sample_idx:04d}_integer.npy", integer_sig)
    np.save(save_dir / f"sample_{sample_idx:04d}_fractional.npy", fractional_sig)
    np.save(save_dir / f"sample_{sample_idx:04d}_depth.npy", depth)
    
    print(f"✅ Saved outputs for sample {sample_idx}")
    print(f"   Integer range: [{integer_sig.min():.4f}, {integer_sig.max():.4f}]")
    print(f"   Fractional range: [{fractional_sig.min():.4f}, {fractional_sig.max():.4f}]")
    print(f"   Depth range: [{depth.min():.4f}, {depth.max():.4f}]m")

# Usage
save_dual_head_outputs(outputs, "outputs/dual_head_npy", sample_idx=0)
```

---

## 🧪 Validation Results

### Epoch 28 Results (NCDB 640x384)

**Checkpoint**: `checkpoints/resnetsan01_dual_head_ncdb_640x384/.../epoch=28_....ckpt`

**Validation Metrics**:
- abs_rel: **0.04257** (4.26% average relative error)
- rmse: **0.4646m** (46cm average error)
- a1 (δ < 1.25): **96.79%** (accuracy within 25%)
- a2 (δ < 1.25²): **98.93%**
- a3 (δ < 1.25³): **99.59%**

**Test Metrics** (eval.py):
- abs_rel: **0.042**
- rmse: **0.471m**
- a1: **96.8%**

✅ **Validation과 Test 결과가 일치** → Dual-Head 구현 검증 완료

---

## 🔧 Implementation Files

### Core Implementation
1. **DualHeadDepthDecoder**: `packnet_sfm/networks/depth/DualHeadDepthDecoder.py`
2. **DualHeadDepthLoss**: `packnet_sfm/losses/dual_head_depth_loss.py`
3. **Helper Functions**: `packnet_sfm/networks/layers/resnet/layers.py`
   - `dual_head_to_depth()`
   - `decompose_depth()`

### Model Wrappers
1. **ResNetSAN01**: `packnet_sfm/networks/depth/ResNetSAN01.py`
   - Handles Dual-Head/Single-Head switching
2. **SemiSupCompletionModel**: `packnet_sfm/models/SemiSupCompletionModel.py`
   - Auto-detects Dual-Head and uses appropriate loss
3. **ModelWrapper**: `packnet_sfm/models/model_wrapper.py`
   - Evaluation pipeline with Dual-Head support

### Utilities
1. **model_utils.py**: `packnet_sfm/models/model_utils.py`
   - `upsample_output()` with tuple key support

---

## � Saving Dual-Head Outputs to NPY/NPZ

### Using save_dual_head_outputs.py Script

**Location**: `scripts/save_dual_head_outputs.py`

**Purpose**: Save integer head, fractional head, and composed depth separately for analysis

**Usage**:
```bash
# Save all test samples as NPZ (compressed)
python scripts/save_dual_head_outputs.py \
    --checkpoint checkpoints/.../epoch=28_....ckpt \
    --output_dir outputs/dual_head_outputs_npy \
    --split test \
    --save_format npz

# Save first 10 validation samples as separate NPY files
python scripts/save_dual_head_outputs.py \
    --checkpoint checkpoints/.../epoch=28_....ckpt \
    --output_dir outputs/dual_head_val_npy \
    --split val \
    --num_samples 10 \
    --save_format npy
```

**Output Structure (NPZ format)**:
```
outputs/dual_head_outputs_npy/
├── 0000000001_dual_head.npz
│   ├── integer_sigmoid     # [H, W], range [0, 1]
│   ├── fractional_sigmoid  # [H, W], range [0, 1]
│   ├── depth_composed      # [H, W], in meters
│   └── intrinsics          # Camera intrinsics
├── 0000000002_dual_head.npz
...
```

**Output Structure (NPY format)**:
```
outputs/dual_head_outputs_npy/
├── 0000000001/
│   ├── integer_sigmoid.npy
│   ├── fractional_sigmoid.npy
│   ├── depth_composed.npy
│   └── intrinsics.npy
├── 0000000002/
...
```

**Loading Saved Outputs**:
```python
import numpy as np

# Load NPZ file
data = np.load('outputs/dual_head_outputs_npy/0000000001_dual_head.npz')

integer_sig = data['integer_sigmoid']      # [H, W], sigmoid [0, 1]
fractional_sig = data['fractional_sigmoid']  # [H, W], sigmoid [0, 1]
depth = data['depth_composed']             # [H, W], meters

# Verify reconstruction
max_depth = 15.0
depth_manual = integer_sig * max_depth + fractional_sig
print(f"Reconstruction error: {np.abs(depth - depth_manual).max():.6f}m")
# Output: Reconstruction error: 0.000000m (should be ~0)
```

### Current Saved Outputs

**Evaluation Outputs** (from eval.py):
- Location: `outputs/resnetsan01_dual_head_ncdb_640x384/depth/ncdb-cls-640x384-combined_test/`
- Format: NPZ files with composed depth only
- Files: `{sample_id}_depth.npz`
- Contents:
  - `depth`: [H, W] composed depth in meters
  - `intrinsics`: [18] camera intrinsics

**Note**: eval.py saves only the final composed depth, not individual integer/fractional heads.
Use `save_dual_head_outputs.py` to save all components separately.

---

## �📚 Related Documentation

- [ST2 Implementation Guide](./ST2_IMPLEMENTATION.md)
- [Training Configuration](../../configs/train_resnet_san_ncdb_dual_head_640x384.yaml)
- [Test Results](../tests/DUAL_HEAD_TESTS.md)

---

## ⚠️ Important Notes

1. **Tuple Keys**: Dual-Head uses tuple keys `(head_type, scale)`, not string keys
2. **Value Ranges**: Both heads output sigmoid [0, 1], not actual depth values
3. **Reconstruction Required**: Must use `dual_head_to_depth()` to get actual depth
4. **Loss Weighting**: Fractional loss has 10x weight for higher precision
5. **Backward Compatibility**: Single-Head models still work with same codebase

---

## 🐛 Troubleshooting

### Issue: KeyError 'inv_depths'
**Cause**: Code assumes Single-Head output format  
**Solution**: Check for Dual-Head format first:
```python
if 'inv_depths' in outputs:
    # Single-Head path
else:
    # Dual-Head path
```

### Issue: Incorrect depth values
**Cause**: Treating sigmoid outputs as depth  
**Solution**: Always use `dual_head_to_depth()`:
```python
depth = dual_head_to_depth(integer_sig, fractional_sig, max_depth)
```

### Issue: Loss computation error
**Cause**: Passing reconstructed depth to DualHeadDepthLoss  
**Solution**: Pass original dict with tuple keys:
```python
loss = dual_head_loss(outputs, gt_depth)  # outputs is dict with tuple keys
```

---

## 📞 Contact

For questions or issues, refer to:
- Implementation Lead: ST2 Dual-Head Architecture
- Repository: `packnet-sfm-resnet-san`
- Branch: `feat/ST2-implementation`

---

**Last Updated**: November 11, 2025  
**Version**: 1.0  
**Status**: ✅ Validated and Production-Ready
