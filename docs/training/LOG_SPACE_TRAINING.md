# Log Space Training Feature

## Overview

This feature enables **optional log space training** for depth estimation. By default, the model trains in **linear inverse depth space**, but you can now switch to **log inverse depth space** via a simple config parameter.

## Why Log Space?

**Linear Inverse Depth (Default):**
```
inv_depth = min_inv + (max_inv - min_inv) × sigmoid
```
- sigmoid=0 → 80m (FAR)
- sigmoid=0.5 → 0.1m (weighted toward near range)
- sigmoid=1 → 0.05m (NEAR)
- **Distribution**: Heavily focused on near range

**Log Inverse Depth:**
```
log_inv = log(min_inv) + (log(max_inv) - log(min_inv)) × sigmoid
inv_depth = exp(log_inv)
```
- sigmoid=0 → 80m (FAR)
- sigmoid=0.5 → 2.0m (geometric mean - more balanced)
- sigmoid=1 → 0.05m (NEAR)
- **Distribution**: More uniform across depth ranges

## Configuration

Add to your config YAML under `model.params`:

```yaml
model:
    name: 'SemiSupCompletionModel'
    params:
        min_depth: 0.05
        max_depth: 80.0
        use_log_space: false  # Set to true for log space training
```

### Parameters:

- `use_log_space: false` (default) - Linear inverse depth training (backward compatible)
- `use_log_space: true` - Log inverse depth training (uniform distribution)

## Implementation Details

### Training Flow:
```
Model → sigmoid [0,1] 
→ sigmoid_to_inv_depth(use_log_space) 
→ bounded_inv_depth 
→ SSI + Silog Loss
```

### Evaluation Flow:
```
Model → sigmoid [0,1] 
→ sigmoid_to_inv_depth(use_log_space)  # Same as training!
→ bounded_inv_depth 
→ depth 
→ Metrics
```

**Key**: Training and evaluation use the **SAME** transformation method automatically.

## Expected Metrics Behavior

### With `use_log_space: false` (LINEAR training - default):

```
MAIN METRICS (using LINEAR):
  abs_rel=0.040 ✅

LINEAR TRANSFORMATION (same as MAIN):
  depth_lin: abs_rel=0.040 ✅ (identical to MAIN)

LOG TRANSFORMATION (different distribution):
  depth_log: abs_rel=40.1 ❌ (expected - different interpolation)
```

### With `use_log_space: true` (LOG training):

```
MAIN METRICS (using LOG):
  abs_rel=0.040 ✅

LINEAR TRANSFORMATION (different distribution):
  depth_lin: abs_rel=40.1 ❌ (expected - different interpolation)

LOG TRANSFORMATION (same as MAIN):
  depth_log: abs_rel=0.040 ✅ (identical to MAIN)
```

## When to Use Log Space?

**Use Linear (default)** when:
- Focusing on near-range accuracy (0-10m)
- Autonomous driving scenarios
- Urban environments

**Use Log** when:
- Need uniform depth distribution
- Long-range accuracy matters (10-80m)
- Outdoor/natural environments
- Analyzing different depth distributions

## Migration from Previous Code

**No changes needed** - the feature is backward compatible:
- Default behavior: `use_log_space=False` (linear space)
- Existing configs work without modification
- Previous checkpoints are compatible

## Files Modified

1. **`packnet_sfm/utils/post_process_depth.py`**
   - `sigmoid_to_inv_depth()` now accepts `use_log_space` parameter
   - Supports both linear (default) and log interpolation

2. **`packnet_sfm/models/SemiSupCompletionModel.py`**
   - Added `use_log_space` parameter to `__init__`
   - Passes to all `sigmoid_to_inv_depth()` calls
   - Prints transformation mode on startup

3. **`packnet_sfm/models/model_wrapper.py`**
   - Detects `use_log_space` from model
   - Passes to evaluation
   - Computes and displays LOG metrics for comparison
   - Reads `use_log_space` from config.params

## Testing

### Test Linear Mode (default):
```yaml
model:
  params:
    use_log_space: false
```

**Expected output:**
```
🔧 SemiSupCompletionModel: Using LINEAR SPACE interpolation

MAIN METRICS:
  abs_rel=0.040

LINEAR TRANSFORMATION:
  depth_lin: abs_rel=0.040 ✅

LOG TRANSFORMATION:
  depth_log: abs_rel=40.1 ❌ (different distribution)
```

### Test Log Mode:
```yaml
model:
  params:
    use_log_space: true
```

**Expected output:**
```
🔧 SemiSupCompletionModel: Using LOG SPACE interpolation

MAIN METRICS:
  abs_rel=0.040

LINEAR TRANSFORMATION:
  depth_lin: abs_rel=40.1 ❌ (different distribution)

LOG TRANSFORMATION:
  depth_log: abs_rel=0.040 ✅
```

## Technical Notes

### Training-Evaluation Consistency:
- **CRITICAL**: Training and evaluation must use the SAME transformation
- Previously: Training used LINEAR, metrics used LOG → 1000x error
- Now: Both automatically use the same method via `use_log_space` parameter

### Loss Calculation:
- SSI Loss: Computed in **inverse depth space** (same for both modes)
- Silog Loss: Converts to **depth space**, then computes (same for both modes)
- Only difference: How sigmoid maps to inverse depth

### Visualization:
The transformation mode is printed on model initialization:
```
🔧 SemiSupCompletionModel: Using LINEAR SPACE interpolation
```
or
```
🔧 SemiSupCompletionModel: Using LOG SPACE interpolation
```

## References

- `SIGMOID_TO_DEPTH_FLOW.md` - Detailed sigmoid transformation analysis
- `GIT_HISTORY_LOSS_ANALYSIS.md` - Historical loss calculation methods
- `LOG_TRANSFORMATION_ANALYSIS.md` - Why LOG metrics failed previously
