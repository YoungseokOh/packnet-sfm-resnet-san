# Phase 1 Grid Search Automation

## Quick Start: Run Remaining Configs After w30_70

Once w30_70 completes, run the sequential script:

```bash
cd /workspace/packnet-sfm
bash run_grid_search_remaining.sh
```

This will:
1. Wait for w30_70 to complete (if still running)
2. Auto-run w40_60, w50_50, w60_40, w70_30 in sequence
3. Each trains for 5 epochs (~45min-2hours each depending on validation frequency)
4. Total time for all 4: ~6-10 hours

---

## Monitor Training Progress

### Real-time monitoring:
```bash
# Watch w30_70
tail -f /workspace/packnet-sfm/outputs/fixed_multi_domain_w30_70/training.log

# Check GPU usage
watch -n 2 nvidia-smi

# Check all running processes
ps aux | grep train.py
```

### Extract results after each config completes:
```bash
python /workspace/packnet-sfm/scripts/extract_grid_search_results.py
```

---

## Expected Timeline

| Config | w_struct | w_scale | Est. Runtime | Status |
|--------|----------|---------|--------------|--------|
| w30_70 | 0.3      | 0.7     | 40-50 min    | ✅ IN PROGRESS |
| w40_60 | 0.4      | 0.6     | 40-50 min    | ⏳ Next |
| w50_50 | 0.5      | 0.5     | 40-50 min    | ⏳ After w40 |
| w60_40 | 0.6      | 0.4     | 40-50 min    | ⏳ After w50 |
| w70_30 | 0.7      | 0.3     | 40-50 min    | ⏳ After w60 |

**Total sequential time:** ~3.5-4.5 hours (if script runs continuously)
**Start time:** 2025-10-23 06:13 UTC
**Expected completion:** 2025-10-23 10:00-11:00 UTC

---

## Key Observations from w30_70 (Patent Baseline)

**Epoch 0 Progress:**
- Training time: 8m 32s
- Final loss: 0.1157
- abs_rel trend: 0.1113 → 0.1041 → 0.0871 → 0.0945 → 0.0750
- Evaluation at batches: 345, 690, 1035, 1380, 1725 (every 345 batches, 20% of epoch)

**Performance characteristics:**
- w_structure=0.3 (lower) = more emphasis on absolute depth
- w_scale=0.7 (higher) = more emphasis on scale/magnitude consistency
- Good for abs_rel metric (which measures depth accuracy)

**Expected final abs_rel (after 5 epochs):** ~0.065-0.070 (preliminary)
*(Compare vs baseline 0.0295 - still needs improvement from initialization)*

---

## Phase Progression

After grid search completion:

### 1️⃣ Phase 1 Analysis (TODAY)
- Select best weight from grid search
- Verify improvement over SSI-Silog baseline (0.0295)
- Target: Find w_struct/w_scale that gives 0.022-0.024

### 2️⃣ Phase 1 Full Training (TOMORROW)
- Use best weight from grid search
- Run 30 epochs (instead of 5) with max configuration
- Expected abs_rel: **0.022-0.024** (20-26% improvement)

### 3️⃣ Phase 2 Distance-Aware Weighting (LATER)
- Implement adaptive weights based on depth distance
- Focus on near-field improvement (D < 1m)
- Target: abs_rel 0.015-0.018 (46-50% improvement in near-field)

---

## File Locations

**Configs:**
- `/workspace/packnet-sfm/configs/train_fixed_multi_domain_*.yaml` (5 files)

**Training logs:**
- `/workspace/packnet-sfm/outputs/fixed_multi_domain_*/training.log`

**Checkpoints:**
- `/workspace/packnet-sfm/checkpoints/fixed_multi_domain_*/`

**TensorBoard logs:**
- `/workspace/packnet-sfm/outputs/fixed_multi_domain_*/tensorboard_logs_*/`

**Result extraction script:**
- `/workspace/packnet-sfm/scripts/extract_grid_search_results.py`

**Grid search runner:**
- `/workspace/packnet-sfm/run_grid_search_remaining.sh`

---

## Troubleshooting

**If w30_70 gets stuck:**
```bash
pkill -f "train_fixed_multi_domain_w30_70"
# Then re-run from script or manually
python scripts/train.py configs/train_fixed_multi_domain_w30_70.yaml
```

**If GPU runs out of memory:**
- Reduce `batch_size` in config (currently 2)
- Or reduce `eval_subset_size` (currently 25)

**If training is too slow:**
- Check GPU utilization with `nvidia-smi`
- Ensure no other GPU processes running
- Check disk I/O: `iotop` or `iostat`

---

## Next Action

1. **Wait for w30_70 to complete** (~40 min from 06:13 UTC = ~07:00 UTC)
2. **Run grid search script:**
   ```bash
   bash run_grid_search_remaining.sh
   ```
3. **Monitor progress:**
   ```bash
   tail -f outputs/fixed_multi_domain_*/training.log
   ```
4. **Extract results when done:**
   ```bash
   python scripts/extract_grid_search_results.py
   ```

