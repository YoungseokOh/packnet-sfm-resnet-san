# Metrics tables (matplotlib)

This folder contains a small utility to render evaluation results into **PNG table images** (good for reports / slides).

## Script

- `scripts/visualization/render_metrics_tables.py`

## What it generates

By default it produces:
- `outputs/metrics_tables/npu_metrics.png` (key/value table)
- `outputs/metrics_tables/gpu_summary.png` (multi-column table)

## Example (NPU + GPU)

```bash
cd /workspace/packnet-sfm
python scripts/visualization/render_metrics_tables.py \
  --out_dir outputs/metrics_tables \
  --npu_title "NPU Results (depth_synthetic)" \
  --npu_metrics '{"abs_rel":0.0932,"sq_rel":0.0764,"rmse":0.4346,"rmse_log":0.1697,"a1":92.66,"a2":96.90,"a3":98.31}' \
  --gpu_title "GPU Results" \
  --gpu_summary '[{"depth_type":"depth_synthetic","folder":"newest_depth_maps","abs_rel":0.070,"a1":93.6}]'
```

Notes:
- `a1/a2/a3` are expected in **percent** (e.g. `92.66` not `0.9266`).
- `gpu_summary` can be multiple rows (list of dicts).
