# Scripts - Evaluation (평가/검증)

모델 평가 및 출력 검증 관련 스크립트들입니다.

## 📖 주요 스크립트

### `generate_pytorch_predictions.py`
**공식 파이프라인으로 예측 생성**

PyTorch 모델을 사용해 공식 파이프라인에 따라 깊이 예측을 생성합니다. (배치 처리)

```bash
python scripts/evaluation/generate_pytorch_predictions.py \
  --checkpoint path/to/model.ckpt \
  --data_path /path/to/dataset/ \
  --output_dir outputs/predictions/
```

---

### `eval_precomputed_simple.py`
**미리계산된 깊이 평가**

이미 생성된 깊이 맵(`.npy`, `.png` 등)에 대해 평가 지표를 계산합니다.

```bash
python scripts/evaluation/eval_precomputed_simple.py \
  --depth_dir outputs/predictions/ \
  --gt_dir /path/to/groundtruth/
```

---

### `evaluate_npu_direct_depth_official.py`
**NPU Direct Depth 평가**

NPU 출력 (Direct Depth 방식)을 공식 파이프라인으로 평가합니다.

```bash
python scripts/evaluation/evaluate_npu_direct_depth_official.py \
  --output_dir outputs/npu_direct_depth/ \
  --dataset_path /path/to/dataset/
```

---

### `evaluate_dual_head.py`
**Dual-Head NPU 평가**

NPU의 Dual-Head 출력(Integer + Fractional)을 평가합니다.

```bash
python scripts/evaluation/evaluate_dual_head.py \
  --npu_output_dir outputs/npu_dual_head/ \
  --gt_dir /path/to/groundtruth/
```

---

### `evaluate_dual_head_simple.py`
**Dual-Head 간편 평가**

Dual-Head 모델의 평가를 간편하게 수행합니다.

```bash
python scripts/evaluation/evaluate_dual_head_simple.py \
  --checkpoint path/to/model.ckpt \
  --data_path /path/to/dataset/
```

---

### `verify_dual_head_output.py`
**Dual-Head 출력 검증**

Dual-Head 모델의 Integer + Fractional 출력이 올바르게 생성되는지 검증합니다.

```bash
python scripts/evaluation/verify_dual_head_output.py \
  --checkpoint path/to/model.ckpt \
  --image test_image.jpg
```

---

### `verify_gt_rgb_matching.py`
**GT-RGB 매칭 검증**

Ground Truth와 RGB 이미지의 매칭 상태를 검증합니다.

```bash
python scripts/evaluation/verify_gt_rgb_matching.py \
  --dataset_path /path/to/dataset/
```

---

## 🎯 사용 시나리오

| 상황 | 사용 스크립트 |
|------|-------------|
| PyTorch 모델로 예측 생성 | `generate_pytorch_predictions.py` |
| 미리 생성된 깊이 평가 | `eval_precomputed_simple.py` |
| NPU Direct Depth 평가 | `evaluate_npu_direct_depth_official.py` |
| NPU Dual-Head 평가 | `evaluate_dual_head.py` |
| 모델 출력 검증 | `verify_dual_head_output.py` |
| 데이터 검증 | `verify_gt_rgb_matching.py` |

---

## 💡 팁

- 평가 전 출력 디렉토리가 제대로 생성되었는지 확인
- NPU 출력은 규정된 형식(`.npy`, `.npz` 등)이어야 함
- 대량 평가 시 `eval_precomputed_simple.py` 사용 권장
