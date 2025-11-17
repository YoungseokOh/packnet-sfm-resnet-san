# Scripts - ONNX Conversion (ONNX 변환)

PyTorch 모델을 ONNX 형식으로 변환하고 검증하는 스크립트들입니다.

## 📖 주요 스크립트

### `convert_to_onnx.py`
**기본 ONNX 변환**

PyTorch 모델을 기본 ONNX 형식으로 변환합니다.

```bash
python scripts/onnx_conversion/convert_to_onnx.py \
  --checkpoint path/to/model.ckpt \
  --output_path model.onnx \
  --input_size 384 640  # (height, width)
```

---

### `convert_dual_head_to_onnx.py`
**Dual-Head ONNX 변환**

Dual-Head 모델(Integer + Fractional)을 ONNX로 변환합니다.  
Integer/Fractional 헤드를 분리하여 내보냅니다.

```bash
python scripts/onnx_conversion/convert_dual_head_to_onnx.py \
  --checkpoint path/to/model.ckpt \
  --output_dir outputs/onnx/ \
  --simplify  # ONNX 단순화 옵션
```

**출력**:
```
outputs/onnx/
├── model_int_frac.onnx        # 전체 Dual-Head
├── model_integer_only.onnx    # Integer만
└── model_fractional_only.onnx # Fractional만
```

---

### `validate_dual_head_onnx.py`
**Dual-Head ONNX 검증**

ONNX 모델의 Integer/Fractional 출력이 올바른지 검증합니다.

```bash
python scripts/onnx_conversion/validate_dual_head_onnx.py \
  --onnx_path model_int_frac.onnx \
  --pytorch_checkpoint path/to/model.ckpt \
  --test_image test.jpg
```

---

### `test_onnx_with_real_image.py`
**ONNX 실제 이미지 테스트**

ONNX 모델을 실제 이미지로 테스트합니다.

```bash
python scripts/onnx_conversion/test_onnx_with_real_image.py \
  --onnx_model model.onnx \
  --image test.jpg \
  --output_depth output_depth.npy
```

---

### `save_dual_head_outputs.py`
**Dual-Head 출력 저장**

ONNX Dual-Head 모델의 Integer/Fractional 출력을 분리하여 저장합니다.

```bash
python scripts/onnx_conversion/save_dual_head_outputs.py \
  --onnx_model model.onnx \
  --image_dir /path/to/images/ \
  --output_dir outputs/dual_head_outputs/
```

**출력**:
```
outputs/dual_head_outputs/
├── integer/
│   ├── sample_0001.npy
│   └── ...
└── fractional/
    ├── sample_0001.npy
    └── ...
```

---

## 🎯 사용 시나리오

| 목적 | 사용 스크립트 |
|------|-------------|
| 기본 변환 | `convert_to_onnx.py` |
| Dual-Head 변환 | `convert_dual_head_to_onnx.py` |
| 모델 검증 | `validate_dual_head_onnx.py` |
| 실제 데이터 테스트 | `test_onnx_with_real_image.py` |
| 배치 추론 | `save_dual_head_outputs.py` |

---

## 📊 변환 워크플로우

```
PyTorch Model
    ↓
convert_to_onnx.py (또는 convert_dual_head_to_onnx.py)
    ↓
ONNX Model
    ↓
validate_dual_head_onnx.py ✓ 검증
    ↓
test_onnx_with_real_image.py ✓ 테스트
    ↓
save_dual_head_outputs.py ✓ 배치 추론
    ↓
Depth Predictions (NPY/NPZ)
```

---

## 💡 팁

### ONNX 변환 시 주의사항

1. **입력 크기 확인**
   ```bash
   # 모델과 동일한 크기 사용
   python scripts/onnx_conversion/convert_to_onnx.py \
     --checkpoint model.ckpt \
     --input_size 384 640  # (height, width)
   ```

2. **Opset 버전**
   - 최신 ONNX 형식 권장
   - 일부 연산자는 특정 opset 이상 필요

3. **모델 단순화**
   ```bash
   --simplify  # ONNX 최적화 옵션
   ```

### 검증 팁

1. PyTorch와 ONNX 출력 비교
   ```bash
   validate_dual_head_onnx.py \
     --verbose  # 상세 출력
   ```

2. 출력 차이 확인
   - MAE(Mean Absolute Error) < 0.01mm 권장

---

## 🔧 troubleshooting

### ONNX 내보내기 실패
```bash
# 디버그 모드
python scripts/onnx_conversion/convert_to_onnx.py \
  --checkpoint model.ckpt \
  --verbose
```

### 검증 실패
```bash
# 호환성 확인
python scripts/onnx_conversion/validate_dual_head_onnx.py \
  --onnx_path model.onnx \
  --pytorch_checkpoint model.ckpt \
  --num_test_samples 5  # 소수 샘플로 테스트
```

---

## 📚 참고

- [ONNX 공식 문서](https://onnx.ai/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
