# Scripts - Core (핵심 스크립트)

모델 학습, 추론, 평가에 사용되는 **필수 스크립트**입니다.

## 📖 주요 스크립트

### `train.py`
**모델 학습**

학습 설정 파일을 기반으로 모델을 학습합니다.

```bash
# 기본 학습
python scripts/core/train.py configs/train_resnet_san_ncdb_dual_head_640x384.yaml

# Horovod 분산 학습 (multi-GPU)
horovodrun -np 4 python scripts/core/train.py configs/train_resnet_san_ncdb_dual_head_640x384.yaml
```

---

### `infer.py`
**모델 추론**

학습된 모델로 깊이를 예측합니다.

```bash
# 단일 이미지 추론
python scripts/core/infer.py \
  --checkpoint path/to/model.ckpt \
  --image image.jpg

# 배치 추론
python scripts/core/infer.py \
  --checkpoint path/to/model.ckpt \
  --image_dir path/to/images/
```

---

### `eval.py`
**공식 평가 스크립트**

데이터셋에 대해 모델의 평가 지표를 계산합니다.

```bash
# KITTI 평가
python scripts/core/eval.py \
  --checkpoint path/to/model.ckpt \
  --dataset kitti \
  --data_path /path/to/kitti/

# NCDB 평가
python scripts/core/eval.py \
  --checkpoint path/to/model.ckpt \
  --dataset ncdb \
  --data_path /path/to/ncdb/
```

---

### `eval_official.py`
**공식 평가 스크립트 (수정 버전)**

`eval.py`의 개선 버전으로, 더 많은 옵션을 지원합니다.

```bash
# Val/Test split 모두 평가
python scripts/core/eval_official.py \
  --checkpoint path/to/model.ckpt \
  --config configs/eval_resnet_san_kitti.yaml
```

---

## 🎯 사용 순서

1. **학습**: `train.py` 실행
2. **추론**: `infer.py`로 테스트 이미지 추론
3. **평가**: `eval.py` 또는 `eval_official.py`로 지표 계산

---

## 💡 팁

- 학습 전 `configs/` 폴더에서 적절한 설정 파일 선택
- GPU 사용 가능 여부 자동 감지
- Horovod 설치 시 분산 학습 가능
