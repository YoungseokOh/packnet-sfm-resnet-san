# 📊 ResNet18-SAN 발표자료 목차

## 프로젝트: NPU 최적화 Monocular Depth Estimation

---

## 📑 문서 구성

| 섹션 | 파일명 | 설명 |
|------|--------|------|
| 1 | [01_Introduction.md](01_Introduction.md) | 연구 배경, 목표, 기여점 |
| 2 | [02_Related_Work.md](02_Related_Work.md) | 관련 연구 (Depth Estimation, SAN, 양자화) |
| 3 | [03_Method.md](03_Method.md) | ResNet18-SAN 아키텍처 상세 설명 |
| 4 | [04_Experiments.md](04_Experiments.md) | 실험 설정, 데이터셋, 결과 |
| 5 | [05_Future_Works.md](05_Future_Works.md) | 향후 연구 방향 |

---

## 🎯 발표 요약

### 핵심 메시지
> **"NPU INT8 양자화에 최적화된 Dual-Head 아키텍처로 경량 Monocular Depth Estimation 달성"**

### 주요 성과
- **FP32 성능**: abs_rel 0.0414, RMSE 0.469m
- **INT8 양자화 목표**: 14배 정밀도 향상 (±28mm → ±2mm)
- **실시간 처리**: 640×384 해상도, Edge NPU 타겟

### 기술적 차별점
1. **Dual-Head Architecture**: Integer-Fractional 분리 출력
2. **ResNet18 Backbone**: 경량화된 Encoder
3. **SAN (Sparse Attention Network)**: LiDAR Sparse Depth 활용
4. **NPU 최적화**: Per-tensor INT8 양자화 대응 설계

---

## 📅 발표 예상 시간

| 섹션 | 예상 시간 |
|------|-----------|
| Introduction | 5분 |
| Related Work | 5분 |
| Method | 15분 |
| Experiments | 10분 |
| Future Works | 5분 |
| Q&A | 10분 |
| **Total** | **50분** |

---

## 🔗 참고 자료

- **프로젝트 저장소**: `/workspace/packnet-sfm`
- **상세 문서**: `docs/quantization/ST2/`
- **학습 설정**: `configs/train_resnet_san_ncdb_dual_head_640x384.yaml`
- **모델 코드**: `packnet_sfm/networks/depth/ResNetSAN01.py`
