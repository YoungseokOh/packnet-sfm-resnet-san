# 📚 PackNet-SFM ResNet-SAN Documentation

PackNet-SFM ResNet-SAN Dual-Head 모델 관련 모든 문서를 한곳에서 관리합니다.

---

## 📂 폴더 구조 및 가이드

### 🏛️ [`architecture/`](architecture/)
**아키텍처 및 구조 관련 문서**

- `00_FOLDER_STRUCTURE_GUIDE.md` - 전체 폴더 구조 및 파일 위치 설명
- 아키텍처 개요 및 설계 문서
- 모델 구조 상세 설명

**대상**: 신규 사용자, 아키텍처 이해가 필요한 사람

---

### 🎓 [`training/`](training/)
**학습 관련 가이드**

- 학습 방법 및 튜토리얼
- 설정 파일 설명
- 최적화 팁 및 권장사항

**대상**: 모델 학습을 하려는 사람

---

### 🔢 [`quantization/`](quantization/)
**양자화 관련 문서**

- INT8 양자화 방법
- Calibration 가이드
- 양자화 효과 분석

**대상**: ONNX 변환, 양자화를 하려는 사람

---

### 📊 [`analysis/`](analysis/)
**분석, 검증, 시각화 자료**

- `00_INDEX.md` - 분석 자료 색인
- 성능 분석 보고서
- 파라미터 민감도 분석
- 시각화 이미지 (images/ 폴더)
- 분석 스크립트 참고 코드 (reference_scripts/ 폴더)

**대상**: 성능 분석, 파라미터 이해가 필요한 사람

---

### 🏗️ [`implementation/`](implementation/)
**구현 세부사항**

- `00_SCRIPTS_ANALYSIS_AND_ORGANIZATION.md` - scripts 폴더 상세 분석
- Dual-Head 구현 문서
- Direct Depth 구현 문서
- INT8 양자화 구현 문서
- 코드 정리 및 파일 분류 가이드

**대상**: 코드 수정, 새 기능 추가를 하려는 사람

---

### 📖 [`reference/`](reference/)
**참고 자료**

- 논문 링크 및 PDF 저장소 (papers/ 폴더)
- 관련 링크 및 리소스

**대상**: 기술 문헌, 논문 참고가 필요한 사람

---

### 🔧 [`technical/`](technical/)
**기술 명세 및 상세 분석**

- NCDB 비디오 프로젝션 명세
- 기술 구현 세부사항

**대상**: 기술 깊이 있게 이해하려는 사람

---

### 🚀 [`futurework/`](futurework/)
**향후 계획 및 개선 사항**

- 미래 개발 계획
- 개선 아이디어

**대상**: 장기 로드맵 검토

---

## 🎯 사용자별 추천 경로

### 👤 신규 사용자 (처음 시작)
```
1. architecture/00_FOLDER_STRUCTURE_GUIDE.md    (폴더 구조 이해)
2. architecture/                                  (전체 아키텍처)
3. training/                                      (학습 시작)
4. scripts/core/README.md                        (실제 실행)
```

### 👨‍💼 구현자 (코드 수정/개선)
```
1. implementation/00_SCRIPTS_ANALYSIS_AND_ORGANIZATION.md  (코드 구조)
2. implementation/                                           (구현 세부)
3. packnet_sfm/                                             (코드 확인)
4. analysis/                                                (성능 검증)
```

### 🔬 연구자 (새 기능 추가)
```
1. analysis/00_INDEX.md              (기존 분석 검토)
2. analysis/reference_scripts/        (분석 코드 참고)
3. training/                          (학습 방식)
4. quantization/                      (배포 최적화)
```

### 📊 분석가 (성능 평가)
```
1. analysis/00_INDEX.md              (분석 색인)
2. analysis/                          (성능 보고서)
3. scripts/analysis/README.md         (분석 스크립트)
4. scripts/evaluation/README.md       (평가 스크립트)
```

---

## 🔑 핵심 문서 빠른 링크

### 아키텍처 이해
- 📄 [폴더 구조 가이드](architecture/00_FOLDER_STRUCTURE_GUIDE.md)
- 📄 [아키텍처 개요](architecture/)

### 파라미터 분석
- 📄 [가중치 설정 분석](analysis/01_WEIGHT_JUSTIFICATION_SUMMARY.md)
- 📄 [Min-Depth 효과](analysis/03_MIN_DEPTH_EFFECTS.md)
- 📄 [48 레벨 영향](analysis/04_48_IMPACT.md)
- 📄 [Consistency Weight](analysis/05_CONSISTENCY_WEIGHT.md)

### 구현 세부사항
- 📄 [Scripts 폴더 분석](implementation/00_SCRIPTS_ANALYSIS_AND_ORGANIZATION.md)
- 📄 [Dual-Head 구현](implementation/Dual-Head/)
- 📄 [ONNX 변환](implementation/DIRECT_DEPTH_ONNX_CONVERSION.md)

### 학습 및 평가
- 🏃 [학습 방법](training/)
- ✅ [평가 스크립트](../scripts/evaluation/README.md)
- 🎨 [시각화 도구](../scripts/visualization/README.md)

---

## 🚀 빠른 시작 명령어

### 학습
```bash
python scripts/core/train.py configs/train_resnet_san_ncdb_dual_head_640x384.yaml
```

### 추론
```bash
python scripts/core/infer.py --checkpoint model.ckpt --image test.jpg
```

### 평가
```bash
python scripts/core/eval_official.py --checkpoint model.ckpt --config eval_config.yaml
```

### ONNX 변환
```bash
python scripts/onnx_conversion/convert_dual_head_to_onnx.py --checkpoint model.ckpt
```

### 시각화
```bash
python scripts/visualization/visualize_fp32_vs_int8_comparison.py --fp32_dir outputs/fp32/ --int8_dir outputs/int8/
```

---

## 📋 폴더별 파일 목록

| 폴더 | 파일 수 | 설명 |
|------|--------|------|
| `architecture/` | 1+ | 구조 및 아키텍처 |
| `training/` | - | 학습 가이드 |
| `quantization/` | - | 양자화 문서 |
| `analysis/` | 13+ | 분석 및 검증 자료 |
| `implementation/` | 31+ | 구현 세부사항 |
| `reference/` | - | 참고 자료 |
| `technical/` | 1+ | 기술 명세 |
| `futurework/` | - | 향후 계획 |

---

## 💡 팁

1. **문서 업데이트**: 각 폴더마다 상세한 README.md가 있습니다
2. **코드 참고**: `docs/analysis/reference_scripts/`에 분석 코드 저장
3. **이미지 보기**: `docs/analysis/images/`에 시각화 이미지 저장
4. **스크립트 가이드**: `scripts/*/README.md`에 각 스크립트 사용법

---

## 📞 문의 및 피드백

질문이나 문제가 있으면:
1. 해당 폴더의 README.md 확인
2. 분석 문서(`analysis/`) 검토
3. 구현 세부사항(`implementation/`) 참고

---

**마지막 업데이트**: 2025-11-17  
**상태**: 🟢 활성 관리 중
