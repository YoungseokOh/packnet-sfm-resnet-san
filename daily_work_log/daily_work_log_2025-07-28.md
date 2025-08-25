# PM 업무 기록 (2025년 7월 28일)

## 최종 목표
100개 샘플 데이터셋을 사용하여 100 에포크(epoch) 동안 모델을 학습시키고, 그 결과를 분석하여 전문적인 PM 보고서를 작성한다.

## 근본 원인 및 최종 해결 전략
- **근본 원인**: 데이터 생성 파이프라인의 비호환성. `create_ncdb_sample.py`는 딕셔너리 형식의 `mapping_data.json`을 생성하는 반면, `create_depth_maps.py`와 `ncdb_dataset.py`는 리스트 형식을 기대하여 충돌이 발생했다.
- **최종 해결책**: 데이터 생성 스크립트의 출력물(딕셔너리 형식 JSON)에 소비 스크립트(깊이 맵 생성, 데이터셋 로더)를 맞추는 방식으로 데이터 파이프라인을 통일한다.

## 최종 수행 계획

### 1. 데이터셋 재구성 (The Right Way)
- **Step 1: 샘플 데이터 생성.** `scripts/create_ncdb_sample.py`를 실행하여 `ncdb-cls-sample` 디렉토리와 딕셔너리 형식의 `mapping_data.json`을 생성한다.
  ```bash
  python scripts/create_ncdb_sample.py
  ```
- **Step 2: `create_depth_maps.py` 스크립트 패치.** `create_depth_maps.py`가 딕셔너리 형식의 `mapping_data.json`을 파싱하도록 수정 완료.
- **Step 3: 깊이 맵 생성.** 패치된 `create_depth_maps.py`를 실행하여 `depth_maps`를 생성 완료.
  ```bash
  python scripts/create_depth_maps.py --parent ncdb-cls-sample --output-dir ncdb-cls-sample/synced_data/depth_maps
  ```

### 2. 학습 설정 및 실행
- **Step 4: `ncdb_dataset.py` 확인 및 패치.** `ncdb_dataset.py`가 딕셔너리 형식의 `mapping_data.json`을 올바르게 처리하는지 확인하고, 필요시 수정한다. (확인 완료, 수정 불필요)
- **Step 5: 학습 설정 파일(YAML) 생성.** `configs/train_resnet_san_ncdb.yaml`을 기반으로, 100 에포크 및 샘플 데이터셋 경로를 설정한 임시 YAML 파일을 생성한다. (생성 완료: `configs/temp_train_ncdb_100_epochs.yaml`)
- **Step 6: 학습 실행.** 모든 데이터 준비가 완료된 후, 모델 학습을 시작하고 임시 파일들을 정리한다.
  ```bash
  python scripts/train.py configs/temp_train_ncdb_100_epochs.yaml
  rm configs/temp_train_ncdb_100_epochs.yaml
  ```

### 3. 결과 보고
- **Step 7: 최종 보고서 작성.** 학습 완료 후, TensorBoard 로그를 기반으로 정량적/정성적 분석을 포함한 `pm_report_2025-07-28.md`를 작성한다.

## 오늘 한 일 (2025년 7월 28일)
- `@scripts/create_depth_maps.py` 파일 수정: `create_ncdb_sample.py`가 생성하는 딕셔너리 형식의 `mapping_data.json`을 처리하도록 `process_folder` 함수를 업데이트했습니다.
- `@create_ncdb_sample.py` 실행: 100개 샘플 데이터셋을 성공적으로 재생성했습니다.
- `open3d` 라이브러리 설치: PCD 파일 로딩 오류 해결을 위해 `open3d`를 설치했습니다.
- `@scripts/create_depth_maps.py` 실행: 깊이 맵 생성을 완료했습니다. 이전의 `UnicodeDecodeError`는 더 이상 발생하지 않았습니다.
- `ncdb_dataset.py` 확인: `mapping_data.json` 딕셔너리 형식을 올바르게 처리함을 확인했습니다.
- 임시 학습 설정 파일 생성: `configs/temp_train_ncdb_100_epochs.yaml`을 생성했습니다.
- 100개 샘플 데이터셋으로 모델 학습을 1회 완료했습니다.

## 앞으로 할 일
- TensorBoard 이미지 문제 해결: TensorBoard에 표시되는 Depth GT, Image, Inference 결과가 매 epoch마다 다른 이미지로 표시되도록 수정합니다.
- 학습 완료 후 최종 보고서를 작성합니다.