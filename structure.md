## 1. 프로젝트 주요 구조

*   `/packnet_sfm/`: 핵심 라이브러리 코드를 포함합니다.
    *   `models/`: 모델 아키텍처 정의 (`ModelWrapper` 포함).
    *   `datasets/`: 데이터셋 로딩 및 전처리.
    *   `geometry/`: 카메라 및 포즈 관련 기하학적 유틸리티.
    *   `loggers/`: 학습 로깅 (예: WandbLogger, TensorboardLogger).
    *   `losses/`: 손실 함수 정의.
    *   `networks/`: 신경망 구성 요소.
    *   `trainers/`: 학습 및 평가 로직 (`HorovodTrainer` 포함).
    *   `utils/`: 다양한 유틸리티 함수.
*   `/scripts/`: 실행 가능한 스크립트.
    *   `train.py`: 모델 학습을 시작하는 메인 스크립트.
    *   `eval.py`: 모델 평가를 시작하는 메인 스크립트.
*   `/configs/`: 학습 및 평가를 위한 YAML 설정 파일.
    *   `train_*.yaml`: 학습 관련 설정.
    *   `eval_*.yaml`: 평가 관련 설정.
*   `/checkpoints/`: 학습된 모델 체크포인트가 저장되는 디렉토리.
*   `/outputs/`: 일반적으로 평가 결과 (깊이 맵, 시각화 등)가 저장되는 디렉토리.

## 2. 학습 파이프라인

PackNet-SfM의 학습 과정은 `scripts/train.py` 스크립트를 통해 시작됩니다.

1.  **설정 로드**: `train.py`는 `.yaml` 설정 파일 또는 기존 `.ckpt` 파일에서 학습 설정을 파싱합니다. 이 설정은 모델 아키텍처, 데이터셋 경로, 학습 하이퍼파라미터 등을 정의합니다.
2.  **분산 학습 초기화**: Horovod를 사용하여 분산 학습 환경을 초기화합니다.
3.  **로거 설정**: Wandb 및 TensorBoard 로거가 설정되어 학습 진행 상황, 손실, 메트릭 등을 기록합니다. TensorBoard 로그는 `config.save.folder` 내의 타임스탬프가 붙은 폴더에 저장됩니다.
4.  **체크포인트 관리**: `ModelCheckpoint`가 설정되어 학습 중 주기적으로 모델의 상태를 저장합니다.
5.  **모델 및 트레이너 초기화**:
    *   `ModelWrapper`는 실제 모델, 옵티마이저, 스케줄러 및 데이터 로딩을 관리합니다.
    *   `HorovodTrainer`는 학습 루프를 오케스트레이션합니다.
6.  **학습 루프 (`trainer.fit(model_wrapper)`)**:
    *   `HorovodTrainer`의 `fit` 메서드가 호출되어 학습을 시작합니다.
    *   훈련 데이터로더와 검증 데이터로더를 준비합니다.
    *   각 에포크마다 `train_with_eval` 메서드를 통해 훈련 스텝을 수행하며, 이 과정에서 주기적으로 검증 데이터셋의 일부를 사용하여 "빠른 평가"를 수행합니다.
    *   각 에포크의 검증 후, `_save_eval_results` 메서드를 통해 평가 결과가 체크포인트 디렉토리 내의 `evaluation_results` 폴더에 JSON 파일로 저장됩니다.

## 3. 평가 과정

모델 평가는 `scripts/eval.py` 스크립트를 통해 수행됩니다.

1.  **설정 및 체크포인트 로드**: `eval.py`는 평가할 모델의 `.ckpt` 체크포인트 파일을 로드합니다. 이 체크포인트에는 학습 시 사용된 설정 정보가 포함되어 있습니다.
2.  **분산 환경 초기화**: Horovod가 초기화됩니다.
3.  **모델 로드**: `ModelWrapper`를 사용하여 체크포인트로부터 모델 상태를 복원합니다.
4.  **트레이너 초기화**: `HorovodTrainer`가 평가 모드로 초기화됩니다.
5.  **평가 실행 (`trainer.test(model_wrapper)`)**:
    *   `HorovodTrainer`의 `test` 메서드가 호출됩니다.
    *   `test` 메서드는 `module.test_dataloader()`를 통해 테스트 데이터로더를 가져옵니다.
    *   `self.evaluate(test_dataloaders, module)`를 호출하여 실제 평가 루프를 실행합니다.
    *   `evaluate` 메서드는 테스트 데이터셋의 각 배치에 대해 `module.test_step()`을 호출하여 깊이 추정 및 기타 메트릭을 계산합니다.
    *   **결과 저장**: `evaluate` 메서드는 각 배치의 출력을 수집하고, 최종적으로 `module.test_epoch_end()`를 호출하여 모든 테스트 데이터셋에 대한 평가를 완료합니다. 깊이 맵 및 시각화 결과의 실제 파일 저장 로직은 `ModelWrapper` 내부의 `test_step` 또는 `test_epoch_end` 메서드에 구현되어 있으며, 일반적으로 `outputs/` 디렉토리에 저장됩니다. `scripts/eval.py` 자체는 명시적으로 이미지 파일을 저장하는 코드를 포함하지 않지만, `HorovodTrainer`가 수집한 `outputs`를 통해 후처리 및 저장이 이루어집니다.

## 4. `scripts/` 디렉토리 파일 분석

### `scripts/advanced_verify.py`
*   **목적**: 깊이 맵의 데이터 무결성을 고급 방식으로 검증하고 시각화합니다.
*   **기능**:
    *   LiDAR 포인트 클라우드(PCD)에서 깊이 맵으로 포인트를 투영하여 원본 PCD 값과 생성된 깊이 맵의 값을 비교합니다.
    *   로그 스케일 컬러맵을 사용하여 깊이 맵을 시각화하고, 이를 원본 이미지 위에 오버레이하여 저장합니다.
    *   `open3d` 라이브러리가 필요하며, 없으면 기본 PCD 파서로 대체됩니다.

### `scripts/analyze_model_differences.py`
*   **목적**: ResNet-SAN 모델과 YOLOv8-SAN 모델 간의 차이점을 종합적으로 분석합니다.
*   **기능**:
    *   **모델 복잡도 분석**: 파라미터 수, 메모리 사용량, 특징 맵의 크기 및 복잡도를 비교합니다.
    *   **RGB 특징 품질 분석**: 균일, 그라디언트, 체커보드, 노이즈, 에지 등 다양한 합성 이미지와 실제 KITTI 이미지를 사용하여 특징 맵의 분산 및 그라디언트 크기를 측정하여 특징 품질을 평가합니다.
    *   **LiDAR 융합 메커니즘 분석**: 모델 내 융합 레이어의 가중치 분포를 분석하여 LiDAR 데이터가 어떻게 통합되는지 파악합니다.
    *   **레이어별 활성화 분석**: 각 레이어의 활성화 분포(평균, 표준편차, 희소성 등)를 추출하고 비교합니다.
    *   **모델 해석 가능성 분석**: Grad-CAM을 사용하여 모델의 어텐션 맵을 생성하고, 어텐션의 집중도(엔트로피)를 측정하여 모델이 이미지의 어느 부분에 집중하는지 시각화합니다.
    *   분석 결과를 시각화된 플롯과 텍스트 보고서로 저장합니다.

### `scripts/analyze_onnx.py`
*   **목적**: ONNX 모델의 유효성을 검사하고 입력/출력 정보를 출력합니다.
*   **기능**:
    *   주어진 경로의 ONNX 모델을 로드하고 ONNX 체커를 사용하여 모델의 유효성을 검증합니다.
    *   모델의 입력 및 출력 텐서의 이름, 데이터 타입, 형태(shape)를 상세히 출력합니다.

### `scripts/check_mask.py`
*   **목적**: 특정 경로의 이진 마스크(binary mask) 파일의 속성을 확인합니다.
*   **기능**:
    *   마스크 이미지 파일을 로드하여 형태, 데이터 타입, 고유 값, 최소/최대 값을 출력합니다.
    *   마스크가 0 또는 255 값만 포함하는 이진 마스크인지 확인합니다.

### `scripts/check_yolov8_model_type.py`
*   **목적**: YOLOv8 모델의 다양한 타입(COCO Detection, ImageNet Classification)의 구조를 비교합니다.
*   **기능**:
    *   `ultralytics` 라이브러리를 사용하여 `yolov8s.pt` (COCO Detection) 및 `yolov8s-cls.pt` (ImageNet Classification) 모델을 로드합니다.
    *   각 모델의 태스크, 모델 타입, 백본 특징 채널 수, 탐지 헤드 유무 등을 출력하여 구조적 차이를 보여줍니다.

### `scripts/compare_depth_pcd.py`
*   **목적**: 생성된 깊이 맵을 원본 LiDAR 포인트 클라우드와 비교하여 깊이 차이 통계를 계산합니다.
*   **기능**:
    *   LiDAR 포인트 클라우드를 카메라 좌표계로 투영하고, 투영된 포인트의 깊이와 생성된 깊이 맵의 해당 픽셀 값을 비교합니다.
    *   평균 절대 오차(MAE), RMSE, 최대/최소 절대 오차 등 다양한 깊이 메트릭을 계산합니다.
    *   `create_depth_maps.py`와 동일한 카메라 모델 및 캘리브레이션 로직을 사용합니다.

### `scripts/convert_png_to_jpg.py`
*   **목적**: 지정된 입력 디렉토리의 모든 PNG 이미지 파일을 JPG 형식으로 변환하여 출력 디렉토리에 저장합니다.
*   **기능**:
    *   재귀적으로 PNG 파일을 찾아 JPG로 변환하고, 원본 PNG 파일을 삭제하는 옵션을 포함합니다.
    *   변환된 파일 수와 원본 파일 수를 비교하여 변환 성공 여부를 검증합니다.

### `scripts/convert_to_onnx.py`
*   **목적**: PackNet-SfM PyTorch 모델 체크포인트를 ONNX 형식으로 변환합니다.
*   **기능**:
    *   주어진 `.ckpt` 체크포인트 파일에서 깊이 추정 네트워크를 로드합니다.
    *   모델을 ONNX 형식으로 내보내기 위한 간단한 래퍼를 생성합니다.
    *   NNEF 호환성을 위해 `ReflectionPad2d` 레이어를 `ZeroPad2d`로 선택적으로 패치할 수 있습니다.
    *   변환된 ONNX 모델의 크기 및 출력 경로를 보고합니다.

### `scripts/create_combined_splits.py`
*   **목적**: 두 개의 데이터셋(`mapping_data.json` 파일 기반)을 통합하여 단일 학습/검증/테스트 스플릿을 생성합니다.
*   **기능**:
    *   각 데이터셋의 `mapping_data.json`을 로드하고, 각 항목에 원본 데이터셋의 루트 경로를 추가합니다.
    *   모든 데이터를 섞은 후, 지정된 비율에 따라 학습, 검증, 테스트 세트로 분할합니다.
    *   분할된 데이터를 새로운 JSON 파일로 저장합니다.

### `scripts/create_data_splits.py`
*   **목적**: `mapping_data.json` 파일을 학습, 검증, 테스트 세트로 분할합니다.
*   **기능**:
    *   `mapping_data.json`의 구조(리스트 또는 딕셔너리)를 감지하여 적절하게 처리합니다.
    *   데이터를 섞고, 지정된 비율(기본값: 학습 80%, 검증 10%, 테스트 10%)에 따라 분할합니다.
    *   분할된 데이터를 `train.json`, `val.json`, `test.json` 파일로 저장합니다.

### `scripts/create_depth_maps.py`
*   **목적**: LiDAR 포인트 클라우드에서 KITTI 스타일의 깊이 맵을 생성합니다.
*   **기능**:
    *   LiDAR 포인트 클라우드를 카메라 이미지 평면에 투영하여 깊이 맵을 생성합니다.
    *   VADAS Fisheye 카메라 모델 및 캘리브레이션 데이터를 사용하여 정확한 투영을 수행합니다.
    *   생성된 깊이 맵은 16비트 PNG 이미지로 저장됩니다.
    *   특정 LiDAR 포인트(예: 차량 내부)를 깊이 맵 생성에서 제외하는 로직을 포함합니다.

### `scripts/create_kitti_sample.py`
*   **목적**: KITTI 데이터셋에서 단일 이미지를 로드하고 지정된 크기로 리사이즈하여 샘플 이미지를 생성합니다.
*   **기능**:
    *   KITTI 디렉토리에서 `image_02/data`의 첫 번째 이미지를 읽습니다.
    *   지정된 너비와 높이로 이미지를 리사이즈하고 출력 경로에 저장합니다.

### `scripts/create_ncdb_sample.py`
*   **목적**: NCDB 데이터셋에서 지정된 수의 샘플을 추출하여 작은 샘플 데이터셋을 생성합니다.
*   **기능**:
    *   원본 NCDB 데이터셋에서 무작위로 샘플 ID를 선택합니다.
    *   선택된 샘플에 해당하는 이미지, PCD, 투영 결과 파일 및 `mapping_data.json`을 새로운 샘플 디렉토리로 복사합니다.

### `scripts/eval.py`
*   **목적**: PackNet-SfM 모델의 깊이 추정 성능을 평가합니다.
*   **기능**:
    *   학습된 모델 체크포인트와 설정 파일을 로드합니다.
    *   `HorovodTrainer`를 사용하여 모델을 평가 모드로 설정하고, 테스트 데이터셋에 대한 추론을 수행합니다.
    *   평가 결과는 `HorovodTrainer` 내부에서 처리되며, 일반적으로 깊이 맵 및 시각화 결과가 `outputs/` 디렉토리에 저장됩니다.

### `scripts/eval_onnx.py`
*   **목적**: ONNX 형식으로 변환된 PackNet-SfM 모델의 깊이 추정 성능을 평가합니다.
*   **기능**:
    *   ONNX 모델과 원본 PyTorch 체크포인트의 설정을 로드합니다.
    *   ONNX Runtime을 사용하여 이미지에서 깊이 맵을 추론합니다.
    *   추론된 깊이 맵을 KITTI 데이터셋의 Ground Truth 깊이 맵과 비교하여 `abs_rel`, `rmse` 등 표준 깊이 메트릭을 계산합니다.
    *   평가 결과를 콘솔에 출력하고 텍스트 파일로 저장합니다.

### `scripts/eval_pytorch_onnx_comparison.py`
*   **목적**: PyTorch 모델과 ONNX 모델 간의 깊이 추정 성능을 직접 비교합니다.
*   **기능**:
    *   동일한 입력 이미지에 대해 PyTorch 모델과 ONNX 모델 각각으로 깊이 추론을 수행합니다.
    *   두 모델의 추론 결과를 Ground Truth 깊이 맵과 비교하여 각 모델의 성능 메트릭을 계산합니다.
    *   PyTorch와 ONNX 모델 간의 메트릭 차이를 상세하게 보고하여 변환 과정에서의 성능 변화를 분석합니다.

### `scripts/evaluate_depth_maps.py`
*   **목적**: 예측된 깊이 맵과 Ground Truth 깊이 맵을 비교하여 깊이 추정 메트릭을 계산합니다.
*   **기능**:
    *   예측된 깊이 맵 파일(`npz` 또는 `png`)과 Ground Truth 깊이 맵 파일을 로드합니다.
    *   `compute_depth_metrics` 함수를 사용하여 `abs_rel`, `rmse`, `a1` 등 표준 깊이 메트릭을 계산합니다.
    *   계산된 평균 메트릭을 콘솔에 출력하고 지정된 파일에 저장합니다.

### `scripts/infer.py`
*   **목적**: 단일 이미지 또는 이미지 폴더에서 깊이 맵을 추론하고 시각화합니다.
*   **기능**:
    *   학습된 모델 체크포인트를 로드하고, 입력 이미지를 모델 입력 크기에 맞게 전처리합니다.
    *   모델을 사용하여 역깊이(inverse depth)를 추론합니다.
    *   추론된 깊이 맵을 `.npz` 또는 `.png` 형식으로 저장하거나, 원본 이미지와 깊이 시각화 결과를 결합하여 시각화된 이미지로 저장합니다.

### `scripts/prepare_data.py`
*   **목적**: 리스트 기반의 `mapping_data.json`을 딕셔너리 기반으로 변환하고, 학습을 위한 스플릿 파일을 생성합니다.
*   **기능**:
    *   기존 `mapping_data.json` (리스트 형태)을 읽어 이미지 및 PCD 경로를 추출합니다.
    *   `create_depth_maps.py`와 호환되는 딕셔너리 형태의 `mapping_data_dict.json`을 생성합니다.
    *   학습에 사용할 샘플 파일 이름 목록을 포함하는 스플릿 파일을 생성합니다.

### `scripts/train.py`
*   **목적**: PackNet-SfM 모델의 학습 과정을 시작합니다.
*   **기능**:
    *   학습 설정 파일(`.yaml`) 또는 기존 체크포인트(`.ckpt`)를 로드합니다.
    *   Horovod를 사용하여 분산 학습 환경을 초기화합니다.
    *   Wandb 및 TensorBoard 로거를 설정하여 학습 진행 상황을 모니터링합니다.
    *   `ModelWrapper`를 통해 모델을 초기화하고, `HorovodTrainer`를 사용하여 학습 루프를 실행합니다.
    *   학습 중 주기적으로 검증을 수행하고 체크포인트를 저장합니다.

### `scripts/verify_depth_maps.py`
*   **목적**: 생성된 깊이 맵을 원본 이미지 위에 오버레이하여 시각적으로 검증합니다.
*   **기능**:
    *   깊이 맵과 해당 원본 이미지를 로드합니다.
    *   깊이 값을 특정 컬러맵(JET-like)으로 색상화하고, 이를 원본 이미지 위에 점으로 그려서 깊이 정보를 시각적으로 표현합니다.
    *   시각화된 이미지를 출력 디렉토리에 저장합니다.

### `scripts/visualize_results.py`
*   **목적**: 학습 중 저장된 에포크별 평가 결과를 시각화합니다.
*   **기능**:
    *   `evaluation_results` 폴더에서 `epoch_*_results.json` 파일들을 로드합니다.
    *   각 에포크의 메트릭(예: `abs_rel`, `rmse`)을 추출하여 시계열 그래프로 플로팅합니다.
    *   생성된 플롯을 PNG 이미지 파일로 저장합니다.

## 5. `packnet_sfm/` 디렉토리 파일 분석

### `packnet_sfm/__init__.py`
*   **목적**: PackNet-SfM 라이브러리의 루트 패키지 정보를 제공합니다.
*   **내용**: 버전, 저자, 라이선스, 홈페이지, 문서 요약 및 장문 설명을 포함합니다. PackNet 논문(CVPR 2020 oral)에 대한 참조가 있습니다.

### `packnet_sfm/datasets/__init__.py`
*   **목적**: PackNet-SfM에서 사용되는 데이터셋 클래스에 대한 개요를 제공합니다.
*   **내용**: `KITTIDataset`, `DGPDataset`, `ImageDataset` 등 주요 데이터셋의 역할과 특징을 간략하게 설명합니다.

### `packnet_sfm/datasets/augmentations.py`
*   **목적**: 이미지, 깊이 맵, 카메라 내외부 파라미터 등 데이터셋 샘플에 적용되는 다양한 데이터 증강(augmentation) 및 변환 함수를 제공합니다.
*   **기능**:
    *   `resize_image`, `resize_depth`, `resize_depth_preserve`: 이미지 및 깊이 맵의 크기 조절.
    *   `resize_sample_image_and_intrinsics`, `resize_sample`: 샘플 내 이미지, 내외부 파라미터, 깊이 맵을 함께 크기 조절.
    *   `to_tensor`, `to_tensor_sample`: PIL 이미지 또는 NumPy 배열을 PyTorch 텐서로 변환.
    *   `duplicate_sample`: 원본 이미지/컨텍스트를 복제하여 증강 전 버전을 보존.
    *   `colorjitter_sample`, `random_color_jitter_transform`: 이미지에 색상 지터링 적용.
    *   `crop_image`, `crop_intrinsics`, `crop_depth`, `crop_sample_input`, `crop_sample_supervision`, `crop_sample`: 이미지, 내외부 파라미터, 깊이 맵을 자르기.

### `packnet_sfm/datasets/augmentations_kitti_compatible.py`
*   **목적**: KITTI 데이터셋에 최적화된 고급 데이터 증강(RandAugment, RandomErasing, MixUp, CutMix)을 구현합니다.
*   **기능**:
    *   `RandAugment`: 이미지에 무작위 변환(대비, 균등화, 회전, 색상, 밝기, 선명도)을 적용.
    *   `RandomErasing`: 텐서에 무작위 영역을 지우는 증강 적용.
    *   `KITTIAdvancedTrainTransform`: 학습 시 RandAugment, RandomErasing 등을 포함한 고급 변환 파이프라인.
    *   `KITTIAdvancedValTransform`: 검증/테스트 시 사용되는 변환.
    *   `MixUp`, `CutMix`: 배치 레벨에서 이미지와 깊이 맵을 혼합하는 증강.
    *   `create_kitti_advanced_collate_fn`: 고급 증강을 데이터로더의 `collate_fn`으로 통합.

### `packnet_sfm/datasets/camera_lidar_projector.py`
*   **목적**: LiDAR 포인트 클라우드를 카메라 이미지 평면에 투영하여 깊이 맵을 생성하는 유틸리티를 제공합니다. (이 파일은 제공된 내용에 없었지만, `create_depth_maps.py`와 `compare_depth_pcd.py`에서 사용되는 클래스들이 여기에 정의되어 있을 것으로 추정됩니다.)

### `packnet_sfm/datasets/dgp_dataset.py`
*   **목적**: DGP(Deep Generative Prior) 데이터셋을 로드하고 처리하는 클래스를 제공합니다.
*   **기능**:
    *   `SynchronizedSceneDataset`을 사용하여 DGP 데이터셋에서 이미지, 깊이, 포즈, 시맨틱 정보 등을 로드합니다.
    *   `generate_depth_map`: LiDAR 정보를 투영하여 깊이 맵을 생성하고 캐싱합니다.
    *   `stack_sample`: 여러 센서의 샘플을 스택합니다.
    *   컨텍스트 프레임(이전/이후 프레임)을 로드하는 기능을 포함합니다.

### `packnet_sfm/datasets/generate_lut.py`
*   **목적**: (내용이 제공되지 않았지만) Look-Up Table (LUT)을 생성하는 스크립트로 추정됩니다. 이는 이미지 처리나 캘리브레이션에 사용될 수 있습니다.

### `packnet_sfm/datasets/image_dataset.py`
*   **목적**: 이미지 시퀀스 폴더에서 데이터를 로드하는 일반적인 이미지 데이터셋 클래스입니다. 깊이 맵 지원은 없습니다.
*   **기능**:
    *   주어진 디렉토리에서 이미지 파일을 읽고, 컨텍스트 프레임이 존재하는지 확인합니다.
    *   더미 캘리브레이션(intrinsics)을 생성하여 반환합니다.
    *   `__getitem__` 메서드를 통해 이미지와 컨텍스트 이미지를 로드합니다.

### `packnet_sfm/datasets/kitti_dataset.py`
*   **목적**: KITTI 데이터셋을 로드하고 처리하는 클래스입니다.
*   **기능**:
    *   KITTI `raw` 데이터셋에서 이미지, 깊이 맵(NPZ 또는 PNG), 포즈 정보를 로드합니다.
    *   `read_npz_depth`, `read_png_depth`: NPZ 및 PNG 형식의 깊이 맵을 읽습니다.
    *   `_get_intrinsics`, `_read_raw_calib_file`: 카메라 캘리브레이션 파일을 파싱하여 내외부 파라미터를 가져옵니다.
    *   `_get_pose`, `_get_oxts_data`: OXTS 데이터를 사용하여 포즈 정보를 계산하고 캐싱합니다.
    *   컨텍스트 프레임을 로드하고 해당 포즈를 계산하는 기능을 포함합니다.

### `packnet_sfm/datasets/kitti_dataset_debug.py`
*   **목적**: KITTI 데이터셋의 파일 구조와 경로를 디버깅하는 유틸리티 스크립트입니다.
*   **기능**:
    *   주어진 데이터 경로와 스플릿 파일의 존재 여부를 확인합니다.
    *   스플릿 파일의 내용을 일부 출력하고, 실제 데이터 디렉토리 구조를 탐색하여 출력합니다.
    *   첫 번째 이미지 경로에 대한 존재 여부 및 가능한 변형 경로를 테스트하여 디버깅에 도움을 줍니다.

### `packnet_sfm/datasets/kitti_dataset_optimized.py`
*   **목적**: 기존 `KITTIDataset`의 최적화된 버전으로, 병렬 처리와 파일 기반 캐싱 시스템을 도입하여 데이터 로딩 속도를 향상시킵니다.
*   **기능**:
    *   `FileCache`: 파일 기반 캐싱 시스템을 구현하여 이전에 스캔된 파일 목록을 저장하고 재사용합니다.
    *   `validate_file_chunk`, `validate_context_chunk`: 멀티프로세싱을 사용하여 파일 존재 여부 및 컨텍스트 유효성을 병렬로 검증합니다.
    *   `_load_paths_optimized`: 캐시를 활용하고, 캐시가 없을 경우 병렬로 파일 경로를 스캔하여 로딩 시간을 단축합니다.
    *   기존 `KITTIDataset`의 모든 기능을 포함하며, 데이터 로딩 단계에서만 최적화가 적용됩니다.

### `packnet_sfm/datasets/kitti_dataset_utils.py`
*   **목적**: KITTI 데이터셋 로딩 및 파싱을 위한 헬퍼 메서드를 제공합니다.
*   **기능**:
    *   `rotx`, `roty`, `rotz`: X, Y, Z축 회전 행렬 생성.
    *   `transform_from_rot_trans`: 회전 행렬과 변환 벡터로부터 변환 행렬 생성.
    *   `read_calib_file`: 캘리브레이션 파일을 읽고 파싱.
    *   `pose_from_oxts_packet`: OXTS 패킷으로부터 SE(3) 포즈 행렬 계산.
    *   `load_oxts_packets_and_poses`: OXTS 지상 진실(ground truth) 데이터를 읽고 포즈를 생성.

### `packnet_sfm/datasets/nc_dataset.py`
*   **목적**: (내용이 제공되지 않았지만) NC 데이터셋을 처리하는 클래스로 추정됩니다.

### `packnet_sfm/datasets/ncdb_dataset.py`
*   **목적**: NCDB-Cls-Sample 데이터셋을 로드하고 처리하는 클래스입니다.
*   **기능**:
    *   주어진 스플릿 파일(`combined_train.json` 등)을 기반으로 데이터 항목을 로드합니다.
    *   하드코딩된 `DEFAULT_CALIB_A6` 및 `DEFAULT_LIDAR_TO_WORLD`를 사용하여 카메라 캘리브레이션 및 LiDAR 변환 정보를 제공합니다.
    *   이미지, 깊이 맵을 로드하고, 선택적으로 이진 마스크를 적용합니다.
    *   VADAS 카메라 모델의 내외부 파라미터 및 왜곡 계수를 샘플에 포함합니다.

### `packnet_sfm/datasets/transforms.py`
*   **목적**: 학습, 검증, 테스트 모드에 따라 적절한 데이터 증강 변환 파이프라인을 제공합니다.
*   **기능**:
    *   `train_transforms`, `validation_transforms`, `test_transforms`: 각 모드에 특화된 변환(크기 조절, 자르기, 색상 지터링, 텐서 변환 등)을 정의합니다.
    *   `get_transforms`: 모드와 설정에 따라 적절한 변환 함수를 반환하며, `augmentations_kitti_compatible.py`의 고급 증강을 선택적으로 활성화할 수 있습니다.

### `packnet_sfm/geometry/__init__.py`
*   **목적**: 기하학 관련 모듈의 루트 패키지 정보를 제공합니다.

### `packnet_sfm/geometry/camera.py`
*   **목적**: 핀홀 카메라 모델에 대한 재구성(reconstruction) 및 투영(projection) 기능을 구현하는 미분 가능한 카메라 클래스를 제공합니다.
*   **기능**:
    *   `Camera`: 표준 핀홀 카메라 모델을 구현하며, 내외부 파라미터를 관리합니다.
    *   `reconstruct`: 깊이 맵으로부터 픽셀별 3D 포인트를 재구성합니다.
    *   `project`: 3D 포인트를 이미지 평면에 투영합니다.
    *   `FisheyeCamera`: VADAS 모델 기반의 어안(fisheye) 카메라 투영 기능을 구현합니다.

### `packnet_sfm/geometry/camera_generic.py`
*   **목적**: 일반적인 카메라 모델에 대한 재구성 및 투영 기능을 구현하는 미분 가능한 카메라 클래스를 제공합니다. 특히, `RaySurfaceResNet`과 같은 모델에서 생성된 `ray_surface`를 사용하여 투영을 수행합니다.
*   **기능**:
    *   `GenericCamera`: `ray_surface`를 기반으로 3D 포인트를 재구성하고 투영합니다.
    *   `project`: 3D 포인트를 이미지 평면에 투영하며, 소프트맥스(softmax) 근사를 사용하고 학습 진행도에 따라 어닐링(annealing)을 적용하여 투영을 패치 기반으로 수행합니다.

### `packnet_sfm/geometry/camera_utils.py`
*   **목적**: 카메라 관련 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `construct_K`: 핀홀 파라미터로부터 카메라 내외부 행렬 `K`를 구성합니다.
    *   `scale_intrinsics`: 내외부 파라미터를 스케일링합니다.
    *   `view_synthesis`, `view_synthesis_generic`: 참조 이미지를 워핑하여 원본 이미지를 재구성합니다. `view_synthesis_generic`은 `GenericCamera`와 함께 사용됩니다.

### `packnet_sfm/geometry/pose.py`
*   **목적**: 4x4 변환 행렬을 캡슐화하는 포즈(Pose) 클래스를 제공합니다.
*   **기능**:
    *   `identity`: 항등 변환 행렬로 초기화합니다.
    *   `from_vec`: 6자유도 벡터(회전 및 변환)로부터 포즈를 초기화합니다.
    *   `inverse`: 포즈의 역변환을 계산합니다.
    *   `transform_pose`: 다른 포즈와 결합하여 새로운 포즈를 생성합니다.
    *   `transform_points`: 3D 포인트를 변환합니다.
    *   `__matmul__`: `@` 연산자를 오버로드하여 포즈 또는 3D 포인트 변환을 지원합니다.

### `packnet_sfm/geometry/pose_utils.py`
*   **목적**: 포즈 관련 헬퍼 메서드를 제공합니다.
*   **기능**:
    *   `euler2mat`: 오일러 각도를 회전 행렬로 변환합니다.
    *   `pose_vec2mat`: 오일러 파라미터(6자유도 벡터)를 변환 행렬로 변환합니다.
    *   `invert_pose`, `invert_pose_numpy`: PyTorch 텐서 또는 NumPy 배열 형식의 포즈를 역변환합니다.

### `packnet_sfm/loggers/__init__.py`
*   **목적**: 로거 클래스들을 내보냅니다.

### `packnet_sfm/loggers/tensorboard_logger.py`
*   **목적**: TensorBoard를 사용하여 학습 진행 상황을 로깅합니다.
*   **기능**:
    *   `log_metrics`: 스칼라 메트릭을 TensorBoard에 기록합니다.
    *   `log_depth`: 입력 이미지, 예측된 깊이 맵, Ground Truth 깊이 맵, 이진 마스크 등을 시각화하여 TensorBoard에 기록합니다.

### `packnet_sfm/loggers/wandb_logger.py`
*   **목적**: Weights & Biases (Wandb)를 사용하여 학습 진행 상황을 로깅합니다.
*   **기능**:
    *   `create_experiment`: Wandb 실험을 생성하고 관리합니다.
    *   `watch`: 모델의 파라미터(예: 그래디언트)를 모니터링합니다.
    *   `log_config`: 모델 설정을 기록합니다.
    *   `log_metrics`: 학습 메트릭을 기록합니다.
    *   `log_images`, `log_depth`: 이미지 및 깊이 맵을 Wandb에 시각화하여 기록합니다.

### `packnet_sfm/losses/__init__.py`
*   **목적**: 손실 함수 관련 모듈의 루트 패키지 정보를 제공합니다.

### `packnet_sfm/losses/generic_multiview_photometric_loss.py`
*   **목적**: 일반적인 멀티뷰 자기지도(self-supervised) 광도 손실(photometric loss)을 구현합니다. `GenericCamera`와 함께 사용됩니다.
*   **기능**:
    *   `SSIM`: 이미지 간의 구조적 유사성(Structural SIMilarity)을 계산합니다.
    *   `warp_ref_image`: 참조 이미지를 워핑하여 원본 이미지를 재구성합니다. `ray_surface`를 사용하여 투영을 수행합니다.
    *   `calc_photometric_loss`: L1 손실과 SSIM 손실을 결합하여 광도 손실을 계산합니다.
    *   `reduce_photometric_loss`: 모든 컨텍스트 이미지의 광도 손실을 결합합니다.
    *   `calc_smoothness_loss`: 역깊이 맵의 부드러움(smoothness) 손실을 계산합니다.
    *   `forward`: 전체 자기지도 광도 손실을 계산합니다.

### `packnet_sfm/losses/loss_base.py`
*   **목적**: 모든 손실 클래스의 기본 클래스를 정의하고, 점진적 스케일링(progressive scaling) 기능을 제공합니다.
*   **기능**:
    *   `ProgressiveScaling`: 학습 진행도에 따라 사용되는 스케일의 수를 동적으로 조절합니다.
    *   `LossBase`: 손실 및 메트릭을 저장하는 기본 구조를 제공합니다.

### `packnet_sfm/losses/multiview_photometric_loss.py`
*   **목적**: 표준 멀티뷰 자기지도 광도 손실을 구현합니다. `Camera` 클래스와 함께 사용됩니다.
*   **기능**:
    *   `SSIM`: 이미지 간의 구조적 유사성(Structural SIMilarity)을 계산합니다.
    *   `warp_ref_image`: 참조 이미지를 워핑하여 원본 이미지를 재구성합니다.
    *   `calc_photometric_loss`: L1 손실과 SSIM 손실을 결합하여 광도 손실을 계산합니다.
    *   `reduce_photometric_loss`: 모든 컨텍스트 이미지의 광도 손실을 결합합니다.
    *   `calc_smoothness_loss`: 역깊이 맵의 부드러움(smoothness) 손실을 계산합니다.
    *   `forward`: 전체 자기지도 광도 손실을 계산합니다.

### `packnet_sfm/losses/ssi_loss.py`
*   **목적**: 스케일-시프트 불변(Scale-Shift-Invariant, SSI) 깊이 손실(로그 깊이 버전)을 구현합니다.
*   **기능**:
    *   예측된 역깊이와 Ground Truth 역깊이 간의 차이를 기반으로 SSI 손실을 계산합니다.

### `packnet_sfm/losses/ssi_loss_enhanced.py`
*   **목적**: L1 정규화와 결합된 향상된 스케일-시프트 불변(SSI) 깊이 손실을 구현합니다.
*   **기능**:
    *   `EnhancedSSILoss`: SSI 손실과 L1 손실을 결합하여 상대적 정확도와 절대적 정확도를 모두 고려합니다. 학습 진행도에 따른 적응형 가중치 부여 기능을 포함합니다.
    *   `ProgressiveEnhancedSSILoss`: 순수 SSI 손실로 시작하여 점진적으로 L1 손실을 추가하는 버전으로, 학습 초기에 상대적 관계 학습에 집중하고 점차 절대 정확도를 향상시킵니다.

### `packnet_sfm/losses/ssi_silog_loss.py`
*   **목적**: 스케일-시프트 불변(SSI) 손실과 Silog 손실을 결합한 손실 함수를 구현합니다.
*   **기능**:
    *   `SSISilogLoss`: SSI 손실(상대적 정확도)과 Silog 손실(로그 스케일 정확도)을 결합하여 스케일-시프트 불변성을 유지하면서 절대 깊이 정확도를 향상시킵니다.

### `packnet_sfm/losses/ssi_trim_loss.py`
*   **목적**: 스케일-시프트 불변 트리밍 L1 손실(MiDaS `L_ssitrim`)을 구현합니다.
*   **기능**:
    *   `_solve_scale_shift`: 예측과 Ground Truth 간의 스케일 및 시프트 파라미터를 최소 제곱법으로 계산합니다.
    *   `forward`: 가장 높은 오차를 가진 픽셀의 일부를 제거(trimming)하여 이상치에 덜 민감하게 손실을 계산합니다.

### `packnet_sfm/losses/supervised_loss.py`
*   **목적**: 역깊이 맵에 대한 지도(supervised) 손실을 구현합니다.
*   **기능**:
    *   `BerHuLoss`: BerHu 손실을 계산합니다.
    *   `SilogLoss`: Silog 손실을 계산합니다.
    *   `get_loss_func`: `l1`, `mse`, `berhu`, `silog`, `abs_rel`, `ssi`, `enhanced-ssi`, `progressive-ssi`, `ssi-trim`, `ssi-silog` 등 다양한 지도 손실 함수를 반환합니다.
    *   `SupervisedLoss`: 선택된 지도 손실 함수를 사용하여 예측된 역깊이와 Ground Truth 역깊이 간의 손실을 계산합니다. 점진적 스케일링을 지원합니다.

### `packnet_sfm/losses/velocity_loss.py`
*   **목적**: 포즈 변환에 대한 속도 손실(velocity loss)을 구현합니다.
*   **기능**:
    *   예측된 포즈와 Ground Truth 포즈 간의 변환 벡터 노름(norm) 차이를 기반으로 속도 손실을 계산합니다.

### `packnet_sfm/models/__init__.py`
*   **목적**: SfM(Structure-from-Motion) 모델 및 래퍼에 대한 개요를 제공합니다.
*   **내용**: `SfmModel`, `SelfSupModel`, `SemiSupModel`, `ModelWrapper`, `ModelCheckpoint` 등 주요 모델 클래스의 역할을 간략하게 설명합니다.

### `packnet_sfm/models/base_model.py`
*   **목적**: PackNet-SfM 모델 래퍼를 위한 API를 정의하는 기본 모델 클래스입니다.
*   **기능**:
    *   로깅 및 손실 딕셔너리를 초기화합니다.
    *   모델이 필요로 하는 네트워크(`_network_requirements`), 학습 시 필요한 Ground Truth 정보(`_train_requirements`), 모델에 제공되는 입력 키(`_input_keys`)를 정의합니다.
    *   네트워크 모듈을 모델에 추가하는 `add_net` 메서드를 제공합니다.

### `packnet_sfm/models/GenericSelfSupModel.py`
*   **목적**: `GenericSfmModel`로부터 깊이 및 포즈 네트워크를 상속받고, 자기지도 학습을 위한 광도 손실을 포함하는 모델입니다.
*   **기능**:
    *   `GenericMultiViewPhotometricLoss`를 사용하여 자기지도 광도 손실을 계산합니다.
    *   `forward`: 배치 데이터를 처리하고, 학습 시 자기지도 손실을 계산하여 반환합니다.

### `packnet_sfm/models/GenericSfmModel.py`
*   **목적**: 포즈 및 깊이 네트워크를 캡슐화하는 일반적인 모델 클래스입니다.
*   **기능**:
    *   `flip_model`: 입력 이미지를 뒤집고 출력 역깊이 맵을 뒤집는 기능을 제공합니다.
    *   `interpolate_scales`: 다양한 해상도의 이미지를 동일한 해상도로 보간합니다.
    *   `compute_depth_net`: 단일 이미지로부터 역깊이 맵을 계산합니다.
    *   `compute_poses`: 이미지와 컨텍스트 이미지 시퀀스로부터 포즈를 계산합니다.
    *   `forward`: 배치 데이터를 처리하고, 예측된 역깊이 맵과 포즈를 반환합니다.

### `packnet_sfm/models/model_checkpoint.py`
*   **목적**: 모델 체크포인트를 저장하고 관리하는 기능을 제공합니다.
*   **기능**:
    *   `_save_model`: 모델 상태를 파일로 저장하고, 선택적으로 S3 버킷과 동기화합니다.
    *   `sync_s3_data`: S3에 데이터를 동기화합니다.
    *   `save_code`: 모델 폴더에 코드 아카이브를 저장합니다.
    *   `check_monitor_top_k`: 모니터링 메트릭을 기반으로 최상위 K개 모델을 추적하고 저장할지 결정합니다.
    *   `format_checkpoint_name`: 에포크 및 메트릭을 포함하는 체크포인트 파일 이름을 형식화합니다.
    *   `check_and_save`: 주기적으로 모델을 저장할지 여부를 확인하고 저장 작업을 수행합니다.

### `packnet_sfm/models/model_utils.py`
*   **목적**: 모델 출력 처리 및 배치 조작을 위한 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `flip`: 텐서 또는 텐서 리스트를 뒤집습니다.
    *   `merge_outputs`: 여러 모델 출력을 병합하여 로깅에 사용합니다.
    *   `stack_batch`: 멀티 카메라 배치를 스택합니다.
    *   `flip_batch_input`: 배치 입력 정보를 뒤집습니다.
    *   `flip_output`: 모델 출력을 뒤집습니다.
    *   `upsample_output`: 다중 스케일 출력을 최고 해상도로 업샘플링합니다.

### `packnet_sfm/models/model_wrapper.py`
*   **목적**: `SfmModel`(포즈+깊이 네트워크)을 감싸는 최상위 PyTorch 모듈 래퍼로, `Trainer` 클래스와 함께 쉽게 학습 및 평가할 수 있도록 설계되었습니다.
*   **기능**:
    *   모델, 옵티마이저, 스케줄러, 데이터셋을 준비하고 관리합니다.
    *   `prepare_model`, `prepare_datasets`: 모델 및 데이터셋을 설정하고 이전 상태를 로드합니다.
    *   `configure_optimizers`: 깊이 및 포즈 네트워크의 옵티마이저와 스케줄러를 구성합니다.
    *   `train_dataloader`, `val_dataloader`, `test_dataloader`: 각 모드에 대한 데이터로더를 준비합니다.
    *   `training_step`, `validation_step`, `test_step`: 학습, 검증, 테스트 배치를 처리합니다.
    *   `training_epoch_end`, `validation_epoch_end`, `test_epoch_end`: 각 에포크 종료 시 손실 및 메트릭을 집계하고 로깅합니다.
    *   `evaluate_depth`: 깊이 메트릭을 계산합니다.
    *   `print_metrics`: 깊이 메트릭을 콘솔에 출력합니다.
    *   `setup_depth_net`, `setup_pose_net`, `setup_model`, `setup_dataset`, `setup_dataloader`: 네트워크, 모델, 데이터셋, 데이터로더를 설정하는 헬퍼 함수.

### `packnet_sfm/models/SelfSupModel.py`
*   **목적**: `SfmModel`로부터 깊이 및 포즈 네트워크를 상속받고, 자기지도 학습을 위한 광도 손실을 포함하는 모델입니다.
*   **기능**:
    *   `MultiViewPhotometricLoss`를 사용하여 자기지도 광도 손실을 계산합니다.
    *   `self_supervised_loss`: 자기지도 광도 손실을 계산합니다.
    *   `forward`: 배치 데이터를 처리하고, 학습 시 자기지도 손실을 계산하여 반환합니다.

### `packnet_sfm/models/SemiSupCompletionModel.py`
*   **목적**: 깊이 예측 및 완성을 위한 반지도(semi-supervised) 모델입니다. `SelfSupModel`을 상속받고 지도 손실을 추가합니다.
*   **기능**:
    *   `SupervisedLoss`를 사용하여 지도 손실을 계산합니다.
    *   `forward`: 자기지도 손실과 지도 손실을 가중치 합산하여 최종 손실을 계산합니다.
    *   특히, `YOLOv8SAN01` 모델과 함께 사용될 경우 RGB-Only 예측과 RGB+D 예측 간의 일관성 손실(consistency loss)을 추가합니다.

### `packnet_sfm/models/SemiSupModel.py`
*   **목적**: 깊이 예측을 위한 반지도 모델입니다. `SelfSupModel`을 상속받고 지도 손실을 추가합니다.
*   **기능**:
    *   `SupervisedLoss`를 사용하여 지도 손실을 계산합니다.
    *   `forward`: 자기지도 손실과 지도 손실을 가중치 합산하여 최종 손실을 계산합니다.

### `packnet_sfm/models/SfmModel.py`
*   **목적**: 포즈 및 깊이 네트워크를 캡슐화하는 모델 클래스입니다. `BaseModel`을 상속받습니다.
*   **기능**:
    *   `add_depth_net`, `add_pose_net`: 깊이 및 포즈 네트워크를 모델에 추가합니다.
    *   `depth_net_flipping`: 깊이 네트워크를 실행할 때 입력 뒤집기 옵션을 제공합니다.
    *   `compute_depth_net`: 단일 이미지로부터 역깊이 맵을 계산합니다.
    *   `compute_pose_net`: 이미지와 컨텍스트 이미지 시퀀스로부터 포즈를 계산합니다.
    *   `forward`: 배치 데이터를 처리하고, 깊이 및 포즈 네트워크의 출력을 반환합니다.

### `packnet_sfm/models/VelSupModel.py`
*   **목적**: 추가적인 속도 지도(velocity supervision) 손실을 포함하는 자기지도 모델입니다.
*   **기능**:
    *   `VelocityLoss`를 사용하여 속도 손실을 계산합니다.
    *   `forward`: 기존 자기지도 손실에 속도 손실을 추가하여 최종 손실을 계산합니다.

### `packnet_sfm/networks/__init__.py`
*   **목적**: 신경망 관련 모듈의 루트 패키지 정보를 제공합니다.

### `packnet_sfm/networks/depth/DepthResNet.py`
*   **목적**: ResNet 아키텍처 기반의 역깊이 네트워크를 구현합니다.
*   **기능**:
    *   `ResnetEncoder`: ResNet 인코더를 사용하여 이미지 특징을 추출합니다.
    *   `DepthDecoder`: 인코더 특징으로부터 깊이 맵을 디코딩합니다.
    *   `disp_to_depth`: 시그모이드 출력(disparity)을 깊이 맵으로 변환합니다.
    *   `forward`: 네트워크를 실행하고 역깊이 맵을 반환합니다.

### `packnet_sfm/networks/depth/PackNet01.py`
*   **목적**: 3D 컨볼루션을 사용하는 PackNet 네트워크(버전 01)를 구현합니다.
*   **기능**:
    *   `PackLayerConv3d`, `UnpackLayerConv3d`: 3D 컨볼루션을 포함하는 패킹/언패킹 레이어.
    *   `Conv2D`, `ResidualBlock`, `InvDepth`: 기본 컨볼루션, 잔차 블록, 역깊이 예측 레이어.
    *   `forward`: 네트워크를 실행하고 역깊이 맵을 반환합니다.

### `packnet_sfm/networks/depth/PackNetSAN01.py`
*   **목적**: SAN(Sparse Auxiliary Network) 기능을 포함하는 PackNet 네트워크를 구현합니다.
*   **기능**:
    *   `Encoder`, `Decoder`: PackNet의 인코더 및 디코더 구조.
    *   `MinkowskiEncoder`: LiDAR 처리를 위한 Minkowski 엔진 기반 인코더.
    *   `weight`, `bias`: LiDAR 특징과 RGB 특징을 융합하기 위한 학습 가능한 가중치 및 편향.
    *   `run_network`: 네트워크를 실행하고, `input_depth`가 제공되면 LiDAR 특징을 융합하여 역깊이 맵을 생성합니다.
    *   `forward`: 학습 시 RGB-only 및 RGB+LiDAR 예측을 모두 수행하고, 특징 일관성 손실을 계산합니다.

### `packnet_sfm/networks/depth/PackNetSlim01.py`
*   **목적**: 더 적은 특징 채널을 사용하는 PackNet 네트워크의 슬림 버전입니다.
*   **기능**:
    *   `PackNet01`과 유사한 구조를 가지지만, 채널 수가 줄어들어 모델 크기가 작습니다.

### `packnet_sfm/networks/depth/PackNetSlimSAN01.py`
*   **목적**: SAN 기능을 포함하는 PackNet Slim 네트워크입니다.
*   **기능**:
    *   `PackNetSlim01`의 슬림한 구조에 SAN 기능을 추가합니다.
    *   `MinkowskiEncoder`를 사용하여 LiDAR 처리를 수행하며, `use_film` 및 `film_scales` 파라미터를 통해 Depth-aware FiLM 변조를 선택적으로 적용할 수 있습니다.
    *   학습 가능한 융합 가중치(`weight`, `bias`)를 사용하여 LiDAR 특징과 RGB 특징을 융합합니다.
    *   `forward`: 학습 시 RGB-only 및 RGB+LiDAR 예측을 모두 수행하고, 특징 일관성 손실을 계산합니다.

### `packnet_sfm/networks/depth/RaySurfaceResNet.py`
*   **목적**: ResNet 아키텍처 기반의 레이 표면(ray surface) 네트워크를 구현합니다. 깊이 맵과 함께 레이 표면을 디코딩합니다.
*   **기능**:
    *   `ResnetEncoder`: ResNet 인코더를 사용하여 이미지 특징을 추출합니다.
    *   `DepthDecoder`: 깊이 맵을 디코딩합니다.
    *   `RaySurfaceDecoder`: 레이 표면을 디코딩합니다.
    *   `forward`: 네트워크를 실행하고 역깊이 맵과 레이 표면을 반환합니다.

### `packnet_sfm/networks/depth/ResNetSAN01.py`
*   **목적**: 향상된 LiDAR 특징 추출 기능을 포함하는 ResNet 기반 SAN 네트워크입니다.
*   **기능**:
    *   `ResnetEncoder`: ResNet 인코더를 사용합니다.
    *   `DepthDecoder`: 깊이 디코더.
    *   `EnhancedMinkowskiEncoder`: 향상된 LiDAR 처리를 위한 Minkowski 엔진 기반 인코더.
    *   `use_film`, `film_scales`: Depth-aware FiLM 변조를 선택적으로 적용합니다.
    *   `use_enhanced_lidar`: 향상된 LiDAR 처리를 활성화합니다.
    *   `feature_refinement`: 특징 정제 레이어를 포함합니다.
    *   `weight`, `bias`: 학습 가능한 융합 가중치 및 편향.
    *   `run_network`: 네트워크를 실행하고, `input_depth`가 제공되면 LiDAR 특징을 융합하여 역깊이 맵을 생성합니다.
    *   `forward`: 학습 시 RGB-only 및 RGB+LiDAR 예측을 모두 수행하고, 특징 일관성 손실을 계산합니다.

### `packnet_sfm/networks/depth/YOLOv8SAN01.py`
*   **목적**: YOLOv8 기반 SAN 네트워크로, Neck 특징 및 ImageNet 사전 학습 지원을 포함합니다.
*   **기능**:
    *   `YOLOv8Neck`: YOLOv8의 Neck(특징 피라미드) 부분을 구현하여 다중 스케일 특징을 처리합니다.
    *   `DepthNeck`: 간단한 FPN(Feature Pyramid Network) 스타일의 Neck을 구현하여 깊이 추정 특징을 융합합니다.
    *   `YOLO`: `ultralytics` 라이브러리에서 YOLOv8 모델을 로드합니다.
    *   `backbone`: YOLOv8 백본을 사용하여 특징을 추출합니다.
    *   `feature_adapters`: YOLOv8 특징을 ResNet 호환 채널 구조로 변환합니다.
    *   `DepthDecoder`: 깊이 디코더.
    *   `MinkowskiEncoder`: LiDAR 처리를 위한 Minkowski 엔진 기반 인코더.
    *   `use_film`, `film_scales`: Depth-aware FiLM 변조를 선택적으로 적용합니다.
    *   `weight`, `bias`: 학습 가능한 융합 가중치 및 편향.
    *   `extract_features`: YOLOv8 백본에서 ResNet 디코더 호환 특징을 추출합니다.
    *   `run_network`: 네트워크를 실행하고, `input_depth`가 제공되면 LiDAR 특징을 융합하여 역깊이 맵을 생성합니다.
    *   `forward`: 학습 시 RGB-only 및 RGB+LiDAR 예측을 모두 수행하고, 특징 일관성 손실을 계산합니다.

### `packnet_sfm/networks/layers/enhanced_minkowski_encoder.py`
*   **목적**: 향상된 LiDAR 특징 인코더로, 개선된 처리 및 오류 처리를 포함합니다.
*   **기능**:
    *   `EnhancedMinkConv2D`: 멀티 스케일 처리 및 채널 어텐션 메커니즘을 포함하는 향상된 Minkowski 컨볼루션 블록.
    *   `GeometryAwareLiDARProcessor`: 기하학적 인식을 통해 희소 깊이를 처리하는 간소화된 LiDAR 특징 프로세서.
    *   `EnhancedMinkowskiEncoder`: `GeometryAwareLiDARProcessor` 및 `EnhancedMinkConv2D`를 사용하여 LiDAR 특징을 인코딩합니다.
    *   FiLM 생성기를 포함하여 RGB 채널에 대한 Depth-aware FiLM 파라미터를 생성합니다.
    *   `prep`: 희소 깊이 데이터를 처리 준비합니다.
    *   `forward`: 향상된 LiDAR 특징 인코딩을 수행하고, 선택적으로 FiLM 파라미터를 반환합니다.

### `packnet_sfm/networks/layers/minkowski.py`
*   **목적**: MinkowskiEngine을 사용하여 희소(sparse) 텐서를 처리하는 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `sparsify_features`, `sparsify_depth`: 밀집(dense) 특징 맵 또는 깊이 맵을 희소 텐서로 변환합니다.
    *   `densify_features`: 희소 텐서를 밀집 특징 맵으로 변환합니다.
    *   `densify_add_features_unc`: 불확실성을 고려하여 희소 특징을 밀집 특징에 추가하고 밀집화합니다.
    *   `map_add_features`: 밀집 특징을 희소 텐서에 매핑하고 추가합니다.

### `packnet_sfm/networks/layers/minkowski_encoder.py`
*   **목적**: 깊이 완성(depth completion)을 위한 Minkowski 인코더와 선택적인 Depth-aware FiLM 기능을 구현합니다.
*   **기능**:
    *   `MinkConv2D`: Minkowski 컨볼루션 블록.
    *   `MinkowskiEncoder`: 희소 깊이 특징을 처리하는 인코더.
    *   `rgb_channels`, `film_generators`: Depth-aware FiLM을 위한 RGB 채널 및 FiLM 파라미터 생성기.
    *   `prep`: 희소 깊이 데이터를 처리 준비합니다.
    *   `forward`: 희소 특징을 처리하고, 선택적으로 FiLM 파라미터를 반환합니다.

### `packnet_sfm/networks/layers/packnet/layers01.py`
*   **목적**: PackNet 아키텍처에서 사용되는 기본 레이어(2D/3D 컨볼루션, 잔차 블록, 역깊이 레이어, 패킹/언패킹)를 정의합니다.
*   **기능**:
    *   `Conv2D`: GroupNorm 및 ELU를 포함하는 2D 컨볼루션.
    *   `ResidualConv`, `ResidualBlock`: 2D 컨볼루션 잔차 블록.
    *   `InvDepth`: 역깊이 예측 레이어.
    *   `packing`: 텐서의 공간 픽셀을 채널로 재구성하는 패킹 함수.
    *   `PackLayerConv2d`, `UnpackLayerConv2d`: 2D 컨볼루션을 포함하는 패킹/언패킹 레이어.
    *   `PackLayerConv3d`, `UnpackLayerConv3d`: 3D 컨볼루션을 포함하는 패킹/언패킹 레이어.

### `packnet_sfm/networks/layers/resnet/depth_decoder.py`
*   **목적**: ResNet 기반 깊이 디코더를 구현합니다.
*   **기능**:
    *   `ConvBlock`, `Conv3x3`, `upsample`: 기본 컨볼루션 블록, 3x3 컨볼루션, 업샘플링 함수.
    *   `DepthDecoder`: 인코더 특징으로부터 다중 스케일 깊이 맵을 디코딩합니다. 스킵 연결(skip connections)을 사용합니다.

### `packnet_sfm/networks/layers/resnet/layers.py`
*   **목적**: ResNet 기반 네트워크에서 사용되는 기본 레이어 및 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `disp_to_depth`: 네트워크의 시그모이드 출력(disparity)을 실제 깊이 예측으로 변환합니다.
    *   `ConvBlock`: 컨볼루션과 ELU(또는 ReLU) 활성화 함수를 포함하는 블록.
    *   `Conv3x3`: 패딩과 3x3 컨볼루션을 수행하는 레이어.
    *   `upsample`: 텐서를 2배 업샘플링합니다.

### `packnet_sfm/networks/layers/resnet/pose_decoder.py`
*   **목적**: ResNet 기반 포즈 디코더를 구현합니다.
*   **기능**:
    *   인코더 특징으로부터 카메라의 회전(axisangle) 및 변환(translation)을 예측합니다.

### `packnet_sfm/networks/layers/resnet/raysurface_decoder.py`
*   **목적**: ResNet 기반 레이 표면(ray surface) 디코더를 구현합니다.
*   **기능**:
    *   `RaySurfaceDecoder`: 인코더 특징으로부터 레이 표면을 디코딩합니다. `tanh` 활성화 함수를 사용하여 출력을 -1과 1 사이로 제한합니다.

### `packnet_sfm/networks/layers/resnet/resnet_encoder.py`
*   **목적**: ResNet 인코더를 구현합니다.
*   **기능**:
    *   `ResNetMultiImageInput`: 여러 입력 이미지를 처리할 수 있는 ResNet 모델을 구성합니다.
    *   `resnet_multiimage_input`: ImageNet 사전 학습 여부와 입력 이미지 수에 따라 ResNet 모델을 생성합니다.
    *   `ResnetEncoder`: ResNet 모델을 사용하여 입력 이미지로부터 다중 스케일 특징을 추출합니다.

### `packnet_sfm/networks/layers/yolov8/yolov8_backbone.py`
*   **목적**: 깊이 추정을 위한 YOLOv8 백본을 구현합니다.
*   **기능**:
    *   `autopad`: 자동 패딩 계산.
    *   `Conv`, `DWConv`, `DWConvTranspose2d`: 표준 컨볼루션, 깊이별 컨볼루션, 깊이별 전치 컨볼루션.
    *   `Bottleneck`, `C2f`, `SPPF`: YOLOv8의 핵심 빌딩 블록.
    *   `YOLOv8Backbone`: YOLOv8의 백본 구조를 정의하고, 선택적으로 `ultralytics`에서 사전 학습된 가중치를 로드합니다.
    *   `forward`: 다중 스케일 특징을 반환합니다.

### `packnet_sfm/networks/layers/yolov8/yolov8_depth_decoder.py`
*   **목적**: YOLOv8 특징을 기반으로 깊이 맵을 디코딩하는 안정성 개선된 디코더를 구현합니다.
*   **기능**:
    *   `SimpleDepthHead`: 간단한 깊이 예측 헤드.
    *   `YOLOv8DepthDecoder`: 인코더 특징으로부터 다중 스케일 깊이 맵을 디코딩합니다.
    *   `feature_convs`: 특징 변환 레이어.
    *   `fusion_convs`: 특징 융합 레이어.
    *   `depth_heads`: 각 스케일별 깊이 예측 헤드.
    *   `forward`: 입력 특징을 처리하고, 다중 스케일 깊이 맵을 출력합니다. NaN/Inf 값에 대한 안정성 체크를 포함합니다.

### `packnet_sfm/networks/pose/PoseNet.py`
*   **목적**: 포즈 네트워크를 구현합니다.
*   **기능**:
    *   `conv_gn`: GroupNorm을 포함하는 컨볼루션 블록.
    *   `PoseNet`: 입력 이미지와 컨텍스트 이미지로부터 카메라 포즈(6자유도 벡터)를 예측합니다.

### `packnet_sfm/networks/pose/PoseResNet.py`
*   **목적**: ResNet 아키텍처 기반의 포즈 네트워크를 구현합니다.
*   **기능**:
    *   `ResnetEncoder`: ResNet 인코더를 사용하여 이미지 특징을 추출합니다.
    *   `PoseDecoder`: 인코더 특징으로부터 카메라 포즈(회전 및 변환)를 디코딩합니다.
    *   `forward`: 타겟 이미지와 참조 이미지로부터 포즈를 예측합니다.

### `packnet_sfm/trainers/__init__.py`
*   **목적**: 트레이너 클래스들을 내보냅니다.

### `packnet_sfm/trainers/base_trainer.py`
*   **목적**: 트레이너 클래스의 기본 기능을 정의합니다.
*   **기능**:
    *   `sample_to_cuda`: 배치 데이터를 GPU로 이동합니다.
    *   `BaseTrainer`: 최소/최대 에포크, 초기 검증, 체크포인트 관리 등 기본 학습 설정을 관리합니다.
    *   `proc_rank`, `world_size`, `is_rank_0`: 분산 학습 환경에서 프로세스 순위 및 월드 크기를 제공합니다.
    *   `check_and_save`: 체크포인트 저장 여부를 확인하고 저장합니다.
    *   `train_progress_bar`, `val_progress_bar`, `test_progress_bar`: 학습/검증/테스트 진행률 바를 관리합니다.

### `packnet_sfm/trainers/horovod_trainer.py`
*   **목적**: Horovod를 사용하여 분산 학습을 지원하는 트레이너 클래스입니다. `BaseTrainer`를 상속받습니다.
*   **기능**:
    *   Horovod를 초기화하고 GPU 장치를 설정합니다.
    *   `fit`: 모델 학습 루프를 시작하고, 옵티마이저 및 스케줄러를 구성합니다.
    *   `train_with_eval`: 학습 중 주기적으로 중간 평가를 수행합니다.
    *   `_quick_eval`: 효율적인 중간 평가를 수행하여 학습 진행 상황을 모니터링합니다.
    *   `validate`, `test`: 검증 및 테스트 루프를 실행합니다.
    *   `_save_eval_results`: 중간 평가 결과를 JSON 파일로 저장합니다.

### `packnet_sfm/utils/__init__.py`
*   **목적**: 유틸리티 함수 관련 모듈의 루트 패키지 정보를 제공합니다.

### `packnet_sfm/utils/config.py`
*   **목적**: 모델 학습 및 테스트를 위한 설정을 파싱하고 관리합니다.
*   **기능**:
    *   `prep_dataset`: 데이터셋 설정을 확장하여 스플릿 길이에 맞춥니다.
    *   `set_name`: 실행 이름을 설정합니다.
    *   `set_checkpoint`: 체크포인트 저장 경로 및 모니터링 설정을 구성합니다.
    *   `get_default_config`, `merge_cfg_file`, `merge_cfgs`: 설정 파일을 로드하고 병합합니다.
    *   `backwards_config`: 이전 버전과의 호환성을 위해 설정을 업데이트합니다.
    *   `parse_train_file`, `parse_train_config`, `prepare_train_config`: 학습을 위한 설정 파일을 파싱하고 준비합니다.
    *   `parse_test_file`, `parse_test_config`, `prepare_test_config`: 테스트를 위한 설정 파일을 파싱하고 준비합니다.

### `packnet_sfm/utils/depth.py`
*   **목적**: 깊이 맵 처리 및 평가를 위한 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `load_depth`, `write_depth`: 깊이 맵을 파일에서 로드하고 파일로 저장합니다.
    *   `viz_inv_depth`: 역깊이 맵을 시각화 가능한 컬러맵으로 변환합니다.
    *   `inv2depth`, `depth2inv`: 역깊이와 깊이 맵 간을 변환합니다.
    *   `inv_depths_normalize`: 역깊이 맵을 정규화합니다.
    *   `calc_smoothness`: 역깊이 맵의 부드러움(smoothness)을 계산합니다.
    *   `fuse_inv_depth`, `post_process_inv_depth`: 역깊이 맵과 뒤집힌 역깊이 맵을 융합하여 후처리합니다.
    *   `compute_depth_metrics`: 예측된 깊이 맵과 Ground Truth 깊이 맵으로부터 깊이 메트릭(abs_rel, rmse 등)을 계산합니다.
    *   `scale_depth`: 깊이 맵을 Ground Truth 해상도에 맞게 스케일링합니다.

### `packnet_sfm/utils/horovod.py`
*   **목적**: Horovod 분산 학습 프레임워크를 위한 헬퍼 함수를 제공합니다.
*   **기능**:
    *   `hvd_init`: Horovod를 초기화합니다.
    *   `on_rank_0`: 랭크 0 프로세스에서만 함수를 실행하도록 하는 데코레이터.
    *   `rank`, `world_size`: 현재 프로세스의 랭크 및 전체 월드 크기를 반환합니다.
    *   `print0`: 랭크 0 프로세스에서만 출력합니다.
    *   `reduce_value`: 모든 GPU에서 텐서의 평균값을 줄입니다.

### `packnet_sfm/utils/image.py`
*   **목적**: 이미지 처리 및 조작을 위한 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `load_image`, `write_image`: 이미지를 로드하고 저장합니다.
    *   `flip_lr`: 이미지를 수평으로 뒤집습니다.
    *   `flip_model`: 모델을 실행할 때 입력 이미지를 뒤집고 출력을 다시 뒤집습니다.
    *   `gradient_x`, `gradient_y`: 이미지의 X 및 Y 방향 그래디언트를 계산합니다.
    *   `interpolate_image`, `interpolate_scales`, `match_scales`: 이미지 해상도를 보간하거나 일치시킵니다.
    *   `meshgrid`, `image_grid`: 특정 해상도의 메시그리드 또는 이미지 그리드를 생성합니다.

### `packnet_sfm/utils/load.py`
*   **목적**: 클래스 로딩, 네트워크 가중치 로딩, 디버그 설정 등을 위한 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `set_debug`: 디버그 터미널 로깅을 활성화/비활성화합니다.
    *   `filter_args`, `filter_args_create`: 함수의 인수에 해당하는 딕셔너리 키를 필터링합니다.
    *   `load_class`, `load_class_args_create`: 파일 경로에서 클래스를 로드하고 인스턴스를 생성합니다.
    *   `load_network`: 사전 학습된 네트워크 가중치를 로드합니다.
    *   `backwards_state_dict`: 이전 모델의 상태 딕셔너리를 역호환성을 위해 수정합니다.

### `packnet_sfm/utils/logging.py`
*   **목적**: 터미널 출력 및 로깅을 위한 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `pcolor`: 문자열에 색상을 입힙니다.
    *   `prepare_dataset_prefix`: 메트릭 로깅을 위한 데이터셋 접두사를 준비합니다.
    *   `s3_url`: S3 버킷 URL을 생성합니다.
    *   `print_config`: 모델 설정을 보기 좋게 출력합니다.
    *   `AvgMeter`: 평균값을 계산하고 추적하는 클래스.

### `packnet_sfm/utils/misc.py`
*   **목적**: 다양한 일반 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `filter_dict`: 딕셔너리에서 특정 키워드만 필터링합니다.
    *   `make_list`: 변수를 리스트로 래핑하고, 필요에 따라 반복합니다.
    *   `same_shape`: 두 형태(shape)가 동일한지 확인합니다.
    *   `parse_crop_borders`: 자르기(cropping)를 위한 테두리 값을 파싱합니다.

### `packnet_sfm/utils/reduce.py`
*   **목적**: 분산 학습 환경에서 메트릭을 집계하고 줄이는 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `reduce_dict`: 모든 GPU에서 딕셔너리의 평균값을 줄입니다.
    *   `all_reduce_metrics`: Horovod를 사용하여 모든 배치 및 데이터셋에 대한 메트릭을 줄입니다.
    *   `collate_metrics`: 에포크 출력을 집계하여 평균 메트릭을 생성합니다.
    *   `create_dict`: 집계된 메트릭으로부터 딕셔너리를 생성합니다.
    *   `average_key`, `average_sub_key`, `average_loss_and_metrics`: 손실 및 메트릭 값을 평균화합니다.

### `packnet_sfm/utils/save.py`
*   **목적**: 깊이 예측 결과를 다양한 형식으로 저장하는 유틸리티 함수를 제공합니다.
*   **기능**:
    *   `save_depth`: 예측된 깊이 맵을 NPZ, PNG, RGB 이미지, 시각화된 역깊이 이미지 등으로 저장합니다.

### `packnet_sfm/utils/types.py`
*   **목적**: 다양한 데이터 타입(NumPy 배열, PyTorch 텐서, 튜플, 리스트, 딕셔너리, 문자열, 정수, YACS CfgNode)을 확인하는 헬퍼 함수를 제공합니다.