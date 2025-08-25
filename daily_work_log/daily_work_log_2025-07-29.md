# PM 업무 기록 (2025년 7월 29일)

## 최종 목표
원본 NCDB 데이터셋을 사용하여 30 에포크(epoch) 동안 모델을 학습시키고, 그 결과를 분석하여 전문적인 PM 보고서를 작성한다.

## 오늘 한 일 (2025년 7月 29日)
- `pm_work_log_2025-07-28.md` 업데이트 및 TensorBoard 이미지 로깅 문제 해결 (무작위 이미지 로깅).
- `outputs/resnetsan01-quantized_Floatfunc/tensorboard_logs/` 폴더 정리 (가장 큰 로그 파일만 남김).
- 100개 샘플 데이터셋으로 모델 학습 1회 완료.
- `packnet_sfm/models/model_wrapper.py` 파일의 `validation_step` 및 `validation_epoch_end` 메서드를 원본 코드로 되돌려 TensorBoard 이미지 로깅 관련 오류 해결 시도.
- **TensorBoard 로깅 개선 완료:**
    - `packnet_sfm/loggers/tensorboard_logger.py`의 `log_depth` 함수에 `image_idx` 매개변수 추가.
    - `packnet_sfm/models/model_wrapper.py`의 `training_step`에서 학습 이미지 로깅 (100 스텝마다 첫 번째 이미지).
    - `packnet_sfm/models/model_wrapper.py`의 `validation_step`에서 검증 이미지 로깅 (첫 번째 이미지).
    - `packnet_sfm/models/model_wrapper.py`의 `training_epoch_end`에서 학습 손실/메트릭에 `train/` 접두사 추가.
    - `packnet_sfm/models/model_wrapper.py`의 `validation_epoch_end`에서 검증 손실/메트릭에 `val/` 접두사 추가.

## 앞으로 할 일
1.  **어안 렌즈 왜곡 모델 학습 반영:**
    *   **문제 정의:** 현재 학습 파이프라인(ResNet-SAN, Semi-supervised)은 핀홀 카메라 모델을 가정하며, NCDB 데이터셋의 어안 렌즈 왜곡 계수를 반영하지 못하고 있음. 현재 `supervised_loss_weight: 1.0` 설정으로 자기 지도 학습은 비활성화되어 있으나, GT 깊이 맵이 어안 렌즈 특성을 반영하여 생성되었을 경우 모델이 이를 암묵적으로 학습하도록 돕고, 향후 3D 재구성 및 자기 지도 학습 활성화를 위한 기반을 마련해야 함.
    *   **목표:** 어안 렌즈의 왜곡 계수를 데이터셋 로딩 및 카메라 모델에 통합하여 정확한 깊이 추정 및 3D 재구성을 가능하게 함.
    *   **세부 계획:**
        *   **`packnet_sfm/geometry/camera.py`에 `FisheyeCamera` 클래스 추가:** `VADASFisheyeCameraModel`의 투영 로직을 기반으로 PyTorch 텐서 연산을 사용하여 어안 렌즈 왜곡을 처리할 수 있는 `FisheyeCamera` 클래스를 구현 (주로 투영 메서드에 집중). 역투영 메서드는 현재 지도 학습에서는 필요하지 않으므로 나중에 구현.

2.  **원본 데이터셋 깊이 맵 생성:**
    *   `scripts/create_depth_maps.py`를 사용하여 원본 NCDB 데이터셋에 대한 깊이 맵 생성 완료.
3.  **모델 학습 실행:**
    *   샘플 데이터셋으로 모델 학습을 다시 시작하여 변경 사항 동작 확인.
4.  **결과 분석 및 보고서 작성:**
    *   학습 완료 후 TensorBoard 로그를 기반으로 정량적/정성적 분석을 포함한 최종 보고서 작성.
