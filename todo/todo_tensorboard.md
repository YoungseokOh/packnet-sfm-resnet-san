# TensorBoard 시각화 개선 계획 (v2)

**To. 개발자**

이 문서는 TensorBoard 시각화를 개선하기 위한 구체적인 요구사항을 다시 정의합니다. 아래 내용을 바탕으로 코드를 수정하고, 완료 후 검토를 요청해 주시기 바랍니다.

---

### **요구사항 1: 특정 메트릭 자동 로깅 비활성화**

**목표**: `validation_epoch_end`에서 모든 메트릭이 TensorBoard에 자동으로 기록되면서 발생하는 `synced_data-mapping_data-*` 태그 문제를 해결합니다.

**수행할 작업**:
1.  `packnet_sfm/models/model_wrapper.py`의 `validation_epoch_end` 함수를 찾습니다.
2.  현재 `logger.log_metrics` 함수는 `metrics_dict`의 모든 내용을 기록하고 있습니다. 이로 인해 원치 않는 `synced_data-mapping_data-*` 메트릭이 자동으로 추가됩니다.
3.  `metrics_dict`에서 `avg_val/` 로 시작하는 핵심 검증 손실 및 정확도 메트릭만 필터링하여 `log_metrics`에 전달하도록 코드를 수정합니다.
    *   **예시**: `avg_val/loss`, `avg_val/abs_rel`, `avg_val/sqr_rel` 등 필요한 메트릭만 선택적으로 로깅합니다.
    *   이를 통해 `synced_data-mapping_data-*` 와 같은 불필요한 메트릭이 TensorBoard에 기록되는 것을 원천적으로 차단합니다.

---

### **요구사항 2: 예측 깊이 맵(Predicted Depth) 시각화 추가**

**목표**: 검증(validation) 단계에서 모델이 예측한 깊이 맵을 시각화하여 TensorBoard의 **Images** 탭에 추가합니다.

**수행할 작업**:
1.  `packnet_sfm/models/model_wrapper.py`의 `validation_step` 함수를 수정합니다.
2.  `packnet_sfm.utils.depth` 모듈에서 `viz_inv_depth` 함수를 `import` 합니다.
3.  `validation_step` 함수 내에서 `evaluate_depth`로부터 반환된 예측 역 깊이(`output['inv_depth']`)를 가져옵니다.
4.  `viz_inv_depth` 함수를 사용하여 이 역 깊이 맵을 시각적으로 표현 가능한 컬러 이미지로 변환합니다.
5.  `packnet_sfm/loggers/tensorboard_logger.py`의 `log_depth` 함수를 수정하여, 변환된 예측 깊이 이미지를 `val/Predicted Depth` 태그로 TensorBoard에 기록하도록 로직을 추가합니다.
    *   기존 `log_depth` 함수는 `output['depth']`를 사용하려고 하지만, 실제 `validation_step`에서는 `output['inv_depth']`가 반환되므로 이 부분을 수정해야 합니다.

---
**From. PM**
_위 요구사항에 따라 명확한 결과물을 기대합니다. 코드 수정 후 결과를 보고해 주세요._
