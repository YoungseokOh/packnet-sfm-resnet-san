### `SemiSupCompletionModel`의 `supervised_loss` 해상도 불일치 해결

**문제**: `SemiSupCompletionModel.forward`에서 `supervised_loss` 계산 시 예측된 깊이 맵(`inv_depths`)과 GT 깊이 맵(`batch['depth']`)의 해상도가 일치하지 않아 오류 발생.

**원인 분석**:
1.  `configs/train_resnet_san_ncdb_self_sup+supervised.yaml`의 `datasets.augmentation.image_shape` 설정으로 인해 `NcdbDataset`에서 로드되는 이미지가 `192x640`으로 크기 조정됨.
2.  `depth_net`은 `192x640` 크기의 이미지를 입력으로 받아 `192x640` 해상도의 `inv_depths`를 출력.
3.  `batch['depth']`는 `NcdbDataset`에서 로드될 때 원본 이미지 크기(`1920x1536`)를 가짐.
4.  `supervised_loss`는 두 입력의 해상도가 일치해야 하므로 불일치로 인해 오류 발생.

**해결**:
1.  `packnet_sfm/models/SemiSupCompletionModel.py`의 `forward` 메서드 내에서 `supervised_loss`를 호출하기 전에 `self_sup_output['inv_depths']`를 `batch['depth']`의 해상도에 맞게 `F.interpolate`를 사용하여 업샘플링하도록 수정.

**검증**:
*   수정된 스크립트로 학습을 다시 실행하여 정상 작동 확인 필요.

### `scripts/verify_vadas_lookup_table.py` 기능 개선

**요청 사항**: 기존 LuT(Lookup Table) 검증 스크립트는 데이터 분포만 시각화하여 직관적이지 않음. 실제 어안 이미지에 LuT를 적용하여 시각적으로 검증할 수 있도록 기능 개선 요청.

**개선 내용**:
1.  **`--image_path` CLI 인자 추가**: 사용자가 검증에 사용할 어안 이미지 경로를 직접 지정할 수 있도록 기능 추가.
2.  **이미지 기반 시각화 기능 추가**:
    *   **LuT 오버레이 (Overlay)**: 입력 이미지 위에 Angle/Theta LuT를 반투명한 히트맵 형태로 겹쳐서 시각화. 이미지 위치별 각도 값을 직관적으로 파악 가능.
    *   **LuT 등고선 (Contour)**: 입력 이미지 위에 Angle/Theta 값의 등고선을 그려 왜곡 모델의 정확성을 시각적으로 검증. Angle은 동심원, Theta는 방사형으로 나타남.
3.  **의존성 처리**: 이미지 처리에 필요한 `Pillow` 라이브러리가 없을 경우, 사용자에게 설치 방법을 안내하는 메시지 출력 기능 추가.

**기대 효과**:
*   LuT 데이터의 정확성을 실제 이미지와 함께 직관적으로 검증 가능.
*   어안 렌즈 왜곡 모델이 올바르게 적용되었는지 시각적으로 빠르게 확인할 수 있어 디버깅 효율성 증대.