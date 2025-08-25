# ncdb-cls-sample 데이터셋 학습 파이프라인 구현 계획 (v4)

**To. 개발자**

이 문서는 `instructions_ncdb_cls.txt` 명세서를 기반으로, `ncdb-cls-sample` 데이터셋을 위한 학습 파이프라인을 구축하는 구체적인 개발 단계를 정의합니다. 각 단계를 순서대로 완료한 후, "PM 검토 요청" 섹션에 따라 검토를 요청해 주시기 바랍니다.

**PM 업데이트**: 데이터셋 확장 및 TensorBoard 통합을 위한 새로운 단계들이 추가되었습니다. 이에 따라 기존 단계들의 번호가 조정됩니다.

---

### **Step 1: `NcdbDataset` PyTorch Dataset 클래스 생성**

**상태**: **완료 및 승인됨.**
- **결과물**: `packnet_sfm/datasets/ncdb_dataset.py`

---

### **Step 2: `ModelWrapper.py`에 `NcdbDataset` 로딩 로직 추가**

**상태**: **완료 및 승인됨.**
- **결과물**: `packnet_sfm/models/model_wrapper.py` 내 `setup_dataset` 함수 수정 완료.

---

### **Step 3: 데이터 변환(Transform) 파이프라인 정의 및 적용**

**상태**: **완료 및 승인됨.**
- **결과물**: `packnet_sfm/datasets/ncdb_dataset.py` 수정 완료.

---

### **Step 4: `ncdb-cls-sample` 데이터셋 전용 학습 설정 파일 생성**

**상태**: **완료 및 승인됨.**
- **결과물**: `configs/train_resnet_san_ncdb.yaml`

---

### **Step 5: 학습 실행 및 최종 검증 (초기 10개 샘플)**

**상태**: **완료 및 승인됨.** (초기 10개 샘플로 학습 성공 확인)

---

### **Step 6: 데이터셋 확장 및 `mapping_data.json` 재생성 (100개 샘플)**

**목표**: 실제 데이터셋 경로(`/data/datasets/ncdb-cls/synced_data/`)에서 100개의 랜덤 샘플을 선택하여 새로운 `mapping_data.json` 파일을 생성하고, 이에 해당하는 깊이 맵이 존재하는지 확인합니다. (필요시 생성)

**수행할 작업**:

1.  **원본 데이터셋 파일 목록 확보**: `/data/datasets/ncdb-cls/synced_data/image_a6/` 디렉토리에서 모든 `.png` 파일의 파일명(확장자 제외) 목록을 가져옵니다.
2.  **랜덤 샘플 선택**: 확보된 파일명 목록에서 100개의 고유한 파일명을 무작위로 선택합니다.
3.  **새 `mapping_data.json` 생성**: 선택된 100개의 파일명을 포함하는 새로운 `mapping_data.json` 파일을 `/workspace/packnet-sfm/ncdb-cls-sample/synced_data/` 경로에 생성합니다. 파일 형식은 기존 `mapping_data.json`과 동일하게 `{"pcd": [...], "image_a6": [...]}` 형태를 유지하되, `pcd` 경로는 더 이상 사용되지 않으므로 `image_a6`와 동일한 파일명으로 더미 경로를 채워도 무방합니다.
4.  **깊이 맵 존재 확인 (필요시 생성)**:
    -   선택된 100개 샘플에 해당하는 `depth_maps/*.png` 파일이 `/data/datasets/ncdb-cls/synced_data/depth_maps/`에 이미 존재하는지 확인합니다.
    -   만약 존재하지 않는다면, `scripts/create_depth_maps.py` 스크립트를 사용하여 해당 깊이 맵을 생성해야 합니다. 이때, `create_depth_maps.py`의 `--parent` 인자를 `/data/datasets/ncdb-cls/`로, `--output-dir` 인자를 `/data/datasets/ncdb-cls/synced_data/depth_maps/`로 설정하여 실행합니다.

**PM 검토 요청**: 새로 생성된 `mapping_data.json` 파일의 내용과, 100개 샘플에 대한 깊이 맵이 성공적으로 준비되었음을 보고합니다.

---

### **Step 7: TensorBoard 통합 및 학습 재실행**

**목표**: 학습 진행 상황을 TensorBoard로 시각화하고, 확장된 데이터셋으로 모델 학습을 재실행합니다.

**수행할 작업**:

1.  **TensorBoard 로거 구현**: `packnet_sfm/loggers/tensorboard_logger.py` 파일을 생성하고, `torch.utils.tensorboard.SummaryWriter`를 사용하여 로깅 기능을 구현합니다.
    -   `__init__`: `SummaryWriter`를 초기화하고 로그 디렉토리를 설정합니다.
    -   `log_metrics(metrics, step)`: 학습/검증 손실 및 메트릭을 스칼라로 로깅합니다.
    -   `log_depth(mode, batch, output, args, dataset, world_size, config)`: 예측 깊이, Ground Truth 깊이, 입력 이미지를 시각화하여 로깅합니다.
        -   **용량 제약 준수**: `log_depth`는 **매우 드물게(예: 1 에포크당 1회)** 또는 **일부 샘플에 대해서만** 로깅하도록 로직을 구현합니다. (예: `if step % log_interval == 0` 또는 `if batch_idx == 0`)
        -   이미지/깊이 맵은 PyTorch 텐서로 변환된 후 `add_image` 또는 `add_images`를 사용하여 로깅합니다. 필요시 `torchvision.utils.make_grid`를 활용하여 여러 이미지를 하나의 그리드로 묶어 로깅할 수 있습니다.
2.  **로거 등록**: `packnet_sfm/loggers/__init__.py` 파일을 수정하여 `TensorboardLogger`를 등록합니다.
3.  **`train.py` 및 설정 파일 수정**: `train.py`가 `TensorboardLogger`를 사용하도록 설정하고, `configs/train_resnet_san_ncdb.yaml` 파일에 TensorBoard 관련 설정을 추가합니다.
    -   `wandb` 섹션과 유사하게 `tensorboard` 섹션을 추가하고, `dry_run`을 `False`로 설정하여 활성화합니다.
4.  **학습 재실행**: 확장된 데이터셋과 TensorBoard 로깅이 활성화된 상태로 학습을 시작합니다.
    ```bash
    python scripts/train.py configs/train_resnet_san_ncdb.yaml
    ```
5.  **TensorBoard 확인**: 별도의 터미널에서 `tensorboard --logdir=outputs/` 명령어를 실행하여 학습 진행 상황을 시각적으로 확인합니다.

**PM 검토 요청**: TensorBoard 로그가 성공적으로 생성되고, 손실 및 깊이 시각화가 예상대로 나타나는지 확인한 후, TensorBoard 스크린샷 또는 로그 요약을 제출하여 최종 검토를 받습니다.

---

### **Step 8: 평가(Evaluation) 로직 검증**

**목표**: `ncdb-cls-sample` 데이터셋에 대한 평가가 올바르게 수행되고 있는지, 특히 유효한(값이 있는) 깊이 데이터만 사용하여 평가가 이루어지는지 확인합니다.

**수행할 작업**:

1.  **평가 메트릭 확인**: 학습 로그(콘솔 출력 또는 TensorBoard)에서 `validation_epoch_end` 또는 `_quick_eval`에서 보고되는 깊이 평가 메트릭(`abs_rel`, `rmse` 등)을 면밀히 검토합니다.
2.  **유효 깊이 값 확인**:
    *   `packnet_sfm/utils/depth.py` 파일에서 `compute_depth_metrics` 함수를 찾아, 이 함수가 깊이 맵의 유효한 픽셀(예: 0이 아닌 값)만 사용하여 메트릭을 계산하는지 확인합니다.
    *   필요하다면, `compute_depth_metrics` 함수 내부에 디버깅 `print` 문을 추가하여 마스킹 또는 유효 픽셀 계산 로직을 검증합니다.
3.  **데이터로더 확인**: `packnet_sfm/datasets/ncdb_dataset.py`에서 깊이 데이터가 로드될 때 유효성 검사 또는 마스킹이 제대로 적용되는지 다시 한번 확인합니다.
4.  **TensorBoard 시각화 재확인**: TensorBoard에서 Ground Truth 깊이 맵과 예측 깊이 맵을 비교하여, 특히 깊이 값이 없는 영역(검은색)이 평가에 영향을 미치지 않는지 육안으로 확인합니다.

**PM 검토 요청**: 평가 로직이 올바르게 작동하며, 유효한 깊이 값만으로 평가가 이루어지고 있음을 증명하는 분석 결과(예: 코드 스니펫, 디버그 출력 요약, TensorBoard 시각화 설명)를 보고합니다.

---
**From. PM**
_각 단계별로 명확한 결과물을 기대하며, 막히는 부분이 있다면 언제든지 질문해 주세요._
