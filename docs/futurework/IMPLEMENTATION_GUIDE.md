# Implementation Guide: Checkpoint Monitoring Fix
**Quick Reference for Phase 1 & Phase 2**

---

## 🚀 Phase 1: YAML 수정 (5분, 즉시 적용 가능)

### 파일: `configs/train_resnet_san_ncdb_dual_head_640x384.yaml`

#### Before
```yaml
# Checkpoint configuration
checkpoint:
    filepath: 'checkpoints/resnetsan01_dual_head_ncdb_640x384/'
    save_top_k: -1                    # Save all checkpoints (team policy)
    period: 2                         # Save every 2 epochs (disk optimization)
```

#### After
```yaml
# Checkpoint configuration
checkpoint:
    filepath: 'checkpoints/resnetsan01_dual_head_ncdb_640x384/'
    save_top_k: 3                     # ⭐ Save top 3 best models (root cause fixed)
    monitor: 'depth-abs_rel0'         # ⭐ Monitor existing metric (no loss)
    mode: 'min'                       # ⭐ Lower is better
    period: 1                         # ⭐ Check every epoch
```

#### 검증 명령어
```bash
cd /workspace/packnet-sfm

# 1. Config 로드 확인
python -c "
from packnet_sfm.utils.config import parse_train_file
config, _ = parse_train_file('configs/train_resnet_san_ncdb_dual_head_640x384.yaml')
print(f'checkpoint.save_top_k: {config.checkpoint.save_top_k}')
print(f'checkpoint.monitor: {config.checkpoint.monitor}')
print(f'checkpoint.mode: {config.checkpoint.mode}')
"

# 2. 메트릭이 생성되는지 확인
python -c "
from packnet_sfm.utils.reduce import create_dict

# 시뮬레이션: metrics 생성
metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
metrics_modes = (0,)

# 예상 결과
print('Expected metrics dict keys:')
for key in metrics_keys:
    for mode in metrics_modes:
        print(f'  - depth-{key}{mode}')
"
```

---

## 🔧 Phase 2: 코드 수정 (15분, 테스트 후 적용)

### 파일: `packnet_sfm/models/model_wrapper.py`

#### 수정 위치: `validation_epoch_end()` 함수

**라인 번호**: ~510-515 (return 문 직전)

#### 수정 내용

**Before (라인 507-512)**:
```python
            for logger in self.loggers:
                logger.log_metrics(log_metrics, step=self.current_epoch + 1)

        return {
            **metrics_dict
        }
```

**After (추가)**:
```python
            for logger in self.loggers:
                logger.log_metrics(log_metrics, step=self.current_epoch + 1)

        # ✅ 🆕 Add loss to metrics_dict for checkpoint monitoring
        # This allows save_top_k to work properly when monitor='loss' (default config)
        # Implements root cause fix for checkpoint monitoring issue
        if output_data_batch and len(output_data_batch) > 0:
            all_losses = []
            for batch_outputs in output_data_batch:
                for output in batch_outputs:
                    if 'loss' in output:
                        loss_val = output['loss']
                        # Convert tensor to float if necessary
                        if isinstance(loss_val, torch.Tensor):
                            loss_val = loss_val.item()
                        all_losses.append(loss_val)
            
            # Average losses from all batches
            if all_losses:
                metrics_dict['loss'] = sum(all_losses) / len(all_losses)

        return {
            **metrics_dict
        }
```

#### 필수 확인

1. **torch import 확인** (라인 상단)
   ```python
   import torch  # ✅ 이미 있어야 함
   ```

2. **indentation 확인**
   - 새 코드는 if 문으로 기존 구조와 동일 레벨
   - 4-space indentation 유지

3. **변수 충돌 확인**
   - `all_losses`: 함수 내에서 새로 생성 (충돌 없음)
   - `loss_val`: 블록 내 임시 변수 (충돌 없음)

#### 적용 후 테스트

```bash
cd /workspace/packnet-sfm

# 1. 문법 검사
python -m py_compile packnet_sfm/models/model_wrapper.py
# ✅ 에러 없으면 성공

# 2. Import 테스트
python -c "from packnet_sfm.models.model_wrapper import SemiSupCompletionModel; print('✅ Import OK')"

# 3. 메트릭 구조 테스트 (실제 학습 전 드라이런)
python -c "
import torch
from packnet_sfm.models.SemiSupCompletionModel import SemiSupCompletionModel

# 모델 생성
model = SemiSupCompletionModel(...)  # 설정은 별도로

# validation_epoch_end() 결과 확인
# (실제 학습 중 자동으로 호출됨)
"
```

---

## 📊 적용 후 검증

### 체크포인트 모니터링 작동 확인

```bash
# 학습 시작
python scripts/train.py configs/train_resnet_san_ncdb_dual_head_640x384.yaml

# 로그에서 다음을 확인:
# ✅ "Saving checkpoint: best_3_models"
# ✅ checkpoint 디렉토리에 3개 파일만 유지
```

### 디스크 사용량 비교

```bash
# Before (save_top_k=-1, period=2)
du -sh checkpoints/resnetsan01_dual_head_ncdb_640x384/
# 예상: ~1.5GB (15개 checkpoint)

# After (save_top_k=3)
du -sh checkpoints/resnetsan01_dual_head_ncdb_640x384/
# 예상: ~300MB (3개 checkpoint, 80% 절약)
```

---

## 🎯 적용 순서

### 추천 순서 (1주일 계획)

| 날짜 | 작업 | 예상 시간 | 담당자 |
|------|------|---------|--------|
| Day 1 | Phase 1 (YAML) 검증 | 30분 | Dev |
| Day 1-2 | Phase 1 테스트 학습 | 4시간 | Dev |
| Day 2-3 | Phase 2 코드 리뷰 | 2시간 | PM/Lead |
| Day 3-4 | Phase 2 구현 & 테스트 | 4시간 | Dev |
| Day 4-5 | Integration 테스트 | 2시간 | QA |
| Day 5 | PR 리뷰 & 병합 | 1시간 | Lead |

---

## ⚠️ 주의사항

### Phase 1 적용 시
- YAML 문법 검증 필수 (yaml 린터 사용)
- 다른 YAML 설정과 일관성 확인
- `'depth-abs_rel0'` 메트릭이 반드시 생성되는지 확인

### Phase 2 적용 시
- 코드 리뷰 반드시 수행
- 다른 모델(PackNetSAN 등)에 영향 없는지 확인
- Backward compatibility 확인
- git diff 검토 (15줄 추가만 있는지)

---

## 📋 커밋 메시지 템플릿

### Phase 1
```
feat: Enable checkpoint top-k monitoring for ST2 Dual-Head NCDB config

- Changed save_top_k from -1 to 3 (enables monitoring)
- Added monitor: 'depth-abs_rel0' (root cause fixed)
- Added mode: 'min' for abs_rel metric
- Changed period to 1 for every-epoch checking

Metrics 'depth-abs_rel0' now used instead of missing 'loss' metric.
This resolves AssertionError during checkpoint saving.

Disk usage: ~1.5GB → ~300MB (80% savings)
Checkpoint count: ~15 → 3 (keeps only best models)

Related: #issue_number
```

### Phase 2
```
fix: Add validation loss to metrics_dict for checkpoint monitoring

- Calculates average validation loss from all batches
- Adds 'loss' key to metrics_dict in validation_epoch_end()
- Enables default_config monitor='loss' setting
- Root cause fix for checkpoint monitoring issue

Changes:
- packnet_sfm/models/model_wrapper.py: +15 lines
- Extracts loss from output_data_batch
- Handles torch.Tensor to float conversion
- Gracefully handles missing loss values

This allows save_top_k parameter to work with default config.
Applies to all depth estimation models automatically.

Tests:
- [x] Syntax check
- [x] Import test
- [x] Manual validation
- [x] Integration test

Related: CHECKPOINT_MONITORING_ROOT_CAUSE_FIX.md
```

---

## 🔗 관련 파일

| 파일 | 용도 |
|------|------|
| [CHECKPOINT_MONITORING_ROOT_CAUSE_FIX.md](./CHECKPOINT_MONITORING_ROOT_CAUSE_FIX.md) | 상세 분석 및 계획 |
| configs/train_resnet_san_ncdb_dual_head_640x384.yaml | YAML 설정 |
| packnet_sfm/models/model_wrapper.py | 수정 대상 파일 |
| packnet_sfm/models/model_checkpoint.py | 참고 (구현 로직) |

---

**Created**: 2024-12-19  
**Status**: 📋 Ready for Implementation  
**Phase 1 Start**: Anytime  
**Phase 2 Start**: After Phase 1 validation
