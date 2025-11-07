# Checkpoint Monitoring Root Cause Fix
**담당자**: World-Class Developer  
**작성일**: 2024-12-19  
**상태**: 📋 Planning (미적용)  
**우선순위**: 🟡 Medium (하지만 권장)  
**영향도**: 🔧 Code Quality + 기능성

---

## 📋 Executive Summary

**문제**: `save_top_k` 파라미터가 checkpoint 모니터링 메트릭 부재로 작동하지 않음  
**근본 원인**: `validation_epoch_end()`에서 `'loss'` 메트릭이 반환되지 않음  
**현재 상태**: 우회 해결책(save_top_k: -1) 적용됨  
**최적 해결책**: 2단계 근본 원인 해결

---

## 🔍 문제 분석

### 1. 문제 지점

#### 현상
```yaml
checkpoint:
    save_top_k: 3              # ❌ 작동하지 않음
    monitor: 'loss'            # ❌ 메트릭 없음
```

학습 중 checkpoint 저장 시:
```
AssertionError: Checkpoint metric is not available
```

#### 근본 원인 추적

**Step 1**: model_checkpoint.py (Line 140-141)
```python
if self.save_top_k != -1:
    current = metrics.get(self.monitor)        # 'loss' 찾음
    assert current, 'Checkpoint metric is not available'  # ❌ 실패
```

**Step 2**: horovod_trainer.py (Line 123)
```python
validation_output = self.validate(val_dataloaders, module)
self.check_and_save(module, validation_output)  # metrics 전달
```

**Step 3**: model_wrapper.py (Line 449-515)
```python
def validation_epoch_end(self, output_data_batch):
    metrics_dict = create_dict(...)  # 'loss' 없음!
    return metrics_dict              # ❌ 'loss' 키 부재
```

**Step 4**: reduce.py (Line 117-150)
```python
def create_dict(metrics_data, metrics_keys, metrics_modes, ...):
    # metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', ...)
    # 'loss'는 포함되지 않음!
    for i, key in enumerate(metrics_keys):  # ← 'loss' 없음
        metrics_dict[f'{prefix}-{key}{mode}'] = ...
    return metrics_dict  # ❌ 'loss' 키 없는 상태
```

---

## 💡 해결 방안 (2가지)

### 방안 A: YAML 설정 변경 (즉시 적용 가능)

#### 설명
이미 `validation_epoch_end()`에서 반환되는 메트릭(`'depth-abs_rel0'`)을 사용

#### 파일: `configs/train_resnet_san_ncdb_dual_head_640x384.yaml`

**수정 전**:
```yaml
checkpoint:
    filepath: 'checkpoints/resnetsan01_dual_head_ncdb_640x384/'
    save_top_k: -1                    # 모든 checkpoint 저장 (우회)
    period: 2                         # 2 epoch마다 저장
```

**수정 후**:
```yaml
checkpoint:
    filepath: 'checkpoints/resnetsan01_dual_head_ncdb_640x384/'
    save_top_k: 3                     # ⭐ 상위 3개 유지
    monitor: 'depth-abs_rel0'         # ⭐ 기존 메트릭 사용
    mode: 'min'                       # ⭐ 낮을수록 좋음
    period: 1                         # ⭐ 매 epoch 확인
```

#### 작동 원리

```python
# create_dict() 실행 결과
metrics_dict = {
    'depth-abs_rel0': 0.120,      # ✅ 존재!
    'depth-sqr_rel0': 0.045,
    'depth-rmse0': 0.50,
    ...
}

# ModelCheckpoint.check_and_save()
current = metrics_dict.get('depth-abs_rel0')  # ✅ 0.120 획득
assert current  # ✅ 성공!
# → save_top_k 모니터링 정상 작동
```

#### 장점
- ✅ YAML만 수정 (1분)
- ✅ 코드 수정 불필요
- ✅ 즉시 적용 가능
- ✅ 안전 (검증된 메트릭)

#### 단점
- ⚠️ 메트릭 키 이름 명시적 필요 (`depth-abs_rel0`)
- ⚠️ `monitor` 기본값(`'loss'`)과 불일치

#### 검증 체크리스트
- [ ] 'depth-abs_rel0'가 create_dict()에서 생성되는지 확인
  ```python
  # model_wrapper.py Line 60
  self.metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', ...)  # ✅
  # reduce.py Line 137
  metrics_dict['depth-abs_rel0'] = ...  # ✅ 생성됨
  ```
- [ ] ModelCheckpoint에서 접근 가능한지 확인
  ```python
  # model_checkpoint.py Line 140
  current = metrics.get('depth-abs_rel0')  # ✅ 접근 가능
  ```
- [ ] YAML 파일 유효성 확인
  ```bash
  python -c "from packnet_sfm.utils.config import parse_train_file; \
  config, _ = parse_train_file('configs/train_resnet_san_ncdb_dual_head_640x384.yaml'); \
  print(config.checkpoint.monitor)"
  ```

---

### 방안 B: 코드 수정 (근본 해결)

#### 설명
`validation_epoch_end()`에서 validation loss를 metrics_dict에 추가

#### 파일 1: `packnet_sfm/models/model_wrapper.py`

**위치**: `validation_epoch_end()` 함수 내, 마지막 return 전

**현재 코드** (라인 490-512):
```python
        # Log to wandb
        if self.loggers:
            # Filter metrics to log only essential validation metrics
            log_metrics = {
                'global_step': self.current_epoch + 1,
            }
            for key, val in metrics_dict.items():
                if key.startswith('depth'):
                    log_metrics[f'val/{key}'] = val
            
            # Add validation loss if available
            if val_loss is not None:
                log_metrics['val/loss'] = val_loss

            for logger in self.loggers:
                logger.log_metrics(log_metrics, step=self.current_epoch + 1)

        return {
            **metrics_dict
        }
```

**수정 후** (이 코드 추가):
```python
        # Log to wandb
        if self.loggers:
            # Filter metrics to log only essential validation metrics
            log_metrics = {
                'global_step': self.current_epoch + 1,
            }
            for key, val in metrics_dict.items():
                if key.startswith('depth'):
                    log_metrics[f'val/{key}'] = val
            
            # Add validation loss if available
            if val_loss is not None:
                log_metrics['val/loss'] = val_loss

            for logger in self.loggers:
                logger.log_metrics(log_metrics, step=self.current_epoch + 1)

        # ✅ 🆕 Add loss to metrics_dict for checkpoint monitoring
        # This allows save_top_k to work properly when monitor='loss' (default config)
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

#### 작동 원리

```python
# 수정 후 결과
metrics_dict = {
    'depth-abs_rel0': 0.120,
    'depth-sqr_rel0': 0.045,
    'depth-rmse0': 0.50,
    'loss': 0.2654,            # ✅ 🆕 추가됨!
    ...
}

# ModelCheckpoint.check_and_save()
current = metrics_dict.get('loss')  # ✅ 0.2654 획득
assert current  # ✅ 성공!
# → save_top_k + monitor='loss' 정상 작동
```

#### 장점
- ✅ default_config 기본값 사용 가능 (`monitor: 'loss'`)
- ✅ 코드 일관성 (val_loss 이미 계산되고 있음)
- ✅ 다른 YAML 파일에도 자동 적용
- ✅ 범용성 높음

#### 단점
- ⚠️ 코드 수정 필요 (~15줄)
- ⚠️ 테스트 필요
- ⚠️ 다른 모델 호환성 확인 필요

#### 코드 검증 사항
- [ ] imports 확인 (torch 이미 임포트됨)
  ```python
  # model_wrapper.py 상단
  import torch  # ✅ 이미 있음
  ```
- [ ] output_data_batch 구조 확인
  ```python
  # 라인 469-475 분석
  for n, dataloader in enumerate(dataloaders):
      outputs = []
      for i, batch in progress_bar:
          output = module.validation_step(batch, i, n)
          outputs.append(output)  # ← 각 output에 'loss' 있음
  ```
- [ ] 테스트 코드 작성
  ```python
  # 확인 방법
  def test_validation_epoch_end_loss():
      # validation_epoch_end() 호출 후
      metrics = model.validation_epoch_end(output_data_batch)
      assert 'loss' in metrics, "Loss key missing!"
      assert isinstance(metrics['loss'], float), "Loss should be float"
  ```

---

## 📊 방안 비교표

| 기준 | 방안 A (YAML) | 방안 B (코드) | 우회책 (현재) |
|------|--------|--------|---------|
| **복잡도** | ⭐ 매우 간단 | ⭐⭐⭐ 중간 | - |
| **코드 수정** | 불필요 | ~15줄 필요 | - |
| **즉시 적용** | ✅ 1분 | ⚠️ 테스트 필요 | ✅ 적용됨 |
| **운영 편의성** | ⭐ 간단 | ⭐⭐ 일반 | - |
| **코드 품질** | - | ⭐⭐⭐ 최고 | - |
| **범용성** | 해당 YAML만 | 전체 모델 | - |
| **안정성** | ⭐ 높음 | ⭐⭐ 검증 필요 | ✅ |
| **default_config 호환성** | ⚠️ 낮음 | ✅ 높음 | - |

---

## 🎯 권장 적용 계획

### Phase 1: 즉시 (당일)
**적용**: 방안 A (YAML 수정)
- 시간: ~1분
- 검증: ~2분
- 리스크: 최소

```yaml
checkpoint:
    save_top_k: 3
    monitor: 'depth-abs_rel0'
    mode: 'min'
    period: 1
```

### Phase 2: 단기 (1주일 이내)
**적용**: 방안 B (코드 수정)
- 준비: 코드 리뷰
- 테스트: 2-3시간
- PR 검토: 1시간

**단계별 실행**:
1. 코드 수정 (model_wrapper.py)
2. Unit test 작성
3. Integration test 실행
4. 다른 모델 호환성 확인
5. PR 제출 및 리뷰

### Phase 3: 최종
**결과**:
- YAML에서 `monitor: 'loss'` 사용 가능 (default_config 기본값)
- 모든 depth estimation 모델에 자동 적용
- 더 우아한 checkpoint 관리

---

## 📝 실행 시 주의사항

### 방안 B 코드 수정 시 확인사항

1. **Loss 계산 방식 검증**
   ```python
   # val_loss 이미 계산되어 있는지 확인 (라인 446)
   if 'loss' in output:
       losses.append(output['loss'])
   if losses:
       val_loss = torch.tensor(losses).mean().item() if isinstance(...) else sum(losses) / len(losses)
   ```
   → val_loss가 이미 계산됨, metrics_dict에만 추가하면 됨

2. **Tensor 변환 안정성**
   ```python
   # torch.Tensor이거나 float일 수 있음
   if isinstance(loss_val, torch.Tensor):
       loss_val = loss_val.item()
   ```

3. **None 값 처리**
   ```python
   # 모든 output에 'loss'가 있는지 확실하지 않음
   if all_losses:
       metrics_dict['loss'] = sum(all_losses) / len(all_losses)
   ```

4. **다른 모델 영향 확인**
   ```
   - SemiSupCompletionModel ✓ (현재 테스트됨)
   - PackNetSANModel ? (확인 필요)
   - 다른 Depth models ? (확인 필요)
   ```

---

## 🧪 테스트 계획

### 방안 B 적용 후 테스트

```python
# Test 1: Metrics 딕셔너리 구조
def test_validation_epoch_end_structure():
    model = SemiSupCompletionModel(...)
    output = model.validation_epoch_end(output_data_batch)
    
    assert 'loss' in output, "Loss should be in metrics"
    assert 'depth-abs_rel0' in output, "abs_rel should be in metrics"
    assert isinstance(output['loss'], float), "Loss should be float"
    assert 0 < output['loss'] < 100, "Loss value reasonable"

# Test 2: Checkpoint 저장 동작
def test_checkpoint_with_loss_monitoring():
    config = parse_train_file('configs/train_resnet_san_ncdb_dual_head_640x384.yaml')
    
    # save_top_k=3, monitor='loss'로 설정
    checkpoint = ModelCheckpoint(
        filepath=config.checkpoint.filepath,
        save_top_k=3,
        monitor='loss',
        mode='min'
    )
    
    # validation_output에 'loss' 있는지 확인
    validation_output = model.validation_epoch_end(batch_outputs)
    checkpoint.check_and_save(model, validation_output)
    
    # AssertionError 발생하지 않는지 확인
    # ✓ checkpoint 정상 저장

# Test 3: 우회책(save_top_k=-1) 대비 성능
def test_disk_usage_comparison():
    # Before: save_top_k=-1, period=2 → 15개 checkpoint (30 epoch)
    # After: save_top_k=3 → 3개 checkpoint 최대 유지
    
    disk_saved = 12 / 15 * 100  # 80% 디스크 절약
    assert disk_saved > 50, "Should save significant disk space"
```

---

## 📚 참고 자료

### 관련 코드 위치

| 파일 | 라인 | 내용 |
|------|------|------|
| model_wrapper.py | 449-515 | validation_epoch_end() 함수 |
| reduce.py | 117-150 | create_dict() 함수 |
| model_checkpoint.py | 132-150 | check_and_save() 로직 |
| horovod_trainer.py | 123 | checkpoint 호출 지점 |
| default_config.py | 245-251 | checkpoint 기본 설정 |

### 관련 이슈

**이슈**: save_top_k가 작동하지 않음  
**근본 원인**: validation_epoch_end()에서 monitor 메트릭 부재  
**영향 범위**: save_top_k > 0 사용하는 모든 config  
**심각도**: 🔴 High (checkpoint 관리 비효율)

---

## ✅ 체크리스트

### 방안 A (YAML) 적용 전
- [ ] 'depth-abs_rel0' 메트릭 반환 확인
- [ ] ModelCheckpoint에서 접근 가능 확인
- [ ] YAML 파일 유효성 검증

### 방안 B (코드) 적용 전
- [ ] loss 값 범위 확인 (reasonable)
- [ ] Tensor 변환 안정성 확인
- [ ] 다른 모델 호환성 확인
- [ ] 코드 리뷰 완료
- [ ] Unit test 작성 완료

### 방안 B 적용 후
- [ ] Integration test 통과
- [ ] 다른 모델에서 테스트
- [ ] Checkpoint 생성 확인
- [ ] Disk 사용량 비교
- [ ] PR 리뷰 완료

---

## 📞 연락처 및 참고

**작성자**: World-Class Developer  
**검수자**: (필요시)  
**최종 승인**: (필요시)

**관련 문서**:
- [DEVELOPER_CHECKPOINT_ANALYSIS.md](../quantization/ST2/DEVELOPER_CHECKPOINT_ANALYSIS.md) - 초기 분석
- [model_checkpoint.py](../../packnet_sfm/models/model_checkpoint.py) - 구현 코드
- [model_wrapper.py](../../packnet_sfm/models/model_wrapper.py) - 수정 대상

---

**Status**: 📋 Documentation Complete - Ready for Implementation  
**Next Step**: Phase 1 (YAML 수정) 적용 시작  
**Last Updated**: 2024-12-19
