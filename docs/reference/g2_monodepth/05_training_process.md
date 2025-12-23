# G2-MonoDepth 학습 프로세스 상세

## 1. 학습 파이프라인 개요

### 1.1 전체 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                         train.py                                    │
├─────────────────────────────────────────────────────────────────────┤
│  1. Config 로드                                                     │
│  2. DDP 초기화 (Multi-GPU)                                          │
│  3. Trainer 인스턴스 생성                                            │
│  4. 학습 시작                                                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       src_main.py (Trainer)                         │
├─────────────────────────────────────────────────────────────────────┤
│  - 모델 초기화 (UNet)                                                │
│  - Loss 함수 초기화                                                  │
│  - Optimizer 초기화 (Adam)                                          │
│  - Scheduler 초기화                                                  │
│  - DataLoader 초기화                                                │
│  - 학습 루프 실행                                                    │
│  - 검증 및 체크포인트 저장                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. train.py 분석

### 2.1 진입점

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.src_main import Trainer
from config import Config

def main_worker(rank, world_size, config):
    """
    각 GPU에서 실행되는 worker 함수
    
    Args:
        rank: 현재 GPU 번호 (0, 1, 2, ...)
        world_size: 총 GPU 개수
        config: 설정 객체
    """
    # DDP 초기화
    setup_ddp(rank, world_size, config.port)
    
    # Trainer 생성 및 학습
    trainer = Trainer(rank, world_size, config)
    trainer.train()
    
    # 정리
    dist.destroy_process_group()

def main():
    config = Config()
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # Multi-GPU: DDP 사용
        mp.spawn(
            main_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU
        main_worker(0, 1, config)

if __name__ == '__main__':
    main()
```

### 2.2 DDP 초기화

```python
def setup_ddp(rank, world_size, port):
    """
    Distributed Data Parallel 초기화
    """
    import os
    
    # 환경 변수 설정
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Process Group 초기화
    dist.init_process_group(
        backend='nccl',      # NVIDIA GPU용 백엔드
        rank=rank,           # 현재 프로세스 순번
        world_size=world_size  # 총 프로세스 수
    )
    
    # 현재 프로세스의 GPU 설정
    torch.cuda.set_device(rank)
```

---

## 3. src_main.py (Trainer 클래스) 분석

### 3.1 초기화

```python
class Trainer:
    def __init__(self, rank, world_size, config):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.device = torch.device(f'cuda:{rank}')
        
        # 모델 초기화
        self.model = self._build_model()
        
        # Loss 함수 초기화
        self.loss_fns = self._build_loss()
        
        # Optimizer 초기화
        self.optimizer = self._build_optimizer()
        
        # Scheduler 초기화
        self.scheduler = self._build_scheduler()
        
        # DataLoader 초기화
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # 로깅
        if rank == 0:  # 메인 프로세스에서만 로깅
            self.logger = self._setup_logger()
```

### 3.2 모델 빌드

```python
def _build_model(self):
    """
    UNet 모델 생성 및 DDP 래핑
    """
    from src.networks import UNet
    
    model = UNet(
        in_channels=5,        # RGB + point + hole_point
        out_channels=1,       # depth
        layers=self.config.layers,      # 7
        features=self.config.features   # 64
    )
    
    # GPU에 올리기
    model = model.to(self.device)
    
    # DDP 래핑
    if self.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.rank],
            output_device=self.rank
        )
    
    return model
```

### 3.3 Loss 함수 빌드

```python
def _build_loss(self):
    """
    Loss 함수들 초기화
    """
    from src.losses import WeightedDataLoss, WeightedMSGradLoss
    from src.utils import StandardizeData
    
    return {
        'data_loss': WeightedDataLoss().to(self.device),
        'grad_loss': WeightedMSGradLoss(scales=[1, 2, 4, 8]).to(self.device),
        'standardize': StandardizeData().to(self.device)
    }
```

### 3.4 Optimizer 및 Scheduler

```python
def _build_optimizer(self):
    """
    Adam optimizer
    """
    return torch.optim.Adam(
        self.model.parameters(),
        lr=self.config.learning_rate,  # 1e-4
        betas=(0.9, 0.999),
        weight_decay=self.config.weight_decay  # 0 또는 1e-5
    )

def _build_scheduler(self):
    """
    Learning rate scheduler
    """
    return torch.optim.lr_scheduler.StepLR(
        self.optimizer,
        step_size=self.config.lr_step_size,  # 예: 10 epochs
        gamma=self.config.lr_gamma           # 예: 0.5
    )
```

---

## 4. 학습 루프

### 4.1 메인 학습 함수

```python
def train(self):
    """
    전체 학습 루프
    """
    for epoch in range(self.config.start_epoch, self.config.epochs):
        # 학습 모드
        self.model.train()
        
        # DDP: epoch별 sampler 동기화
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)
        
        # 에폭 학습
        train_loss = self._train_epoch(epoch)
        
        # 검증
        if (epoch + 1) % self.config.val_interval == 0:
            val_metrics = self._validate(epoch)
        
        # Scheduler step
        self.scheduler.step()
        
        # 체크포인트 저장 (메인 프로세스에서만)
        if self.rank == 0 and (epoch + 1) % self.config.save_interval == 0:
            self._save_checkpoint(epoch)
```

### 4.2 에폭 학습

```python
def _train_epoch(self, epoch):
    """
    1 에폭 학습
    """
    total_loss = 0.0
    num_batches = len(self.train_loader)
    
    for batch_idx, batch in enumerate(self.train_loader):
        # 데이터 GPU로 이동
        rgb = batch['rgb'].to(self.device)              # [B, 3, H, W]
        depth_gt = batch['depth'].to(self.device)       # [B, 1, H, W]
        point = batch['point'].to(self.device)          # [B, 1, H, W]
        hole_point = batch['hole_point'].to(self.device) # [B, 1, H, W]
        
        # 입력 구성
        inputs = torch.cat([rgb, point, hole_point], dim=1)  # [B, 5, H, W]
        
        # 마스크: GT가 유효한 위치
        mask = (depth_gt > 0).float()
        
        # Forward
        pred_depth = self.model(inputs)
        
        # Loss 계산
        loss, loss_dict = self._compute_loss(pred_depth, depth_gt, mask)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (선택적)
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
        
        self.optimizer.step()
        
        total_loss += loss.item()
        
        # 로깅
        if self.rank == 0 and batch_idx % self.config.log_interval == 0:
            self._log_training(epoch, batch_idx, num_batches, loss_dict)
    
    return total_loss / num_batches
```

### 4.3 Loss 계산

```python
def _compute_loss(self, pred_depth, gt_depth, mask):
    """
    Total loss 계산
    
    total_loss = loss_adepth + loss_rdepth + 0.5 * loss_rgrad
    """
    # 1. Absolute Depth Loss
    loss_adepth = self.loss_fns['data_loss'](pred_depth, gt_depth, mask)
    
    # 2. Relative Depth Loss
    sta_depth, sta_gt = self.loss_fns['standardize'](pred_depth, gt_depth, mask)
    loss_rdepth = self.loss_fns['data_loss'](sta_depth, sta_gt, mask)
    
    # 3. Gradient Loss (on relative domain)
    loss_rgrad = self.loss_fns['grad_loss'](sta_depth, sta_gt, mask)
    
    # 4. Total
    total_loss = loss_adepth + loss_rdepth + 0.5 * loss_rgrad
    
    loss_dict = {
        'total': total_loss.item(),
        'adepth': loss_adepth.item(),
        'rdepth': loss_rdepth.item(),
        'rgrad': loss_rgrad.item()
    }
    
    return total_loss, loss_dict
```

---

## 5. 검증 루프

### 5.1 검증 함수

```python
def _validate(self, epoch):
    """
    검증 수행 및 메트릭 계산
    """
    self.model.eval()
    
    metrics = {
        'rmse': 0.0,
        'mae': 0.0,
        'abs_rel': 0.0,
        'sq_rel': 0.0,
        'delta1': 0.0,  # δ < 1.25
        'delta2': 0.0,  # δ < 1.25²
        'delta3': 0.0,  # δ < 1.25³
    }
    num_samples = 0
    
    with torch.no_grad():
        for batch in self.val_loader:
            rgb = batch['rgb'].to(self.device)
            depth_gt = batch['depth'].to(self.device)
            point = batch['point'].to(self.device)
            hole_point = batch['hole_point'].to(self.device)
            
            # Forward
            inputs = torch.cat([rgb, point, hole_point], dim=1)
            pred_depth = self.model(inputs)
            
            # 메트릭 계산
            mask = (depth_gt > 0)
            batch_metrics = self._compute_metrics(pred_depth, depth_gt, mask)
            
            # 누적
            batch_size = rgb.size(0)
            for key in metrics:
                metrics[key] += batch_metrics[key] * batch_size
            num_samples += batch_size
    
    # 평균
    for key in metrics:
        metrics[key] /= num_samples
    
    # 로깅
    if self.rank == 0:
        self._log_validation(epoch, metrics)
    
    return metrics
```

### 5.2 메트릭 계산

```python
def _compute_metrics(self, pred, gt, mask):
    """
    Depth estimation 표준 메트릭 계산
    """
    pred = pred[mask]
    gt = gt[mask]
    
    # 기본 오차
    diff = pred - gt
    abs_diff = torch.abs(diff)
    
    # RMSE: Root Mean Squared Error
    rmse = torch.sqrt(torch.mean(diff ** 2))
    
    # MAE: Mean Absolute Error
    mae = torch.mean(abs_diff)
    
    # Abs Rel: |pred - gt| / gt
    abs_rel = torch.mean(abs_diff / gt)
    
    # Sq Rel: (pred - gt)² / gt
    sq_rel = torch.mean((diff ** 2) / gt)
    
    # Threshold accuracy: δ < thr
    ratio = torch.max(pred / gt, gt / pred)
    delta1 = torch.mean((ratio < 1.25).float())
    delta2 = torch.mean((ratio < 1.25 ** 2).float())
    delta3 = torch.mean((ratio < 1.25 ** 3).float())
    
    return {
        'rmse': rmse.item(),
        'mae': mae.item(),
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item(),
    }
```

---

## 6. 체크포인트 관리

### 6.1 저장

```python
def _save_checkpoint(self, epoch):
    """
    모델 체크포인트 저장
    """
    # DDP의 경우 module 접근
    model_state = self.model.module.state_dict() if self.world_size > 1 \
                  else self.model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'config': self.config,
    }
    
    save_path = f'{self.config.checkpoint_dir}/checkpoint_epoch{epoch:03d}.pth'
    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved: {save_path}')
```

### 6.2 로드

```python
def _load_checkpoint(self, checkpoint_path):
    """
    체크포인트에서 복원
    """
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
    # 모델 로드
    model_state = checkpoint['model_state_dict']
    if self.world_size > 1:
        self.model.module.load_state_dict(model_state)
    else:
        self.model.load_state_dict(model_state)
    
    # Optimizer 로드
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Scheduler 로드
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 시작 에폭
    self.config.start_epoch = checkpoint['epoch'] + 1
    
    print(f'Checkpoint loaded: {checkpoint_path}')
```

---

## 7. DDP 상세

### 7.1 DDP의 동작 원리

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Master Process (rank=0)                          │
├─────────────────────────────────────────────────────────────────────┤
│  [GPU 0]                                                            │
│  - 데이터의 1/N 처리                                                 │
│  - Forward → Loss → Backward                                        │
│  - Gradient 수집 및 평균                                             │
│  - Parameter 업데이트                                                │
│  - 체크포인트 저장, 로깅                                              │
└─────────────────────────────────────────────────────────────────────┘
        ↑                   ↑                   ↑
        │    Gradient       │    Gradient       │
        │    Sync (NCCL)    │    Sync (NCCL)    │
        ↓                   ↓                   ↓
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Worker (rank=1)│   │ Worker (rank=2)│   │ Worker (rank=N)│
│  [GPU 1]      │   │  [GPU 2]      │   │  [GPU N]      │
│  - 데이터 1/N │   │  - 데이터 1/N │   │  - 데이터 1/N │
└───────────────┘   └───────────────┘   └───────────────┘
```

### 7.2 DistributedSampler

```python
def _build_dataloaders(self):
    """
    DDP용 DataLoader 생성
    """
    train_dataset = NYUv2Dataset(split='train', transform=TrainAugmentation())
    val_dataset = NYUv2Dataset(split='val', transform=None)
    
    if self.world_size > 1:
        # DDP: 각 GPU가 다른 데이터를 보도록 분배
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=self.config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=self.config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=self.config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

---

## 8. Config 설정

### 8.1 config.py

```python
class Config:
    # 데이터
    data_root = '/path/to/nyu_v2'
    img_size = (480, 640)  # H, W
    
    # 모델
    layers = 7
    features = 64
    in_channels = 5
    out_channels = 1
    
    # 학습
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 0
    epochs = 100
    start_epoch = 0
    
    # Scheduler
    lr_step_size = 20
    lr_gamma = 0.5
    
    # Gradient
    grad_clip = 0  # 0이면 비활성화
    
    # 로깅 및 저장
    log_interval = 100
    val_interval = 1
    save_interval = 5
    checkpoint_dir = './checkpoints'
    
    # DDP
    port = 12355
    num_workers = 4
```

---

## 9. 우리 프로젝트와의 비교

### 9.1 학습 구조 비교

| 항목 | G2-MonoDepth | PackNet-SfM (우리) |
|------|--------------|-------------------|
| 프레임워크 | 순수 PyTorch | PyTorch Lightning |
| DDP 방식 | 수동 구현 | Lightning 내장 |
| Config | Python Class | YAML |
| Logging | 수동 | TensorBoard/WandB |
| Checkpoint | 수동 | Lightning 자동 |

### 9.2 학습 파라미터 비교

| 항목 | G2-MonoDepth | 우리 |
|------|--------------|------|
| Optimizer | Adam | Adam |
| Learning Rate | 1e-4 | 2e-4 |
| Batch Size | 8 | 8 |
| Scheduler | StepLR | StepLR |

### 9.3 적용 가능한 요소

1. **Gradient Clipping**:
   - 학습 안정성 향상
   - 특히 gradient loss 추가 시 유용

2. **Loss Logging 세분화**:
   - 각 loss term 별도 로깅
   - 학습 분석에 유용

3. **메트릭 계산 통합**:
   - 검증 시 표준 메트릭 계산
   - δ1, δ2, δ3 등

---

## 10. 핵심 인사이트

### 10.1 학습 전략

1. **Multi-scale Loss**: Absolute + Relative + Gradient
2. **Robust Training**: DDP로 대규모 배치, 다양한 augmentation
3. **Progressive Learning**: Scheduler로 점진적 LR 감소

### 10.2 우리 프로젝트에의 시사점

- Gradient Loss 추가 시 loss logging 세분화 필요
- 학습 안정성 위해 gradient clipping 고려
- 검증 메트릭 다양화 (δ 계열 추가)

---

## 요약

G2-MonoDepth의 학습 프로세스는:
1. **DDP 기반** 효율적인 multi-GPU 학습
2. **3-term Loss** (Absolute + Relative + Gradient)
3. **표준 메트릭** 기반 검증
4. **깔끔한 체크포인트** 관리

우리 PackNet-SfM은 PyTorch Lightning을 사용하여 많은 부분이 자동화되어 있지만,
G2-MonoDepth의 **Loss 설계 철학**과 **메트릭 계산 방식**은 참고할 가치가 있습니다.
