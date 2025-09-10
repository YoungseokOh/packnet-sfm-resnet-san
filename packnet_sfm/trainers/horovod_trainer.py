# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import traceback
import json
from packnet_sfm.trainers.base_trainer import BaseTrainer, sample_to_cuda
from packnet_sfm.utils.logging import print_config, pcolor
from packnet_sfm.utils.logging import AvgMeter
from tqdm import tqdm
from packnet_sfm.utils.config import s3_url
from packnet_sfm.datasets.ncdb_dataset import NcdbDataset  # 필요시 import


class HorovodTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Single GPU setup
        print("🔧 Running in single GPU mode")
        torch.cuda.set_device(0)  # Use first GPU
            
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.avg_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)

        # 중간 평가 설정
        self.eval_during_training = kwargs.get('eval_during_training', True)
        self.eval_progress_interval = kwargs.get('eval_progress_interval', 0.1)
        self.eval_subset_size = kwargs.get('eval_subset_size', 50)
        
        if self.is_rank_0:
            print(pcolor('  |  eval_subset_size: {}'.format(self.eval_subset_size), 'yellow'))
            print(pcolor('  |  Single GPU mode (no Horovod)', 'yellow'))

    @property
    def proc_rank(self):
        return 0  # Always rank 0 in single GPU mode

    @property
    def world_size(self):
        return 1  # Always world size 1 in single GPU mode

    def fit(self, module):
        # Prepare module for training
        self.module = module
        module.trainer = self
        
        # Handle loggers and checkpoint path updates
        if module.loggers:
            for logger in module.loggers:
                if hasattr(logger, 'run_name') and hasattr(logger, 'run_url') and hasattr(logger, 'log_config') and not module.config.wandb.dry_run:
                    module.config.name = module.config.wandb.name = logger.run_name
                    module.config.wandb.url = logger.run_url
                    # If we are saving models we need to update the path
                    if module.config.checkpoint.filepath != '':
                        # Change checkpoint filepath
                        filepath = module.config.checkpoint.filepath.split('/')
                        filepath[-2] = module.config.name
                        module.config.checkpoint.filepath = '/'.join(filepath)
                        # Change callback dirpath
                        dirpath = os.path.join(os.path.dirname(
                            self.checkpoint.dirpath), module.config.name)
                        self.checkpoint.dirpath = dirpath
                        os.makedirs(dirpath, exist_ok=True)
                        module.config.checkpoint.s3_url = s3_url(module.config)
                    # Log updated configuration
                    logger.log_config(module.config)
        print_config(module.config)

        # Send module to GPU
        module = module.to('cuda')
        module.configure_optimizers()

        # Use regular optimizer (no distribution)
        optimizer = module.optimizer
        scheduler = module.scheduler

        # Get train and val dataloaders
        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()

        # cache total batches to avoid re-creating loaders inside steps
        try:
            module._train_total_batches = len(train_dataloader)
        except Exception:
            module._train_total_batches = None
        try:
            module._val_total_batches = sum(len(dl) for dl in val_dataloaders) if val_dataloaders else None
        except Exception:
            module._val_total_batches = None

        # Setup evaluation dataloaders
        self.eval_dataloaders = None
        if self.eval_during_training and val_dataloaders:
            try:
                self.eval_dataloaders = val_dataloaders
                if self.is_rank_0:
                    print("✅ Using validation dataloaders for intermediate evaluation:")
                    for i, dataloader in enumerate(val_dataloaders):
                        dataset_config = module.config.datasets.validation
                        input_depth_type = dataset_config.input_depth_type[i] if i < len(dataset_config.input_depth_type) else ''
                        eval_type = "RGB+LiDAR" if input_depth_type else "RGB-only"
                        print(f"   [{i}] {eval_type} evaluation")
            except Exception as e:
                if self.is_rank_0:
                    print(f"⚠️ Failed to prepare eval dataloaders: {e}")
                self.eval_during_training = False

        # Validate before training if requested
        if self.validate_first:
            validation_output = self.validate(val_dataloaders, module)
            self._save_eval_results(module.current_epoch, validation_output)
            self.check_and_save(module, validation_output)

        # Epoch loop
        for epoch in range(module.current_epoch, self.max_epochs):
            self.train_with_eval(train_dataloader, module, optimizer)
            validation_output = self.validate(val_dataloaders, module)
            self._save_eval_results(epoch, validation_output)
            self.check_and_save(module, validation_output)
            module.current_epoch += 1
            scheduler.step()

    @torch.no_grad()
    def _quick_eval(self, module):
        """효율적인 중간 평가 - validation 데이터로더 재사용"""
        if self.eval_dataloaders is None:
            if self.is_rank_0:
                print("   ❌ No eval dataloaders available")
            return {}
        
        module.eval()
        
        try:
            eval_size = max(50, self.eval_subset_size)
            if self.is_rank_0:
                print(f"   📊 Running evaluation on {eval_size} samples per dataloader...")
            
            results = {}
            
            # 🆕 각 validation 데이터로더에서 빠른 평가
            for i, dataloader in enumerate(self.eval_dataloaders):
                # 데이터로더 타입 확인
                dataset_config = module.config.datasets.validation
                input_depth_type = dataset_config.input_depth_type[i] if i < len(dataset_config.input_depth_type) else ''
                eval_type = "RGB+LiDAR" if input_depth_type else "RGB-only"
                
                # 빠른 평가 수행
                metrics = self._evaluate_single_dataloader(module, dataloader, eval_size, eval_type)
                
                # 결과 저장
                if eval_type == "RGB-only":
                    results['rgb_abs_rel'] = metrics.get('abs_rel', 0.0)
                else:  # RGB+LiDAR
                    results['rgbd_abs_rel'] = metrics.get('abs_rel', 0.0)
            
            return results
            
        except Exception as e:
            if self.is_rank_0:
                print(f"   ❌ Evaluation error: {e}")
            return {}
        
        finally:
            module.train()

    def _evaluate_single_dataloader(self, module, dataloader, eval_size, mode_name):
        """단일 데이터로더 평가 - 간소화된 버전 with image resizing"""
        metrics = []
        
        if self.is_rank_0:
            print(f"   🔍 Evaluating {mode_name}...")
        
        for i, batch in enumerate(dataloader):
            if i >= eval_size:
                break
            
            try:
                # 🆕 Validation 시 이미지 크기 맞추기
                # Training config에서 image_shape 가져오기
                train_config = module.config.datasets.train
                image_shape = getattr(train_config, 'image_shape', None)
                
                if image_shape and len(image_shape) == 2:
                    # RGB 이미지 리사이즈
                    if 'rgb' in batch and hasattr(batch['rgb'], 'shape'):
                        current_shape = batch['rgb'].shape[-2:]  # [H, W]
                        target_shape = tuple(image_shape)  # (H, W)
                        
                        if current_shape != target_shape:
                            # 🔧 이미지 크기 맞추기
                            import torch.nn.functional as F
                            batch['rgb'] = F.interpolate(
                                batch['rgb'], size=target_shape, 
                                mode='bilinear', align_corners=False
                            )
                            
                            # input_depth도 있으면 리사이즈
                            if 'input_depth' in batch and batch['input_depth'] is not None:
                                batch['input_depth'] = F.interpolate(
                                    batch['input_depth'], size=target_shape,
                                    mode='nearest'  # depth는 nearest 사용
                                )
                            
                            # context 이미지들도 리사이즈
                            if 'rgb_context' in batch and batch['rgb_context']:
                                resized_context = []
                                for ctx_img in batch['rgb_context']:
                                    if hasattr(ctx_img, 'shape'):
                                        ctx_resized = F.interpolate(
                                            ctx_img, size=target_shape,
                                            mode='bilinear', align_corners=False
                                        )
                                        resized_context.append(ctx_resized)
                                    else:
                                        resized_context.append(ctx_img)
                                batch['rgb_context'] = resized_context
                
                batch = sample_to_cuda(batch)
                
                # 🔍 배치 정보 확인 (첫 번째 샘플만)
                if self.is_rank_0 and i == 0:
                    has_input_depth = 'input_depth' in batch and batch['input_depth'] is not None
                    if has_input_depth:
                        valid_points = (batch['input_depth'] > 0).sum().item()
                        print(f"     LiDAR points: {valid_points}")
                        print(f"     Image shape: {batch['rgb'].shape[-2:]}")
                    else:
                        print(f"     RGB-only mode")
                        print(f"     Image shape: {batch['rgb'].shape[-2:]}")
                
                # validation_step 실행
                output = module.validation_step(batch, i, 0)
                
                # 메트릭 추출 (depth_gt의 첫 번째 값이 abs_rel)
                if isinstance(output, dict) and 'depth_gt' in output:
                    depth_gt_metrics = output['depth_gt']
                    if isinstance(depth_gt_metrics, torch.Tensor) and depth_gt_metrics.numel() >= 1:
                        abs_rel = depth_gt_metrics[0].item()
                        if abs_rel > 0:
                            metrics.append(abs_rel)
            
            except Exception as batch_error:
                if self.is_rank_0:
                    print(f"     ⚠️ Error processing {mode_name} batch {i}: {batch_error}")
                continue
        
        # 결과 계산 및 반환
        if metrics:
            avg_abs_rel = sum(metrics) / len(metrics)
            if self.is_rank_0:
                print(f"     📈 {mode_name} abs_rel: {avg_abs_rel:.4f} (from {len(metrics)} samples)")
            return {'abs_rel': avg_abs_rel}
        else:
            if self.is_rank_0:
                print(f"     ⚠️ No valid {mode_name} metrics collected")
            return {}

    def train_with_eval(self, dataloader, module, optimizer):
        module.train()

        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)

        progress_bar = self.train_progress_bar(dataloader, module.config.datasets.train)
        outputs = []
        total_batches = len(dataloader)
        
        # ❗ 수정: max(50, ...) 부분을 제거하여 YAML 설정이 직접 반영되도록 함
        # 0이 되는 것을 방지하기 위해 max(1, ...) 사용
        eval_interval_batches = max(1, int(total_batches * self.eval_progress_interval))
        
        if self.is_rank_0 and self.eval_during_training:
            print(pcolor('\n🔍 Will evaluate every {} batches'.format(eval_interval_batches), 'yellow', attrs=['bold']))

        for batch_idx, batch in progress_bar:
            # 🆕 간소화된 중간 평가
            eval_info = ""
            if (self.eval_during_training and 
                self.eval_dataloaders is not None and
                batch_idx > 0 and 
                batch_idx % eval_interval_batches == 0):
                
                if self.is_rank_0:
                    print(f"\n🎯 EVALUATION at batch {batch_idx}/{total_batches}")
                
                eval_metrics = self._quick_eval(module)
                
                # 🆕 간단한 결과 표시
                rgb_abs_rel = eval_metrics.get('rgb_abs_rel', None)
                rgbd_abs_rel = eval_metrics.get('rgbd_abs_rel', None)
                
                if rgb_abs_rel is not None and rgbd_abs_rel is not None:
                    improvement = rgb_abs_rel - rgbd_abs_rel
                    eval_info = f" | RGB={rgb_abs_rel:.4f} vs RGB+D={rgbd_abs_rel:.4f} (Δ{improvement:+.4f})"
                elif rgb_abs_rel is not None:
                    eval_info = f" | RGB={rgb_abs_rel:.4f}"
                elif rgbd_abs_rel is not None:
                    eval_info = f" | RGB+D={rgbd_abs_rel:.4f}"

            # 정상 훈련 스텝
            optimizer.zero_grad()
            batch = sample_to_cuda(batch)

            with torch.autograd.set_detect_anomaly(True):
                output = module.training_step(batch, batch_idx)
                loss = output['loss']
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss at step {batch_idx}: {loss}")
                loss.backward()

            optimizer.step()
            output['loss'] = output['loss'].detach()
            outputs.append(output)

            # Progress bar 업데이트
            if self.is_rank_0 and hasattr(progress_bar, 'set_description'):
                desc = f'Epoch {module.current_epoch} | Loss {self.avg_loss(output["loss"].item()):.4f}{eval_info}'
                progress_bar.set_description(desc)

        return module.training_epoch_end(outputs)

    def train(self, dataloader, module, optimizer):
        # Set module to train
        module.train()
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        outputs = []
        # For all batches
        for i, batch in progress_bar:
            # Reset optimizer
            optimizer.zero_grad()
            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch)

            # wrap both forward/backward in anomaly detection
            with torch.autograd.set_detect_anomaly(True):
                output = module.training_step(batch, i)
                torch.cuda.synchronize()
                loss = output['loss']
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss at step {i}: {loss}")
                loss.backward()

            optimizer.step()
            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            outputs.append(output)
            # Update progress bar if in rank 0
            if self.is_rank_0:
                progress_bar.set_description(
                    'Epoch {} | Avg.Loss {:.4f}'.format(
                        module.current_epoch, self.avg_loss(output['loss'].item())))
        # Return outputs for epoch end
        return module.training_epoch_end(outputs)

    def validate(self, dataloaders, module):
        # Set module to eval
        module.eval()

        all_outputs = []
        for n, dataloader in enumerate(dataloaders):
            progress_bar = self.val_progress_bar(dataloader, module.config.datasets.validation, n)
            outputs = []

            for i, batch in progress_bar:
                batch = sample_to_cuda(batch)
                output = module.validation_step(batch, i, n)
                outputs.append(output)

            all_outputs.append(outputs)
        return module.validation_epoch_end(all_outputs)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()

        all_outputs = []
        for n, dataloader in enumerate(dataloaders):
            progress_bar = self.val_progress_bar(dataloader, module.config.datasets.test, n)
            outputs = []

            for i, batch in progress_bar:
                batch = sample_to_cuda(batch, self.dtype)
                output = module.test_step(batch, i, n)
                outputs.append(output)

            all_outputs.append(outputs)
        return module.test_epoch_end(all_outputs)

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda', dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        self.evaluate(test_dataloaders, module)

    def _save_eval_results(self, epoch: int, results: dict):
        """중간 평가 후 checkpoint 폴더에 JSON 결과 저장"""
        if not (self.is_rank_0 and self.checkpoint and results):
            return
        # 체크포인트 폴더 경로
        ckpt_dir = getattr(self.checkpoint, 'dirpath', None)
        if not ckpt_dir:
            return
        save_dir = os.path.join(ckpt_dir, 'evaluation_results')
        os.makedirs(save_dir, exist_ok=True)
        fname = f'epoch_{epoch}_results.json'
        path = os.path.join(save_dir, fname)
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
