# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import horovod.torch as hvd
import traceback
from packnet_sfm.trainers.base_trainer import BaseTrainer, sample_to_cuda
from packnet_sfm.utils.config import prep_logger_and_checkpoint
from packnet_sfm.utils.logging import print_config
from packnet_sfm.utils.logging import AvgMeter
from tqdm import tqdm


class HorovodTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hvd.init()
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        torch.cuda.set_device(hvd.local_rank())
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.avg_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)

        # 중간 평가 설정
        self.eval_during_training = kwargs.get('eval_during_training', True)
        self.eval_progress_interval = kwargs.get('eval_progress_interval', 0.1)
        self.eval_subset_size = kwargs.get('eval_subset_size', 50)
        self.last_eval_progress = 0.0

    @property
    def proc_rank(self):
        return hvd.rank()

    @property
    def world_size(self):
        return hvd.size()

    def fit(self, module):
        # Prepare module for training
        module.trainer = self
        prep_logger_and_checkpoint(module)
        print_config(module.config)

        # Send module to GPU
        module = module.to('cuda')
        module.configure_optimizers()

        # Create distributed optimizer
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(module.optimizer,
            named_parameters=module.named_parameters(), compression=compression)
        scheduler = module.scheduler

        # Get train and val dataloaders
        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()

        # 테스트 데이터로더 준비 (중간 평가용)
        if self.eval_during_training:
            try:
                test_dataloaders = module.test_dataloader()
                if test_dataloaders is not None and len(test_dataloaders) > 0:
                    self._prepare_test_subset(test_dataloaders)
                    if self.is_rank_0:
                        print("✅ Test dataloader prepared for intermediate evaluation")
                else:
                    self.eval_during_training = False
            except Exception:
                self.eval_during_training = False

        # Validate before training if requested
        if self.validate_first:
            validation_output = self.validate(val_dataloaders, module)
            self.check_and_save(module, validation_output)

        # Epoch loop
        for epoch in range(module.current_epoch, self.max_epochs):
            self.train_with_eval(train_dataloader, module, optimizer)
            validation_output = self.validate(val_dataloaders, module)
            self.check_and_save(module, validation_output)
            module.current_epoch += 1
            scheduler.step()

    def _prepare_test_subset(self, test_dataloaders):
        """테스트 데이터로더 준비"""
        self.test_dataloader = test_dataloaders[0]

    @torch.no_grad()
    def _quick_eval(self, module):
        """빠른 중간 평가 - 상세한 디버깅 포함"""
        if not hasattr(self, 'test_dataloader'):
            if self.is_rank_0:
                print("   ❌ No test_dataloader available")
            return {}
        
        module.eval()
        metrics = []
        
        try:
            eval_size = min(25, self.eval_subset_size)
            if self.is_rank_0:
                print(f"   📊 Running evaluation on {eval_size} samples...")
            
            for i, batch in enumerate(self.test_dataloader):
                if i >= eval_size:
                    break
                
                batch = sample_to_cuda(batch)
                
                # quick_test_step 호출 및 결과 확인
                output = module.quick_test_step(batch, i, 0)
                
                if self.is_rank_0 and i == 0:
                    print(f"   🔍 First batch output: {output}")
                
                # abs_rel 값 추출
                abs_rel = output.get('abs_rel', 0.0)
                if abs_rel > 0:
                    metrics.append(abs_rel)
                    if self.is_rank_0 and i < 3:
                        print(f"   ✅ Sample {i}: abs_rel = {abs_rel:.4f}")
            
            if metrics:
                avg_abs_rel = sum(metrics) / len(metrics)
                if self.is_rank_0:
                    print(f"   📈 Average abs_rel: {avg_abs_rel:.4f} (from {len(metrics)} samples)")
                return {'abs_rel': avg_abs_rel}
            else:
                if self.is_rank_0:
                    print("   ⚠️  No valid metrics collected")
                return {}
        
        except Exception as e:
            if self.is_rank_0:
                print(f"   ❌ Evaluation error: {e}")
                import traceback
                traceback.print_exc()
            return {}
        
        finally:
            module.train()

    def train_with_eval(self, dataloader, module, optimizer):
        """중간 평가가 포함된 훈련 - 단순 버전"""
        module.train()

        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)

        progress_bar = self.train_progress_bar(dataloader, module.config.datasets.train)
        outputs = []
        total_batches = len(dataloader)
        
        # 🆕 평가 간격을 배치 수로 계산 (더 예측 가능)
        eval_interval_batches = max(1, int(total_batches * self.eval_progress_interval))
        
        if self.is_rank_0:
            print(f"🔍 Will evaluate every {eval_interval_batches} batches")

        for batch_idx, batch in progress_bar:
            # 🆕 간단한 배치 기준 평가
            eval_info = ""
            if (self.eval_during_training and 
                batch_idx > 0 and 
                batch_idx % eval_interval_batches == 0):
                
                if self.is_rank_0:
                    print(f"\n🎯 EVALUATION at batch {batch_idx}/{total_batches}")
                
                eval_metrics = self._quick_eval(module)
                
                abs_rel = eval_metrics.get('abs_rel', 0.0)
                if abs_rel > 0:
                    eval_info = f" | Test abs_rel: {abs_rel:.4f}"
                    if self.is_rank_0:
                        print(f"✅ Result: {abs_rel:.4f}")

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
        # Start validation loop
        all_outputs = []
        # For all validation datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.validation, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a validation step
                batch = sample_to_cuda(batch)
                output = module.validation_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.validation_epoch_end(all_outputs)

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda', dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        self.evaluate(test_dataloaders, module)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start evaluation loop
        all_outputs = []
        # For all test datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a test step
                batch = sample_to_cuda(batch, self.dtype)
                output = module.test_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.test_epoch_end(all_outputs)
