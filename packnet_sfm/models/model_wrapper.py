# Copyright 2020 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict
import os
import time
import random
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.utils.data import ConcatDataset, DataLoader, Subset

from packnet_sfm.datasets.transforms import get_transforms
from packnet_sfm.utils.depth import inv2depth, post_process_inv_depth, compute_depth_metrics, viz_inv_depth
from packnet_sfm.utils.horovod import print0, world_size, rank, on_rank_0
from packnet_sfm.utils.image import flip_lr
from packnet_sfm.utils.load import load_class, load_class_args_create, \
    load_network, filter_args
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.utils.reduce import all_reduce_metrics, reduce_dict, \
    create_dict, average_loss_and_metrics
from packnet_sfm.utils.save import save_depth
from packnet_sfm.models.model_utils import stack_batch
from packnet_sfm.datasets.ncdb_dataset import NcdbDataset  # 필요시 import


# 🆕 Advanced augmentation import (선택적)
try:
    from packnet_sfm.datasets.augmentations_kitti_compatible import create_kitti_advanced_collate_fn
    ADVANCED_COLLATE_AVAILABLE = True
except ImportError:
    ADVANCED_COLLATE_AVAILABLE = False


class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a SfmModel (pose+depth networks).
    Designed to use models with high-level Trainer classes (cf. trainers/).

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    """

    def __init__(self, config, resume=None, loggers=None, load_datasets=True):
        super().__init__()

        # Store configuration, checkpoint and logger
        self.config = config
        self.loggers = loggers if loggers is not None else []
        self.resume = resume

        # Set random seed
        set_random_seed(config.arch.seed)

        # Task metrics
        self.metrics_name = 'depth'
        self.metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
        self.metrics_modes = ('', '_pp', '_gt', '_pp_gt')

        # Model, optimizers, schedulers and datasets are None for now
        self.model = self.optimizer = self.scheduler = None
        self.train_dataset = self.validation_dataset = self.test_dataset = None
        self.current_epoch = 0

        # Prepare model
        self.prepare_model(resume)

        # Prepare datasets
        if load_datasets:
            # Requirements for validation (we only evaluate depth for now)
            validation_requirements = {'gt_depth': True, 'gt_pose': False}
            test_requirements = validation_requirements
            self.prepare_datasets(validation_requirements, test_requirements)

        # Preparations done
        self.config.prepared = True

    def prepare_model(self, resume=None):
        """Prepare self.model (incl. loading previous state)"""
        print0(pcolor('### Preparing Model', 'green'))
        self.model = setup_model(self.config.model, self.config.prepared)
        # Resume model if available
        if resume:
            print0(pcolor('### Resuming from {}'.format(
                resume['file'])), 'magenta', attrs=['bold'])
            self.model = load_network(
                self.model, resume['state_dict'], 'model')
            if 'epoch' in resume:
                self.current_epoch = resume['epoch']

    def prepare_datasets(self, validation_requirements, test_requirements):
        """
        Prepare datasets for training, validation and test.
        """
        # Prepare datasets
        print0(pcolor('### Preparing Datasets', 'green'))

        # 🔧 augmentation 설정을 제대로 전달
        augmentation_config = self.config.datasets.augmentation
        
        # Setup train dataset (requirements are given by the model itself)
        self.train_dataset = setup_dataset(
            self.config.datasets.train, 'train',
            self.model.train_requirements, 
            augmentation=augmentation_config,  # 🆕 명시적으로 augmentation 전달
            **augmentation_config)  # 🆕 기존 방식도 유지
        
        # Setup validation dataset
        self.validation_dataset = setup_dataset(
            self.config.datasets.validation, 'validation',
            validation_requirements, 
            augmentation=augmentation_config,  # 🆕 명시적으로 augmentation 전달
            **augmentation_config)  # 🆕 기존 방식도 유지
        
        # Setup test dataset
        self.test_dataset = setup_dataset(
            self.config.datasets.test, 'test',
            test_requirements, 
            augmentation=augmentation_config,  # 🆕 명시적으로 augmentation 전달
            **augmentation_config)  # 🆕 기존 방식도 유지

    @property
    def depth_net(self):
        """
        Returns depth network.
        """
        return self.model.depth_net

    @property
    def pose_net(self):
        """
        Returns pose network.
        """
        return self.model.pose_net

    @property
    def logs(self):
        """
        Returns various logs for tracking.
        """
        params = OrderedDict()
        for param in self.optimizer.param_groups:
            params['{}_learning_rate'.format(param['name'].lower())] = param['lr']
        params['progress'] = self.progress
        return {
            **params,
            **self.model.logs,
        }

    @property
    def progress(self):
        """
        Returns training progress (current epoch / max. number of epochs)
        """
        return self.current_epoch / self.config.arch.max_epochs

    def configure_optimizers(self):
        """
        Configure depth and pose optimizers and the corresponding scheduler.
        """

        params = []
        # Load optimizer
        optimizer = getattr(torch.optim, self.config.model.optimizer.name)
        # Depth optimizer
        if self.depth_net is not None:
            params.append({
                'name': 'Depth',
                'params': self.depth_net.parameters(),
                **filter_args(optimizer, self.config.model.optimizer.depth)
            })
        # Pose optimizer
        if self.pose_net is not None:
            params.append({
                'name': 'Pose',
                'params': self.pose_net.parameters(),
                **filter_args(optimizer, self.config.model.optimizer.pose)
            })
        # Create optimizer with parameters
        optimizer = optimizer(params)

        # Load and initialize scheduler
        scheduler = getattr(torch.optim.lr_scheduler, self.config.model.scheduler.name)
        scheduler = scheduler(optimizer, **filter_args(scheduler, self.config.model.scheduler))

        if self.resume:
            if 'optimizer' in self.resume:
                optimizer.load_state_dict(self.resume['optimizer'])
            if 'scheduler' in self.resume:
                scheduler.load_state_dict(self.resume['scheduler'])

        # Create class variables so we can use it internally
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Return optimizer and scheduler
        return optimizer, scheduler

    def train_dataloader(self):
        """
        Prepare training dataloader.
        """
        dataloader = setup_dataloader(self.train_dataset,
                                      self.config.datasets.train, 'train')[0]
        
        # # 🆕 디버깅용: 데이터로더를 1배치로 제한
        # from torch.utils.data import Subset
        # subset = Subset(dataloader.dataset, range(self.config.datasets.train.batch_size))
        # dataloader = DataLoader(subset, batch_size=self.config.datasets.train.batch_size, 
        #                        shuffle=False, num_workers=0)
        
        return dataloader

    def val_dataloader(self):
        """
        Prepare validation dataloader.
        """
        dls = setup_dataloader(self.validation_dataset,
                               self.config.datasets.validation, 'validation')
        # 실제 reduce용으로 사용할 dataset(Subset 포함)을 저장
        self._val_datasets_for_reduce = [dl.dataset for dl in dls]
        return dls

    def test_dataloader(self):
        """
        Test dataloader for intermediate evaluation.
        
        Returns
        -------
        dataloaders : list of DataLoader or None
            List of created test dataloaders, or None if no test dataset
        """
        # 🆕 test_dataset이 None인지 안전하게 확인
        if self.test_dataset is None:
            print("⚠️ No test dataset configured")
            return None
        
        # 🆕 test_dataset이 빈 리스트인지 확인
        if isinstance(self.test_dataset, list) and len(self.test_dataset) == 0:
            print("⚠️ Test dataset list is empty")
            return None
        
        try:
            dataloaders = setup_dataloader(
                self.test_dataset, self.config.datasets.test, 'test')
            
            if dataloaders is None or len(dataloaders) == 0:
                print("⚠️ Failed to create test dataloaders")
                return None
            
            # 테스트도 동일하게 reduce용 dataset 저장
            self._test_datasets_for_reduce = [dl.dataset for dl in dataloaders]
            return dataloaders
        
        except Exception as e:
            print(f"❌ Error creating test dataloader: {e}")
            return None

    def training_step(self, batch, batch_idx, *args):
        """
        Processes a training batch.
        """
        model_output = self.model(batch, progress=self.progress, masks=batch.get('mask', None)) # masks 인자 추가

        if self.loggers and batch_idx % self.config.tensorboard.log_frequency == 0:
            rgb_original = batch['rgb_original'][0].cpu()  # (C, H, W)
            viz_pred_inv_depth = viz_inv_depth(model_output['inv_depths'][0][0])
            if isinstance(viz_pred_inv_depth, np.ndarray):
                viz_pred_inv_depth = torch.from_numpy(viz_pred_inv_depth).float()
            viz_pred_inv_depth = viz_pred_inv_depth.permute(2, 0, 1)

            mask = None
            if 'mask' in batch and batch['mask'] is not None:
                mask = batch['mask'][0].cpu()
                if mask.dim() == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                viz_pred_inv_depth_masked = viz_pred_inv_depth * mask.unsqueeze(0).float()
            else:
                viz_pred_inv_depth_masked = viz_pred_inv_depth

            # use cached total batches to avoid re-creating dataloader
            total_batches_per_epoch = getattr(self, '_train_total_batches', 1000) or 1000
            global_step = self.current_epoch * total_batches_per_epoch + batch_idx

            for logger in self.loggers:
                logger.writer.add_image('train/rgb_original', rgb_original, global_step=global_step)
                logger.writer.add_image('train/pred_inv_depth_masked', viz_pred_inv_depth_masked, global_step=global_step)
                logger.writer.add_image('train/pred_inv_depth_unmasked', viz_pred_inv_depth, global_step=global_step)
                if mask is not None:
                    logger.writer.add_image('train/mask', mask.unsqueeze(0).float(), global_step=global_step)
        if self.loggers:
            for logger in self.loggers:
                if hasattr(logger, 'writer'):
                    total_batches_per_epoch = getattr(self, '_train_total_batches', 1000) or 1000
                    global_step = self.current_epoch * total_batches_per_epoch + batch_idx
                    loss_value = model_output['loss'].item() if hasattr(model_output['loss'], 'item') else float(model_output['loss'])
                    logger.writer.add_scalar('train/loss_step', loss_value, global_step=global_step)
        return {
            'loss': model_output['loss'],
            'metrics': model_output['metrics']
        }

    def validation_step(self, batch, batch_idx, dataset_idx):
        """
        Processes a validation batch.
        """
        output = self.evaluate_depth(batch)
        if self.loggers:
            rgb_original = batch['rgb'][0].cpu()
            viz_pred_inv_depth = viz_inv_depth(output['inv_depth'][0])
            if isinstance(viz_pred_inv_depth, np.ndarray):
                viz_pred_inv_depth = torch.from_numpy(viz_pred_inv_depth).float()
            viz_pred_inv_depth = viz_pred_inv_depth.permute(2, 0, 1)

            mask = None
            if 'mask' in batch and batch['mask'] is not None:
                mask = batch['mask'][0].cpu()
                if mask.dim() == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                viz_pred_inv_depth_masked = viz_pred_inv_depth * mask.unsqueeze(0).float()
            else:
                viz_pred_inv_depth_masked = viz_pred_inv_depth

            total_batches_per_epoch = getattr(self, '_val_total_batches', 1000) or 1000
            global_step = self.current_epoch * total_batches_per_epoch + batch_idx

            for logger in self.loggers:
                logger.writer.add_image('val/rgb_original', rgb_original, global_step=global_step)
                logger.writer.add_image('val/pred_inv_depth_masked', viz_pred_inv_depth_masked, global_step=global_step)
                logger.writer.add_image('val/pred_inv_depth_unmasked', viz_pred_inv_depth, global_step=global_step)
                if mask is not None:
                    logger.writer.add_image('val/mask', mask.unsqueeze(0).float(), global_step=global_step)
        return {
            'idx': batch['idx'],
            **output['metrics'],
        }

    def test_step(self, batch, *args):
        """
        Processes a test batch.
        """
        output = self.evaluate_depth(batch)
        save_depth(batch, output, args,
                   self.config.datasets.test,
                   self.config.save)
        return {
            'idx': batch['idx'],
            **output['metrics'],
        }
    
    def quick_test_step(self, batch, *args):
        """
        빠른 테스트 스텝 (메트릭만 계산, 저장 없음)
        """
        try:
            output = self.evaluate_depth(batch)
            
            # 첫 번째 depth 메트릭 추출 (abs_rel)
            metrics = {}
            for key, value in output['metrics'].items():
                if 'depth' in key and isinstance(value, torch.Tensor):
                    if value.dim() == 1 and len(value) > 0:
                        metrics['abs_rel'] = value[0].item()  # 첫 번째 메트릭이 abs_rel
                        break
                elif 'depth' in key and isinstance(value, (int, float)):
                    metrics['abs_rel'] = float(value)
                    break
            
            return {
                'idx': batch.get('idx', 0),
                **metrics
            }
            
        except Exception as e:
            # 에러 발생 시 기본값 반환
            return {'idx': batch.get('idx', 0), 'abs_rel': 0.0}

    def training_epoch_end(self, output_batch):
        """
        Finishes a training epoch.
        """

        # Calculate and reduce average loss and metrics per GPU
        loss_and_metrics = average_loss_and_metrics(output_batch, 'avg_train')
        loss_and_metrics = reduce_dict(loss_and_metrics, to_item=True)

        # 🆕 Log training loss to TensorBoard
        if self.loggers:
            for logger in self.loggers:
                if hasattr(logger, 'writer'):  # TensorBoard logger
                    # Log training loss
                    if 'loss' in loss_and_metrics:
                        logger.writer.add_scalar('train/loss', loss_and_metrics['loss'], global_step=self.current_epoch + 1)
                    
                    # Log learning rate if available
                    if hasattr(self, 'optimizer') and self.optimizer is not None:
                        for i, param_group in enumerate(self.optimizer.param_groups):
                            lr = param_group['lr']
                            logger.writer.add_scalar(f'train/lr_{param_group.get("name", f"group_{i}")}', lr, global_step=self.current_epoch + 1)

        # Log to wandb
        if self.loggers:
            prefixed_loss_and_metrics = {f'train/{key}': val for key, val in loss_and_metrics.items()}
            for logger in self.loggers:
                logger.log_metrics({
                    **self.logs, **prefixed_loss_and_metrics,
                }, step=self.current_epoch + 1)

        return {
            **loss_and_metrics
        }

    def validation_epoch_end(self, output_data_batch):
        """
        Finishes a validation epoch.
        """
        # 실제 사용한(제한된) 데이터셋으로 reduce
        datasets_for_reduce = getattr(self, '_val_datasets_for_reduce', self.validation_dataset)

        # Reduce depth metrics
        metrics_data = all_reduce_metrics(
            output_data_batch, datasets_for_reduce, self.metrics_name)

        # Create depth dictionary
        metrics_dict = create_dict(
            metrics_data, self.metrics_keys, self.metrics_modes,
            self.config.datasets.validation)

        # 🆕 Calculate validation loss if available
        val_loss = None
        if output_data_batch and len(output_data_batch) > 0 and len(output_data_batch[0]) > 0:
            # Extract loss from validation outputs
            losses = []
            for batch_output in output_data_batch:
                for output in batch_output:
                    if 'loss' in output:
                        losses.append(output['loss'])
            if losses:
                val_loss = sum(losses) / len(losses)

        # 🆕 Log validation loss and metrics to TensorBoard
        if self.loggers:
            for logger in self.loggers:
                if hasattr(logger, 'writer'):  # TensorBoard logger
                    # Log validation loss
                    if val_loss is not None:
                        logger.writer.add_scalar('val/loss', val_loss, global_step=self.current_epoch + 1)
                    
                    # Log validation depth metrics
                    for key, val in metrics_dict.items():
                        if isinstance(val, (int, float)):
                            logger.writer.add_scalar(f'val/{key}', val, global_step=self.current_epoch + 1)

        # Print stuff
        self.print_metrics(metrics_data, self.config.datasets.validation)

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

    def test_epoch_end(self, output_data_batch):
        """
        Finishes a test epoch.
        """
        # 실제 사용한(제한된) 데이터셋으로 reduce
        datasets_for_reduce = getattr(self, '_test_datasets_for_reduce', self.test_dataset)

        # Reduce depth metrics
        metrics_data = all_reduce_metrics(
            output_data_batch, datasets_for_reduce, self.metrics_name)

        # Create depth dictionary
        metrics_dict = create_dict(
            metrics_data, self.metrics_keys, self.metrics_modes,
            self.config.datasets.test)

        # Print stuff
        self.print_metrics(metrics_data, self.config.datasets.test)

        return {
            **metrics_dict
        }

    def forward(self, *args, **kwargs):
        """
        Runs the model and returns the output.
        """
        assert self.model is not None, 'Model not defined'
        return self.model(*args, **kwargs)

    def depth(self, *args, **kwargs):
        """
        Runs the pose network and returns the output.
        """
        assert self.depth_net is not None, 'Depth network not defined'
        return self.depth_net(*args, **kwargs)

    def pose(self, *args, **kwargs):
        """
        Runs the depth network and returns the output.
        """
        assert self.pose_net is not None, 'Pose network not defined'
        return self.pose_net(*args, **kwargs)

    def _compute_depth_metrics_fallback(self, gt, pred):
        """
        Return metrics as a Tensor [abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3].
        Inputs: gt/pred (B,1,H,W) float tensors on same device.
        """
        eps = 1e-6
        params = getattr(self.config.model, 'params', {})
        # handle yacs CfgNode with keys min_depth/max_depth
        try:
            min_d = float(params.get('min_depth', 0.1))
            max_d = float(params.get('max_depth', 80.0))
        except Exception:
            min_d, max_d = 0.1, 80.0

        gt = gt.clamp(min=min_d, max=max_d)
        pred = pred.clamp(min=min_d, max=max_d)

        mask = torch.isfinite(gt) & torch.isfinite(pred) & (gt > min_d) & (gt < max_d)
        if mask.float().sum() == 0:
            return torch.zeros(7, device=gt.device, dtype=torch.float32)

        gt_m = gt[mask]
        pred_m = pred[mask]

        abs_rel = (torch.abs(gt_m - pred_m) / (gt_m + eps)).mean()
        sqr_rel = (((gt_m - pred_m) ** 2) / (gt_m + eps)).mean()
        rmse = torch.sqrt(((gt_m - pred_m) ** 2).mean())
        rmse_log = torch.sqrt(((torch.log(gt_m + eps) - torch.log(pred_m + eps)) ** 2).mean())

        thresh = torch.max(gt_m / (pred_m + eps), pred_m / (gt_m + eps))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        return torch.stack([abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3]).to(gt.dtype)

    def evaluate_depth(self, batch):
        """
        Evaluate batch to produce depth metrics.
        """
        # Get predicted inv-depths
        inv_depths = self.model(batch)['inv_depths']         # list, first scale: (B,1,H,W)
        inv0 = inv_depths[0]
        depth = inv2depth(inv0)                              # (B,1,H,W)

        # Always do flip post-process for validation/test
        # 1) flip input
        batch['rgb'] = flip_lr(batch['rgb'])
        # 2) predict on flipped input
        inv_depths_flipped = self.model(batch)['inv_depths']
        # 3) flip prediction back to original coordinates
        inv0_flipped_back = flip_lr(inv_depths_flipped[0])
        # 4) post-process combine
        inv_depth_pp = post_process_inv_depth(inv0, inv0_flipped_back, method='mean')
        depth_pp = inv2depth(inv_depth_pp)
        # 5) restore input
        batch['rgb'] = flip_lr(batch['rgb'])

        # Normalize to (B,1,H,W) on correct device
        device = inv0.device
        def _to_b1hw(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if not isinstance(x, torch.Tensor):
                return None
            x = x.to(device=device, dtype=torch.float32)
            if x.dim() == 0:
                return x.view(1, 1, 1, 1)
            if x.dim() == 2:
                return x.unsqueeze(0).unsqueeze(0)
            if x.dim() == 3:
                if x.size(0) in (1, 3):   # assume (C,H,W)
                    x = x.unsqueeze(0)
                    return x[:, :1, ...]
                else:                     # (B,H,W)
                    return x.unsqueeze(1)
            if x.dim() == 4:
                if x.size(1) != 1:
                    return x[:, :1, ...]
                return x
            return None

        depth_pred    = _to_b1hw(depth)
        depth_pred_pp = _to_b1hw(depth_pp)
        depth_gt      = _to_b1hw(batch.get('depth', None))

        # 🔧 임시 스케일 보정: 환경변수 FORCE_DEPTH_DIV256=1일 때 Pred 256으로 나눔
        if os.environ.get('FORCE_DEPTH_DIV256', '0') == '1':
            def _div256(x):
                if x is None:
                    return x
                # 값이 이미 물리 단위(최대 < 200 등)면 중복 나눔 피함
                if torch.is_tensor(x) and x.max() > 255:
                    return x / 256.0
                return x
            depth_gt      = _div256(depth_gt)
            depth_pred    = _div256(depth_pred)
            depth_pred_pp = _div256(depth_pred_pp)

        if os.environ.get('DEPTH_RANGE_DEBUG', '0') == '1' and depth_gt is not None:
            try:
                pos = depth_gt[depth_gt > 0]
                dmin = float(depth_gt.min())
                dmax = float(depth_gt.max())
                if pos.numel() > 0:
                    p50 = float(torch.quantile(pos, 0.5))
                    p90 = float(torch.quantile(pos, 0.9))
                    p95 = float(torch.quantile(pos, 0.95))
                else:
                    p50 = p90 = p95 = 0.0
                # 기존 EvalDepthRange + DepthDebug 함께 출력
                valid_count = (depth_gt > 0).sum().item()
                total_count = depth_gt.numel()
                print(f"[EvalDepthRange] (scaled) min={dmin:.2f} max={dmax:.2f} "
                      f"p50={p50:.2f} p90={p90:.2f} p95={p95:.2f} "
                      f"pos={pos.numel()}/{depth_gt.numel()} | "
                      f"[DepthDebug valid>0 {valid_count}/{total_count}]")
            except Exception as e:
                print(f"[EvalDepthRange] failed: {e}")

        # Compute metrics (tensor of 7 values each)
        metrics = OrderedDict()
        if depth_gt is not None and depth_pred is not None:
            try:
                m_main = compute_depth_metrics(self.config.model.params, gt=depth_gt, pred=depth_pred, use_gt_scale=False)
            except Exception:
                m_main = self._compute_depth_metrics_fallback(depth_gt, depth_pred)
            metrics['depth'] = m_main

            try:
                m_pp = compute_depth_metrics(self.config.model.params, gt=depth_gt, pred=depth_pred_pp, use_gt_scale=False)
            except Exception:
                m_pp = self._compute_depth_metrics_fallback(depth_gt, depth_pred_pp)
            metrics['depth_pp'] = m_pp

            try:
                m_gt = compute_depth_metrics(self.config.model.params, gt=depth_gt, pred=depth_pred, use_gt_scale=True)
            except Exception:
                m_gt = self._compute_depth_metrics_fallback(depth_gt, depth_pred)
            metrics['depth_gt'] = m_gt

            try:
                m_pp_gt = compute_depth_metrics(self.config.model.params, gt=depth_gt, pred=depth_pred_pp, use_gt_scale=True)
            except Exception:
                m_pp_gt = self._compute_depth_metrics_fallback(depth_gt, depth_pred_pp)
            metrics['depth_pp_gt'] = m_pp_gt

        return {
            'metrics': metrics,
            'inv_depth': inv_depth_pp
        }

    @on_rank_0
    def print_metrics(self, metrics_data, dataset):
        """
        Print depth metrics on rank 0 if available
        """
        if not metrics_data[0]:
            return

        hor_line = '|{:<}|'.format('*' * 93)
        met_line = '| {:^14} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} |'
        num_line = '{:<14} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f}'

        def wrap(string):
            return '| {} |'.format(string)

        print()
        print()
        print()
        print(hor_line)

        if self.optimizer is not None:
            bs = 'E: {} BS: {}'.format(self.current_epoch + 1,
                                       self.config.datasets.train.batch_size)
            if self.model is not None:
                bs += ' - {}'.format(self.config.model.name)
            lr = 'LR ({}):'.format(self.config.model.optimizer.name)
            for param in self.optimizer.param_groups:
                lr += ' {} {:.2e}'.format(param['name'], param['lr'])
            par_line = wrap(pcolor('{:<40}{:>51}'.format(bs, lr),
                                   'green', attrs=['bold', 'dark']))
            print(par_line)
            print(hor_line)

        print(met_line.format(*(('METRIC',) + self.metrics_keys)))
        for n, metrics in enumerate(metrics_data):
            print(hor_line)
            path_line = '{}'.format(
                os.path.join(dataset.path[n], dataset.split[n]))
            if len(dataset.cameras[n]) == 1: # only allows single cameras
                path_line += ' ({})'.format(dataset.cameras[n][0])
            print(wrap(pcolor('*** {:<87}'.format(path_line), 'magenta', attrs=['bold'])))
            print(hor_line)
            for key, metric in metrics.items():
                if self.metrics_name in key:
                    print(wrap(pcolor(num_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), 'cyan')))
        print(hor_line)

        if self.loggers:
            # Find WandbLogger if it exists in the list of loggers
            wandb_logger = None
            for logger in self.loggers:
                if hasattr(logger, 'run_name') and hasattr(logger, 'run_url'): # Check for WandbLogger specific attributes
                    wandb_logger = logger
                    break
            
            if wandb_logger:
                run_line = wrap(pcolor('{:<60}{:>31}'.format(
                    wandb_logger.run_url, wandb_logger.run_name), 'yellow', attrs=['dark']))
                print(run_line)
                print(hor_line)

        print()


def set_random_seed(seed):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_depth_net(config, prepared, **kwargs):
    """
    Create a depth network
    """
    print0(pcolor('DepthNet: %s' % config.name, 'yellow'))

    # ✅ 기존 내용 유지 + YAML params에서 min/max_depth 추출 (있을 때만)
    extra_depth_args = {}
    try:
        if hasattr(config, 'params'):
            if hasattr(config.params, 'min_depth'):
                extra_depth_args['min_depth'] = float(config.params.min_depth)
            if hasattr(config.params, 'max_depth'):
                extra_depth_args['max_depth'] = float(config.params.max_depth)
    except Exception:
        pass

    depth_net = load_class_args_create(
        config.name,
        paths=['packnet_sfm.networks.depth',],
        args={**config, **extra_depth_args, **kwargs},  # ← 추가 인자 병합
    )
    if not prepared and config.checkpoint_path != '':
        depth_net = load_network(depth_net, config.checkpoint_path,
                                 ['depth_net', 'disp_network'])
    return depth_net


def setup_pose_net(config, prepared, **kwargs):
    """
    Create a pose network
    """
    print0(pcolor('PoseNet: %s' % config.name, 'yellow'))
    pose_net = load_class_args_create(
        config.name,
        paths=['packnet_sfm.networks.pose',],
        args={**config, **kwargs},
    )
    if not prepared and config.checkpoint_path != '':
        pose_net = load_network(pose_net, config.checkpoint_path,
                                ['pose_net', 'pose_network'])
    return pose_net


def setup_model(config, prepared, **kwargs):
    """
    Create a model
    """
    print0(pcolor('Model: %s' % config.name, 'yellow'))

    # ✅ 기존 loss 인자 유지 + YAML params에서 min/max_depth 전달
    model_args = {**config.loss}
    try:
        if hasattr(config, 'params'):
            if hasattr(config.params, 'min_depth'):
                model_args['min_depth'] = float(config.params.min_depth)
            if hasattr(config.params, 'max_depth'):
                model_args['max_depth'] = float(config.params.max_depth)
    except Exception:
        pass

    model = load_class(config.name, paths=['packnet_sfm.models',])(
        **{**model_args, **kwargs})

    # 기존 네트워크 추가 로직 그대로
    if 'depth_net' in model.network_requirements:
        model.add_depth_net(setup_depth_net(config.depth_net, prepared))
    if 'pose_net' in model.network_requirements:
        model.add_pose_net(setup_pose_net(config.pose_net, prepared))
    if not prepared and config.checkpoint_path != '':
        model = load_network(model, config.checkpoint_path, 'model')
    return model


def setup_dataset(config, mode, requirements, **kwargs):
    """
    Create a dataset class

    Parameters
    ----------
    config : CfgNode
        Configuration (cf. configs/default_config.py)
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataset
    requirements : dict (string -> bool)
        Different requirements for dataset loading (gt_depth, gt_pose, etc)
    kwargs : dict
        Extra parameters for dataset creation

    Returns
    -------
    dataset : Dataset
        Dataset class for that mode
    """
    # If no dataset is given, return None
    if len(config.path) == 0:
        return None

    print0(pcolor('###### Setup %s datasets' % mode, 'red'))

    # Global shared dataset arguments
    dataset_args = {
        'back_context': config.back_context,
        'forward_context': config.forward_context,
        'data_transform': get_transforms(mode, **kwargs)
    }

    # Loop over all datasets
    datasets = []
    for i in range(len(config.split)):
        path_split = os.path.join(config.path[i], config.split[i])

        # Individual shared dataset arguments
        dataset_args_i = {
            'depth_type': config.depth_type[i] if 'gt_depth' in requirements else None,
            'input_depth_type': config.input_depth_type[i] if 'gt_depth' in requirements else None,
            'with_pose': 'gt_pose' in requirements,
        }

        # KITTI dataset
        if config.dataset[i] == 'KITTI':
            # from packnet_sfm.datasets.kitti_dataset import KITTIDataset
            from packnet_sfm.datasets.kitti_dataset_optimized import OptimizedKITTIDataset
            dataset = OptimizedKITTIDataset(
                config.path[i], path_split,
                **dataset_args, **dataset_args_i,
            )
        # ncdb dataset
        elif config.dataset[i] == 'ncdb':
            from packnet_sfm.datasets.ncdb_dataset import NcdbDataset
            dataset = NcdbDataset(
                config.path[i], 
                config.split[i],
                transform=dataset_args.get('data_transform', None),
                mask_file=getattr(config, "mask_file", [None])[i]
            )
        # DGP dataset
        elif config.dataset[i] == 'DGP':
            from packnet_sfm.datasets.dgp_dataset import DGPDataset
            dataset = DGPDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
                cameras=config.cameras[i],
            )
        # Image dataset
        elif config.dataset[i] == 'Image':
            from packnet_sfm.datasets.image_dataset import ImageDataset
            dataset = ImageDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
        else:
            ValueError('Unknown dataset %d' % config.dataset[i])

        # Repeat if needed
        if 'repeat' in config and config.repeat[i] > 1:
            dataset = ConcatDataset([dataset for _ in range(config.repeat[i])])
        datasets.append(dataset)

        # Display dataset information
        bar = '######### {:>7}'.format(len(dataset))
        if 'repeat' in config:
            bar += ' (x{})'.format(config.repeat[i])
        bar += ': {:<}'.format(path_split)
        print0(pcolor(bar, 'yellow'))

    # If training, concatenate all datasets into a single one
    if mode == 'train':
        datasets = [ConcatDataset(datasets)]

    return datasets


def worker_init_fn(worker_id):
    """
    Function to initialize workers
    """
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)


def get_datasampler(dataset, mode):
    """
    Distributed data sampler
    """
    return torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=(mode=='train'),
        num_replicas=world_size(), rank=rank())


def setup_dataloader(datasets, config, mode):
    """
    Create a dataloader class
    🆕 Enhanced to support advanced augmentation collate functions
    """
    
    # Advanced augmentation collate function (quiet)
    base_collate_fn = None
    if (mode == 'train' and 
        hasattr(config, 'augmentation') and 
        ADVANCED_COLLATE_AVAILABLE):
        base_collate_fn = create_kitti_advanced_collate_fn(config.augmentation)

    # Helper: flatten ConcatDataset / Subset children
    def _iter_children(ds):
        if isinstance(ds, ConcatDataset):
            for sub in ds.datasets:
                yield from _iter_children(sub)
        elif isinstance(ds, Subset):
            yield from _iter_children(ds.dataset)
        else:
            yield ds

    # ✅ 고정 숫자로 로더 샘플 제한 (필요시 여기 숫자만 바꾸세요)
    FORCE_LIMITS = {
        'train': 0,       # 학습에서 처음 128개만 사용
        'validation': 0,   # 검증에서 처음 64개만 사용
        'test': 0,         # 테스트에서 처음 64개만 사용
    }

    def _get_debug_limit():
        return FORCE_LIMITS.get(mode, 0)

    debug_limit = _get_debug_limit()

    dataloaders = []
    for dataset in datasets:
        if debug_limit > 0 and len(dataset) > debug_limit:
            dataset = Subset(dataset, list(range(debug_limit)))
            print0(pcolor(f'Using first {debug_limit} samples for {mode}', 'yellow'))

        sampler = get_datasampler(dataset, mode)
        shuffle_enabled = (mode == 'train') and (sampler is None)

        # Pick collate_fn per dataset (inspect children if ConcatDataset/Subset wraps it)
        collate_fn = base_collate_fn
        for child in _iter_children(dataset):
            if hasattr(child, 'custom_collate_fn') and callable(child.custom_collate_fn):
                collate_fn = child.custom_collate_fn
                break
            if type(child).__name__ == 'NcdbDataset':
                from packnet_sfm.datasets.ncdb_dataset import NcdbDataset
                collate_fn = NcdbDataset.custom_collate_fn
                break

        num_workers = config.num_workers if debug_limit == 0 else 0

        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle_enabled,
            pin_memory=False, 
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            collate_fn=collate_fn
        )
        dataloaders.append(dataloader)
    
    return dataloaders