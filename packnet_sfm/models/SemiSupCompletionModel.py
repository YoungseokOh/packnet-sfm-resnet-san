# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as F  # ❗ 추가: F.interpolate를 사용하기 위해 임포트
from packnet_sfm.models.SelfSupModel import SfmModel, SelfSupModel
from packnet_sfm.losses.supervised_loss import SupervisedLoss
from packnet_sfm.models.model_utils import merge_outputs
from packnet_sfm.utils.depth import depth2inv
# ❗ YOLOv8SAN01 모델을 임포트하여 타입 체크에 사용
from packnet_sfm.networks.depth.YOLOv8SAN01 import YOLOv8SAN01


class SemiSupCompletionModel(SelfSupModel):
    """
    Semi-Supervised model for depth prediction and completion.

    Parameters
    ----------
    supervised_loss_weight : float
        Weight for the supervised loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_loss_weight=0.9, weight_rgbd=1.0, 
                 consistency_loss_weight=0.0, **kwargs): # ❗ 일관성 손실 가중치 추가
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        # If supervision weight is 0.0, use SelfSupModel directly
        assert 0. < supervised_loss_weight <= 1., "Model requires (0, 1] supervision"
        # Store weight and initializes supervised loss
        self.supervised_loss_weight = supervised_loss_weight
        self._supervised_loss = SupervisedLoss(**kwargs)
        # ❗ 일관성 손실 가중치 저장
        self.consistency_loss_weight = consistency_loss_weight

        # Pose network is only required if there is self-supervision
        if self.supervised_loss_weight == 1:
            self._network_requirements.remove('pose_net')
        # GT depth is only required if there is supervision
        if self.supervised_loss_weight > 0:
            self._train_requirements.append('gt_depth')

        self._input_keys = ['rgb', 'input_depth', 'intrinsics']

        self.weight_rgbd = weight_rgbd

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._supervised_loss.logs
        }

    def supervised_loss(self, inv_depths, gt_inv_depths,
                        return_logs=False, progress=0.0):
        """
        Calculates the supervised loss.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        gt_inv_depths : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth maps from the original image
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._supervised_loss(
            inv_depths, gt_inv_depths,
            return_logs=return_logs, progress=progress)

    def forward(self, batch, return_logs=False, progress=0.0, **kwargs):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        if not self.training:
            # If not training, no need for self-supervised loss
            return SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
        else:
            if self.supervised_loss_weight == 1.:
                # If no self-supervision, no need to calculate loss
                self_sup_output = SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
                loss = torch.tensor([0.]).type_as(batch['rgb'])
            else:
                # Otherwise, calculate and weight self-supervised loss
                self_sup_output = SelfSupModel.forward(
                    self, batch, return_logs=return_logs, progress=progress, **kwargs)
                loss = (1.0 - self.supervised_loss_weight) * self_sup_output['loss']
            # Calculate and weight supervised loss
            sup_output = self.supervised_loss(
                self_sup_output['inv_depths'], depth2inv(batch['depth']),
                return_logs=return_logs, progress=progress)
            loss += self.supervised_loss_weight * sup_output['loss']
            if 'inv_depths_rgbd' in self_sup_output:
                sup_output2 = self.supervised_loss(
                    self_sup_output['inv_depths_rgbd'], depth2inv(batch['depth']),
                    return_logs=return_logs, progress=progress)
                loss += self.weight_rgbd * self.supervised_loss_weight * sup_output2['loss']
                if 'depth_loss' in self_sup_output:
                    loss += self_sup_output['depth_loss']

                # ❗ [YOLOv8 전용] 일관성 손실(Consistency Loss) 추가
                # self.depth_net이 YOLOv8SAN01 인스턴스이고, 가중치가 0보다 클 때만 작동
                if self.training and isinstance(self.depth_net, YOLOv8SAN01) and self.consistency_loss_weight > 0:
                    # 1. RGB-Only 예측과 RGB+D 예측을 가져옴
                    pred_rgb = self_sup_output['inv_depths']
                    pred_rgbd = self_sup_output['inv_depths_rgbd']

                    # 2. 해상도를 맞춰 일관성 손실 계산
                    consistency_loss = 0.0
                    num_scales = min(len(pred_rgb), len(pred_rgbd))
                    if num_scales > 0:
                        for i in range(num_scales):
                            pred_rgb_i = pred_rgb[i]
                            pred_rgbd_i = pred_rgbd[i]
                            
                            # ❗ 해상도가 다를 경우, pred_rgb_i를 pred_rgbd_i의 크기에 맞춤
                            if pred_rgb_i.shape[-2:] != pred_rgbd_i.shape[-2:]:
                                pred_rgb_i = F.interpolate(
                                    pred_rgb_i, size=pred_rgbd_i.shape[-2:], 
                                    mode='bilinear', align_corners=False
                                )
                            
                            # pred_rgbd의 그래디언트가 pred_rgb로 흐르지 않도록 detach() 사용
                            consistency_loss += torch.abs(pred_rgb_i - pred_rgbd_i.detach()).mean()
                        
                        consistency_loss /= num_scales

                        # 3. 기존 손실에 추가
                        loss += self.consistency_loss_weight * consistency_loss
                        
                        # 로깅
                        if return_logs:
                            self_sup_output['metrics']['consistency_loss'] = consistency_loss
            # Merge and return outputs
            return {
                'loss': loss,
                **merge_outputs(self_sup_output, sup_output),
            }
