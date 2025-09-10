# Copyright 2020 Toyota Research Institute.  All rights reserved.

import random
import torch

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.models.base_model import BaseModel
from packnet_sfm.models.model_utils import flip_batch_input, flip_output, upsample_output
from packnet_sfm.networks.depth.ResNetSAN01 import ResNetSAN01 # ✅ 추가
from packnet_sfm.networks.pose.PoseNet import PoseNet # ✅ 추가
from packnet_sfm.utils.misc import filter_dict


class SfmModel(BaseModel):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """
    def __init__(self, depth_net=None, pose_net=None,
                 rotation_mode='euler', flip_lr_prob=0.0,
                 upsample_depth_maps=False, **kwargs):
        super().__init__()

        # Initialize depth network
        if isinstance(depth_net, dict):
            self.depth_net = ResNetSAN01(**depth_net)
        else:
            self.depth_net = depth_net

        # Initialize pose network
        if isinstance(pose_net, dict):
            self.pose_net = PoseNet(**pose_net)
        else:
            self.pose_net = pose_net

        self.rotation_mode = rotation_mode
        self.flip_lr_prob = flip_lr_prob
        self.upsample_depth_maps = upsample_depth_maps

        self._network_requirements = [
            'depth_net',
            'pose_net',
        ]

    def add_depth_net(self, depth_net):
        """Add a depth network to the model"""
        self.depth_net = depth_net

    def add_pose_net(self, pose_net):
        """Add a pose network to the model"""
        self.pose_net = pose_net

    def depth_net_flipping(self, batch, flip, **kwargs):
        """
        Runs depth net with the option of flipping

        Parameters
        ----------
        batch : dict
            Input batch
        flip : bool
            True if the flip is happening

        Returns
        -------
        output : dict
            Dictionary with depth network output (e.g. 'inv_depths' and 'uncertainty')
        """
        # Which keys are being passed to the depth network
        batch_input = {key: batch[key] for key in filter_dict(batch, self._input_keys)}
        if flip:
            # Run depth network with flipped inputs
            output = self.depth_net(**flip_batch_input(batch_input), **kwargs)
            # Flip output back if training
            output = flip_output(output)
        else:
            # Run depth network
            output = self.depth_net(**batch_input, **kwargs)
        return output

    def compute_depth_net(self, batch, force_flip=False, **kwargs):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flag_flip_lr = random.random() < self.flip_lr_prob if self.training else force_flip
        output = self.depth_net_flipping(batch, flag_flip_lr, **kwargs)
        # If upsampling depth maps at training time
        if self.training and self.upsample_depth_maps:
            output = upsample_output(output, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return output

    def compute_pose_net(self, image, contexts):
        """Compute poses from image and a sequence of context images.
        Adds robustness for different collate formats and empty contexts."""
        # Handle None
        if contexts is None:
            raise ValueError("contexts is None but pose_net was invoked. Check dataloader and dataset back/forward context settings.")
        
        # If contexts came as a per-sample list (len == batch_size) where each element is a list of tensors
        # reshape to expected list-of-length-N_ctx with each element a (B,3,H,W) tensor.
        if isinstance(contexts, list) and len(contexts) > 0 and isinstance(contexts[0], list):
            try:
                # transpose list-of-lists
                transposed = list(zip(*contexts))  # list of tuples length N_ctx
                contexts = [torch.stack([frame for frame in t], dim=0) for t in transposed]
            except Exception as e:
                raise ValueError(f"Failed to transpose nested context lists: {e}")
        
        # Guard against empty context list (will cause torch.cat([]) inside pose net)
        if isinstance(contexts, (list, tuple)) and len(contexts) == 0:
            raise ValueError(
                "Empty rgb_context list passed to pose_net. This indicates that no valid context frames were built. "
                "Verify dataset configuration: train.back_context / train.forward_context (>0) and that sequences "
                "contain enough frames."
            )
        
        pose_vec = self.pose_net(image, contexts)
        if pose_vec.shape[1] == 0:
            raise ValueError("pose_net returned zero context predictions. Check pose network inputs.")
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]

    def forward(self, batch, return_logs=False, force_flip=False, **kwargs):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        force_flip : bool
            If true, force batch flipping for inverse depth calculation

        Returns
        -------
        output : dict
            Dictionary containing the output of depth and pose networks
        """
        # Generate inverse depth predictions
        depth_output = self.compute_depth_net(batch, force_flip=force_flip, **kwargs)
        # Generate pose predictions if available
        pose_output = None
        if 'rgb_context' in batch and self.pose_net is not None:
            pose_output = self.compute_pose_net(
                batch['rgb'], batch['rgb_context'])
        # Return output dictionary
        return {
            **depth_output,
            'poses': pose_output,
        }
