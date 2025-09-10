# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.SfmModel import SfmModel
from packnet_sfm.losses.multiview_photometric_loss import MultiViewPhotometricLoss
from packnet_sfm.losses.fisheye_multiview_photometric_loss import FisheyeMultiViewPhotometricLoss  # Added
from packnet_sfm.models.model_utils import merge_outputs

class SelfSupModel(SfmModel):
    """
    Model that inherits a depth and pose network from SfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self, **kwargs):
        # Initializes SfmModel
        super().__init__(**kwargs)
        # Flag for fisheye loss usage
        self.use_fisheye_loss = kwargs.get('loss', {}).get('use_fisheye_loss', False)

        # Instantiate appropriate photometric loss
        if self.use_fisheye_loss:
            # Map config keys to fisheye loss expected names
            fisheye_args = {
                'num_scales': 1,  # 🔧 LUT 크기 일치를 위해 1스케일로 고정
                'ssim_loss_weight': kwargs.get('ssim_loss_weight', 0.85),
                'smooth_loss_weight': kwargs.get('smooth_loss_weight', 0.001),
                'photometric_reduce_op': kwargs.get('photometric_reduce_op', 'mean'),
                'clip_loss': kwargs.get('clip_loss', 0.0),
                'automask_loss': kwargs.get('automask_loss', False),
                'lut_path': kwargs.get('fisheye_lut_path', None),
                'padding_mode': kwargs.get('padding_mode', 'zeros'),
                'disp_norm': kwargs.get('disp_norm', True),
                'C1': kwargs.get('C1', 1e-4),
                'C2': kwargs.get('C2', 9e-4),
            }
            self._photometric_loss = FisheyeMultiViewPhotometricLoss(**fisheye_args)
        else:
            self._photometric_loss = MultiViewPhotometricLoss(**kwargs)

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs
        }

    def self_supervised_loss(self, image, ref_images, inv_depths, poses,
                             intrinsics_list, distortion_coeffs,
                             return_logs=False, progress=0.0, mask=None):
        """
        Calculates the self-supervised photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_images : list of torch.Tensor [B,3,H,W]
            Reference images from context
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        poses : list of Pose
            List containing predicted poses between original and context images
        intrinsics_list : torch.Tensor [B,N] (VADAS intrinsic list)
            Camera intrinsics (full VADAS intrinsic list)
        distortion_coeffs : dict
            Dictionary containing fisheye camera intrinsic parameters ('k', 's', 'div', 'ux', 'uy').
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage
        mask : torch.Tensor [B,1,H,W], optional
            Binary mask for valid pixels

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        # Check if we have valid distortion coeffs for fisheye
        use_fisheye = self.use_fisheye_loss and distortion_coeffs is not None
        
        # Additional validation for fisheye coeffs
        if use_fisheye and isinstance(distortion_coeffs, dict):
            required_keys = {'k', 's', 'div', 'ux', 'uy'}
            if not all(key in distortion_coeffs for key in required_keys):
                use_fisheye = False
                if not hasattr(self, '_fisheye_disabled_warn_logged'):
                    print('[SelfSupModel] Fisheye requested but distortion coeffs incomplete; falling back to pinhole.')
                    self._fisheye_disabled_warn_logged = True
        elif use_fisheye:
            use_fisheye = False
            if not hasattr(self, '_fisheye_disabled_warn_logged'):
                print('[SelfSupModel] Fisheye requested but distortion coeffs missing; falling back to pinhole.')
                self._fisheye_disabled_warn_logged = True
        
        if use_fisheye:
            return self._photometric_loss(
                image, ref_images, inv_depths, distortion_coeffs, poses,
                return_logs=return_logs, progress=progress)
        else:
            # Fallback to pinhole model
            return self._photometric_loss(
                image, ref_images, inv_depths, intrinsics_list, intrinsics_list, poses,
                return_logs=return_logs, progress=progress, mask=mask)

    def forward(self, batch, return_logs=False, progress=0.0):
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
        output = super().forward(batch, return_logs=return_logs)
        if not self.training:
            return output
        rgb_target = batch.get('rgb_original', batch.get('rgb'))
        ref_context = batch.get('rgb_context_original', batch.get('rgb_context', [])) or []
        self_sup_output = self.self_supervised_loss(
            rgb_target,
            ref_context,
            output['inv_depths'], output['poses'],
            batch.get('intrinsics', None),
            batch.get('distortion_coeffs', None),
            return_logs=return_logs, progress=progress,
            mask=batch.get('mask', None)
        )
        return {
            'loss': self_sup_output['loss'],
            **merge_outputs(output, self_sup_output),
        }