# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.SfmModel import SfmModel
from packnet_sfm.losses.multiview_photometric_loss import MultiViewPhotometricLoss
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
        # Initializes the photometric loss
        self._photometric_loss = MultiViewPhotometricLoss(**kwargs)

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs
        }

    def self_supervised_loss(self, image, ref_images, inv_depths, poses,
                             intrinsics_list, distortion_coeffs, # Changed intrinsics to intrinsics_list, added distortion_coeffs
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
        # Prepare intrinsics dictionary for MultiViewPhotometricLoss
        # MultiViewPhotometricLoss expects a dict for intrinsics
        # We are passing the same intrinsics for both original and reference cameras for now
        # If ref_intrinsics are different, they should be passed separately from batch
        
        # Ensure distortion_coeffs are on the correct device and have correct batch size
        # This is handled in NcdbDataset, but good to be explicit if needed
        
        # For now, we assume intrinsics_list and distortion_coeffs are for the current image
        # and will be used for both original and reference cameras in the photometric loss.
        
        # The MultiViewPhotometricLoss expects a dict for intrinsics,
        # which is already prepared in NcdbDataset as 'distortion_coeffs'.
        # We just need to pass it correctly.
        
        return self._photometric_loss(
            image, ref_images, inv_depths, distortion_coeffs, distortion_coeffs, poses, # Passed distortion_coeffs for both
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
        # Calculate predicted depth and pose output
        output = super().forward(batch, return_logs=return_logs)
        if not self.training:
            # If not training, no need for self-supervised loss
            return output
        else:
            # Otherwise, calculate self-supervised loss
            self_sup_output = self.self_supervised_loss(
                batch['rgb_original'], batch['rgb_context_original'],
                output['inv_depths'], output['poses'],
                batch['intrinsics'], # This is the full VADAS intrinsic list
                batch['distortion_coeffs'], # Added distortion_coeffs
                return_logs=return_logs, progress=progress,
                mask=batch.get('mask', None) # Pass mask if available
            )
            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'],
                **merge_outputs(output, self_sup_output),
            }