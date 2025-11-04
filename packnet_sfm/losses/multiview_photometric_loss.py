# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.geometry.camera import FisheyeCamera # Changed from Camera to FisheyeCamera
from packnet_sfm.geometry.camera_utils import view_synthesis
from packnet_sfm.utils.depth import calc_smoothness, inv2depth
from packnet_sfm.utils.post_process_depth import sigmoid_to_depth_linear  # 🆕 For sigmoid → depth conversion
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling

########################################################################################################################

def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim

########################################################################################################################

class MultiViewPhotometricLoss(LossBase):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scalesto consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    occ_reg_weight : float
        Weight for the occlusion regularization loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1,C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
    disp_norm : bool
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, num_scales=4, ssim_loss_weight=0.85, occ_reg_weight=0.1, smooth_loss_weight=0.1,
                 C1=1e-4, C2=9e-4, photometric_reduce_op='mean', disp_norm=True, clip_loss=0.5,
                 progressive_scaling=0.0, padding_mode='zeros',
                 automask_loss=False, min_depth=0.05, max_depth=80.0, **kwargs):
        super().__init__()
        self.n = num_scales
        self.progressive_scaling = progressive_scaling
        self.ssim_loss_weight = ssim_loss_weight
        self.occ_reg_weight = occ_reg_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.photometric_reduce_op = photometric_reduce_op
        self.disp_norm = disp_norm
        self.clip_loss = clip_loss
        self.padding_mode = padding_mode
        self.automask_loss = automask_loss
        # 🆕 Store depth range for sigmoid → depth conversion
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'

########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'num_scales': self.n,
        }

########################################################################################################################

    def warp_ref_image(self, inv_depths, ref_image, intrinsics, ref_intrinsics, poses, image_size): # Changed K, ref_K to intrinsics, ref_intrinsics, added image_size
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        intrinsics : dict
            Original camera intrinsics (FisheyeCamera format)
        ref_intrinsics : dict
            Reference camera intrinsics (FisheyeCamera format)
        poses : Pose
            Original -> Reference camera transformation
        image_size : tuple (H, W)
            The size of the image.

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image in the original frame of reference
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor_w = DW / float(W)
            scale_factor_h = DH / float(H)
            
            # Create FisheyeCamera instances
            # Note: intrinsics['k'], intrinsics['s'], etc. are already tensors from NcdbDataset
            # We need to ensure they are on the correct device and have the correct batch size
            
            # For original camera
            scaled_intrinsics = {
                'k': intrinsics['k'].clone().to(device),
                's': intrinsics['s'].clone().to(device),
                'div': intrinsics['div'].clone().to(device),
                'ux': (intrinsics['ux'].clone() + 0.5) * scale_factor_w - 0.5,
                'uy': (intrinsics['uy'].clone() + 0.5) * scale_factor_h - 0.5,
            }
            cams.append(FisheyeCamera(intrinsics=scaled_intrinsics, image_size=(DH, DW)).to(device))

            # For reference camera
            scaled_ref_intrinsics = {
                'k': ref_intrinsics['k'].clone().to(device),
                's': ref_intrinsics['s'].clone().to(device),
                'div': ref_intrinsics['div'].clone().to(device),
                'ux': (ref_intrinsics['ux'].clone() + 0.5) * scale_factor_w - 0.5,
                'uy': (ref_intrinsics['uy'].clone() + 0.5) * scale_factor_h - 0.5,
            }
            ref_cams.append(FisheyeCamera(intrinsics=scaled_ref_intrinsics, Tcw=poses, image_size=(DH, DW)).to(device))
        
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = [view_synthesis(
            ref_images[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode) for i in range(self.n)]
        # Return warped reference image
        return ref_warped

########################################################################################################################

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images, masks_scaled=None):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales
        masks_scaled : list of torch.Tensor [B,1,H,W], optional
            List of binary masks for valid pixels, scaled to match image scales

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(self.n)]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(self.n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        
        # Apply mask to photometric loss if provided
        if masks_scaled is not None:
            for i in range(self.n):
                # Ensure mask has 3 channels to multiply with RGB loss
                # Or, if loss is already meaned to 1 channel, ensure mask is 1 channel
                if photometric_loss[i].shape[1] == 3 and masks_scaled[i].shape[1] == 1:
                    mask_for_loss = masks_scaled[i].expand_as(photometric_loss[i])
                else:
                    mask_for_loss = masks_scaled[i]
                photometric_loss[i] = photometric_loss[i] * mask_for_loss

        # Return total photometric loss
        return photometric_loss

    def reduce_photometric_loss(self, photometric_losses):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """
        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif self.photometric_reduce_op == 'min':
                return torch.cat(losses, 1).min(1, True)[0].mean()
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))
        # Reduce photometric loss
        photometric_loss = sum([reduce_function(photometric_losses[i])
                                for i in range(self.n)]) / self.n
        # Store and return reduced photometric loss
        self.add_metric('photometric_loss', photometric_loss)
        return photometric_loss

########################################################################################################################

    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales

        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n
        # Apply smoothness loss weight
        smoothness_loss = self.smooth_loss_weight * smoothness_loss
        # Store and return smoothness loss
        self.add_metric('smoothness_loss', smoothness_loss)
        return smoothness_loss

########################################################################################################################

    def forward(self, image, context, inv_depths,
                intrinsics, ref_intrinsics, poses, return_logs=False, progress=0.0, mask=None): # Changed K, ref_K to intrinsics, ref_intrinsics
        """
        Calculates training photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        context : list of torch.Tensor [B,3,H,W]
            Context containing a list of reference images
        inv_depths : list of torch.Tensor [B,1,H,W]
            🆕 Now receives sigmoid outputs [0, 1] for all scales (not actual inv_depths!)
        intrinsics : dict
            Original camera intrinsics (FisheyeCamera format)
        ref_intrinsics : dict
            Reference camera intrinsics (FisheyeCamera format)
        poses : list of Pose
            Camera transformation between original and context
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage
        mask : torch.Tensor [B,1,H,W], optional
            Binary mask for valid pixels

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # 🆕 Convert sigmoid outputs to depth using Linear transformation
        # inv_depths are actually sigmoid outputs [0, 1] now
        sigmoid_outputs = inv_depths
        depths = [sigmoid_to_depth_linear(sigmoid_outputs[i], self.min_depth, self.max_depth) 
                  for i in range(len(sigmoid_outputs))]
        
        # Convert depth back to inv_depth for view synthesis (required by existing code)
        inv_depths = [1.0 / (depths[i] + 1e-8) for i in range(len(depths))]
        
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)
        # Loop over all reference images
        photometric_losses = [[] for _ in range(self.n)]
        images = match_scales(image, inv_depths, self.n)
        # Scale mask if provided
        if mask is not None:
            masks_scaled = match_scales(mask, inv_depths, self.n, mode='nearest', align_corners=None)
        else:
            masks_scaled = [None] * self.n

        # Get image size from the first image in the batch
        _, _, H, W = image.shape
        image_size = (H, W)

        for j, (ref_image, pose) in enumerate(zip(context, poses)):
            # Calculate warped images
            ref_warped = self.warp_ref_image(inv_depths, ref_image, intrinsics, ref_intrinsics, pose, image_size) # Pass image_size
            # Calculate and store image loss
            photometric_loss = self.calc_photometric_loss(ref_warped, images, masks_scaled) # Pass masks_scaled
            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i])
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(ref_image, inv_depths, self.n)
                unwarped_image_loss = self.calc_photometric_loss(ref_images, images, masks_scaled) # Pass masks_scaled
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i])
        # Calculate reduced photometric loss
        loss = self.reduce_photometric_loss(photometric_losses)
        # Include smoothness loss if requested
        # 🆕 Use sigmoid outputs for smoothness (edge-aware smoothness in sigmoid space)
        if self.smooth_loss_weight > 0.0:
            loss += self.calc_smoothness_loss(sigmoid_outputs, images)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }

########################################################################################################################