# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache
import torch
import torch.nn as nn
import numpy as np  # 🆕 추가

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_utils import scale_intrinsics
from packnet_sfm.utils.image import image_grid

########################################################################################################################

import sys # For sys.float_info.epsilon

class Camera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """
    def __init__(self, K, Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        K : torch.Tensor [3,3]
            Camera intrinsics
        Tcw : Pose
            Camera -> World pose transformation
        """
        super().__init__()
        self.K = K
        self.Tcw = Pose.identity(len(K)) if Tcw is None else Tcw

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.K)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.K = self.K.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self


    @property
    def fx(self):
        """Focal length in x"""
        return self.K[:, 0, 0]

    @property
    def fy(self):
        """Focal length in y"""
        return self.K[:, 1, 1]

    @property
    def cx(self):
        """Principal point in x"""
        return self.K[:, 0, 2]

    @property
    def cy(self):
        """Principal point in y"""
        return self.K[:, 1, 2]

    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

    @property
    @lru_cache()
    def Kinv(self):
        """Inverse intrinsics (for lifting)"""
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1. * self.cx / self.fx
        Kinv[:, 1, 2] = -1. * self.cy / self.fy
        return Kinv


    def scaled(self, x_scale, y_scale=None):
        """
        Returns a scaled version of the camera (changing intrinsics)

        Parameters
        ----------
        x_scale : float
            Resize scale in x
        y_scale : float
            Resize scale in y. If None, use the same as x_scale

        Returns
        -------
        camera : Camera
            Scaled version of the current cmaera
        """
        # If single value is provided, use for both dimensions
        if y_scale is None:
            y_scale = x_scale
        # If no scaling is necessary, return same camera
        if x_scale == 1. and y_scale == 1.:
            return self
        # Scale intrinsics and return new camera with same Pose
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return Camera(K, Tcw=self.Tcw)


    def reconstruct(self, depth, frame='w'):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """
        B, C, H, W = depth.shape
        assert C == 1

        # Create flat index grid
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
        flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

        # Estimate the outward rays in the camera frame
        xnorm = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)
        # Scale rays to metric depth
        Xc = xnorm * depth

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            return self.Twc @ Xc
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project(self, X, frame='w'):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """
        B, C, H, W = X.shape
        assert C == 3

        # Project 3D points onto the camera image plane
        if frame == 'c':
            Xc = self.K.bmm(X.view(B, 3, -1))
        elif frame == 'w':
            Xc = self.K.bmm((self.Tcw @ X).view(B, 3, -1))
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

        # Normalize points
        X = Xc[:, 0]
        Y = Xc[:, 1]
        Z = Xc[:, 2].clamp(min=1e-5)
        Xnorm = 2 * (X / Z) / (W - 1) - 1.
        Ynorm = 2 * (Y / Z) / (H - 1) - 1.

        # Clamp out-of-bounds pixels
        # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
        # Xnorm[Xmask] = 2.
        # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
        # Ynorm[Ymask] = 2.

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)

########################################################################################################################

class FisheyeCamera(nn.Module):
    """
    Simplified fisheye camera for VADAS model - focused on core functionality
    """
    def __init__(self, intrinsics, Tcw=None, image_size=None):
        super().__init__()
        self.intrinsics = intrinsics
        self.Tcw = Pose.identity(1) if Tcw is None else Tcw
        self.image_size = image_size
        
        # Extract individual parameters
        if isinstance(intrinsics, dict):
            self.k = intrinsics['k']
            self.s = intrinsics['s'] 
            self.div = intrinsics['div']
            self.ux = intrinsics['ux']
            self.uy = intrinsics['uy']
        else:
            raise ValueError("Intrinsics must be a dictionary with keys: k, s, div, ux, uy")

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.k)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.k = self.k.to(*args, **kwargs)
        self.s = self.s.to(*args, **kwargs)
        self.div = self.div.to(*args, **kwargs)
        self.ux = self.ux.to(*args, **kwargs)
        self.uy = self.uy.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

    def reconstruct(self, depth, frame='w'):
        """
        🎯 Simplified but accurate fisheye reconstruction using improved polynomial inversion
        Based on fisheye_test-master approach but simplified
        """
        B, _, H, W = depth.shape
        device = depth.device
        
        # Create pixel grid
        i_range = torch.arange(H, dtype=torch.float32, device=device)
        j_range = torch.arange(W, dtype=torch.float32, device=device)
        ii, jj = torch.meshgrid(i_range, j_range)  # Remove indexing parameter for compatibility
        flat_grid = torch.stack([jj.flatten(), ii.flatten()], dim=0)  # [2, HW]
        flat_grid = flat_grid.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, HW]

        u_pixels = flat_grid[:, 0, :]  # [B, HW]
        v_pixels = flat_grid[:, 1, :]  # [B, HW]

        # Prepare intrinsics with proper broadcasting
        def _shape_param(p):
            if isinstance(p, torch.Tensor):
                if p.dim() == 0:
                    return p.view(1, 1, 1).expand(B, 1, 1)
                elif p.dim() == 1 and len(p) == B:
                    return p.view(B, 1, 1)
                elif p.dim() == 1 and len(p) == 1:
                    return p.view(1, 1, 1).expand(B, 1, 1)
                elif p.dim() == 2 and p.shape[0] == B:
                    return p.view(B, 1, 1)
                else:
                    # Handle multi-element tensors safely
                    return p.view(-1, 1, 1).expand(B, 1, 1)
            else:
                # Handle scalar values
                return torch.tensor(float(p), device=device).view(1, 1, 1).expand(B, 1, 1)

        ux = _shape_param(self.ux)
        uy = _shape_param(self.uy)
        s = _shape_param(self.s)
        div = _shape_param(self.div)

        # Distorted coordinates (fisheye_test-master style)
        x_dist = (u_pixels.unsqueeze(1) - ux) / s  # [B, 1, HW]
        y_dist = (v_pixels.unsqueeze(1) - uy) / div

        # Distorted radius
        r_d = torch.sqrt(x_dist ** 2 + y_dist ** 2)

        # 🎯 Improved polynomial inversion (key improvement from fisheye_test-master)
        k_coeffs = self.k  # [B, 7] or [7]
        if k_coeffs.dim() == 1:
            k_coeffs = k_coeffs.unsqueeze(0).expand(B, -1)
        
        # Initial guess: theta ≈ r_d
        theta = r_d.squeeze(1)  # [B, HW]
        
        # Newton-Raphson refinement (2 iterations for good accuracy vs speed)
        for _ in range(2):
            # f(θ) = k0*θ + k1*θ² + ... + k6*θ⁷ - r_d
            theta_powers = torch.stack([theta**(i+1) for i in range(7)], dim=-1)  # [B, HW, 7]
            f_theta = torch.sum(k_coeffs.unsqueeze(1) * theta_powers, dim=-1) - r_d.squeeze(1)
            
            # f'(θ) = k0 + 2*k1*θ + ... + 7*k6*θ⁶
            df_theta = torch.sum(k_coeffs.unsqueeze(1) * torch.stack([
                (i+1) * theta**i for i in range(7)
            ], dim=-1), dim=-1)
            
            # Newton update with safety
            df_theta = torch.clamp(df_theta, min=1e-8)
            theta = theta - f_theta / df_theta
            theta = torch.clamp(theta, min=0, max=np.pi/2)  # Reasonable bounds

        # Convert to Cartesian (fisheye_test-master approach)
        r_world = torch.tan(theta)  # Use tan(theta) instead of sin/cos
        r_d_safe = torch.clamp(r_d.squeeze(1), min=1e-8)
        scale = r_world / r_d_safe
        
        x_norm = scale * x_dist.squeeze(1)
        y_norm = scale * y_dist.squeeze(1)
        
        # 3D points
        depth_flat = depth.view(B, -1)
        x_flat = x_norm * depth_flat
        y_flat = y_norm * depth_flat
        z_flat = depth_flat

        # Reshape to proper format
        X_cam = torch.stack([x_flat, y_flat, z_flat], dim=1)  # [B, 3, HW]
        X_cam = X_cam.view(B, 3, H, W)

        # Transform to world coordinates if needed
        if frame == 'w':
            X_cam_hom = torch.cat([X_cam.view(B, 3, -1), 
                                 torch.ones(B, 1, H*W, device=device)], dim=1)
            X_world_hom = self.Twc.mat @ X_cam_hom  # Use .mat instead of .matrix
            X_world = X_world_hom[:, :3, :].view(B, 3, H, W)
            return X_world
        else:
            return X_cam

    def project(self, X, frame='w'):
        """
        Simplified fisheye projection (VADAS model)
        """
        if len(X.shape) == 4:
            B, C, H, W = X.shape
            assert C == 3
            Xc4d = self.Tcw @ X if frame == 'w' else X
            Xc = Xc4d.view(B, 3, -1)  # [B,3,HW]
        elif len(X.shape) == 3:
            B, C, N = X.shape
            assert C == 3
            H, W = self.image_size
            X4d = X.view(B, 3, H, W)
            Xc4d = self.Tcw @ X4d if frame == 'w' else X4d
            Xc = X4d.view(B, 3, -1)
        else:
            raise ValueError('Input X must be [B,3,H,W] or [B,3,N]')
        
        HW = Xc.shape[-1]
        
        # Normalize
        Z = Xc[:, 2].clamp(min=1e-8)      # [B,HW]
        x_norm = Xc[:, 0] / Z             # [B,HW]
        y_norm = Xc[:, 1] / Z             # [B,HW]
        
        # Radial quantities
        r = torch.sqrt(x_norm ** 2 + y_norm ** 2)  # [B,HW]
        theta = torch.atan(r)
        
        # Polynomial distortion (VADAS model)
        if self.k.dim() == 1:
            k = self.k.view(1, -1).expand(B, -1)
        else:
            k = self.k
        
        theta_poly = k[:, 0:1]  # [B,1]
        for i in range(1, 7):
            theta_powers = torch.pow(theta, i)  # [B,HW]
            theta_poly = theta_poly + k[:, i:i+1] * theta_powers
        
        r_d = theta_poly if theta_poly.shape[-1] == HW else theta_poly.repeat(1, HW)
        r_safe = torch.where(r > 1e-8, r, torch.full_like(r, 1e-8))
        scale = r_d / r_safe  # [B,HW]
        
        x_dist = scale * x_norm
        y_dist = scale * y_norm
        
        # Intrinsic params to shape [B]
        def _vec(p):
            return p if p.dim() == 1 else p.view(p.shape[0])
        
        s = _vec(self.s)
        div = _vec(self.div)
        ux = _vec(self.ux)
        uy = _vec(self.uy)
        
        # Pixel coordinates
        u = s[:, None] * x_dist + ux[:, None]   # [B,HW]
        v = div[:, None] * y_dist + uy[:, None] # [B,HW]
        
        # Normalize to [-1,1]
        u_norm = 2 * u / (W - 1) - 1.
        v_norm = 2 * v / (H - 1) - 1.
        
        coords = torch.stack([u_norm, v_norm], dim=-1)  # [B,HW,2]
        if len(X.shape) == 4:
            return coords.view(B, H, W, 2)
        return coords