import torch
import torch.nn as nn

from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth
from functools import partial


class ResNetSAN01(nn.Module):
    """
    ResNet-18 based SAN network for depth estimation with sparse depth input
    
    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Version string (A for concat, B for add - kept for compatibility)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, **kwargs):
        super().__init__()
        
        self.version = version[1:] if version else 'A'  # Extract A or B from version
        
        # ResNet-18 encoder
        self.encoder = ResnetEncoder(num_layers=18, pretrained=True)
        
        # Standard depth decoder
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        
        # Minkowski encoder for sparse depth processing
        # ResNet-18 encoder channels: [64, 64, 128, 256, 512]
        self.mconvs = MinkowskiEncoder(
            self.encoder.num_ch_enc, 
            with_uncertainty=False
        )
        
        # Learnable weights and biases for feature fusion (5 scales like PackNet)
        self.weight = torch.nn.parameter.Parameter(
            torch.ones(5), requires_grad=True
        )
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(5), requires_grad=True
        )
        
        self.init_weights()

    def init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def run_network(self, rgb, input_depth=None):
        """
        Run the network and return inverse depth maps
        
        Parameters
        ----------
        rgb : torch.Tensor [B,3,H,W]
            RGB input image
        input_depth : torch.Tensor [B,1,H,W], optional
            Sparse depth input
            
        Returns
        -------
        inv_depths : dict
            Dictionary with inverse depth maps at different scales
        skip_features : list of torch.Tensor
            Skip connection features
        """
        # Encode RGB features
        skip_features = self.encoder(rgb)
        
        # If sparse depth is provided, fuse with RGB features
        if input_depth is not None:
            # Prepare sparse depth features
            self.mconvs.prep(input_depth)
            
            # Fuse features at each scale (PackNet-SAN style)
            fused_features = []
            for i, feat in enumerate(skip_features):
                sparse_feat = self.mconvs(feat)
                fused_feat = (feat * self.weight[i].view(1, 1, 1, 1) + 
                             sparse_feat + 
                             self.bias[i].view(1, 1, 1, 1))
                fused_features.append(fused_feat)
            skip_features = fused_features
        
        # Decode to get inverse depth maps
        inv_depths_dict = self.decoder(skip_features)
        
        # Convert dict to list format (PackNet-SAN compatible)
        if self.training:
            # Training: return all scales as list
            inv_depths = [
                inv_depths_dict[("disp", 0)],  # scale 0 (highest resolution)
                inv_depths_dict[("disp", 1)],  # scale 1
                inv_depths_dict[("disp", 2)],  # scale 2
                inv_depths_dict[("disp", 3)],  # scale 3
            ]
        else:
            # Inference: return only highest resolution
            inv_depths = [inv_depths_dict[("disp", 0)]]
        
        return inv_depths, skip_features

    def forward(self, rgb, input_depth=None, **kwargs):
        """
        Forward pass - matches PackNet-SAN interface exactly
        """
        if not self.training:
            inv_depths, _ = self.run_network(rgb, input_depth)
            return {'inv_depths': inv_depths}

        output = {}
        
        # RGB-only forward pass
        inv_depths_rgb, skip_feat_rgb = self.run_network(rgb)
        output['inv_depths'] = inv_depths_rgb
        
        # If no sparse depth provided, return RGB-only results
        if input_depth is None:
            return {'inv_depths': inv_depths_rgb}
        
        # RGB+D forward pass
        inv_depths_rgbd, skip_feat_rgbd = self.run_network(rgb, input_depth)
        output['inv_depths_rgbd'] = inv_depths_rgbd
        
        # Compute consistency loss between RGB and RGB+D features (PackNet-SAN style)
        loss = sum([((feat_rgbd.detach() - feat_rgb) ** 2).mean()
                   for feat_rgbd, feat_rgb in zip(skip_feat_rgbd, skip_feat_rgb)]) / len(skip_feat_rgbd)
        output['depth_loss'] = loss
        
        return output