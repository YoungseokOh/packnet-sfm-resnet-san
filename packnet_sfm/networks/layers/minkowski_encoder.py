# Copyright 2020 Toyota Research Institute.  All rights reserved.

import MinkowskiEngine as ME
import torch.nn as nn
import torch

from packnet_sfm.networks.layers.minkowski import \
    sparsify_depth, densify_features, densify_add_features_unc, map_add_features


class MinkConv2D(nn.Module):
    """
    Minkowski Convolutional Block

    Parameters
    ----------
    in_planes : number of input channels
    out_planes : number of output channels
    kernel_size : convolutional kernel size
    stride : convolutional stride
    with_uncertainty : with uncertainty or now
    add_rgb : add RGB information as channels
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride,
                 with_uncertainty=False, add_rgb=False):
        super().__init__()
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_planes, out_planes * 2, kernel_size=kernel_size, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(out_planes * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_planes * 2, out_planes * 2, kernel_size=kernel_size, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(out_planes * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_planes * 2, out_planes, kernel_size=kernel_size, stride=1, dimension=2),
        )

        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_planes, out_planes * 2, kernel_size=kernel_size, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(out_planes * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_planes * 2, out_planes, kernel_size=kernel_size, stride=1, dimension=2),
        )

        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_planes, out_planes, kernel_size=kernel_size, stride=1, dimension=2),
        )

        self.layer_final = nn.Sequential(
            ME.MinkowskiBatchNorm(out_planes),
            ME.MinkowskiReLU(inplace=True)
        )
        self.pool = None if stride == 1 else ME.MinkowskiMaxPooling(3, stride, dimension=2)

        self.add_rgb = add_rgb
        self.with_uncertainty = with_uncertainty
        if with_uncertainty:
            self.unc_layer = nn.Sequential(
                ME.MinkowskiConvolution(
                    out_planes, 1, kernel_size=3, stride=1, dimension=2),
                ME.MinkowskiSigmoid()
            )

    def forward(self, x):
        """
        Processes sparse information

        Parameters
        ----------
        x : Sparse tensor

        Returns
        -------
        Processed tensor
        """
        if self.pool is not None:
            x = self.pool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        return None, self.layer_final(x1 + x2 + x3)


class MinkowskiEncoder(nn.Module):
    """
    Depth completion Minkowski Encoder with optional Depth-aware FiLM

    Parameters
    ----------
    channels : list[int]
        number of depth feature channels per stage
    rgb_channels : list[int] or None
        list of RGB channels for each scale (for FiLM); if None, FiLM is disabled
    with_uncertainty : bool
        predict uncertainty
    add_rgb : bool
        fuse RGB into sparse features
    """
    def __init__(self, channels, rgb_channels=None, with_uncertainty=False, add_rgb=False):
        super().__init__()
        self.mconvs = nn.ModuleList()
        kernel_sizes = [5, 5] + [3] * (len(channels) - 1)
        
        # depth conv stages
        self.mconvs.append(
            MinkConv2D(1, channels[0], kernel_sizes[0], 2,
                       with_uncertainty=with_uncertainty)
        )
        for i in range(len(channels) - 1):
            self.mconvs.append(
                MinkConv2D(channels[i], channels[i+1], kernel_sizes[i+1], 2,
                           with_uncertainty=with_uncertainty)
            )

        # 🆕 optional Depth-aware FiLM generator
        self.rgb_channels = rgb_channels
        self.film_generators = nn.ModuleDict()
        
        if rgb_channels is not None:
            # rgb_channels가 리스트인 경우 각 스케일별로, 아니면 모든 스케일에 동일하게 적용
            if isinstance(rgb_channels, (list, tuple)):
                for i, (depth_ch, rgb_ch) in enumerate(zip(channels, rgb_channels)):
                    if rgb_ch > 0:  # RGB 채널이 있는 스케일만 FiLM 생성기 생성
                        self.film_generators[str(i)] = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(depth_ch, rgb_ch * 2, kernel_size=1)
                        )
            else:
                # 단일 값인 경우 첫 번째 스케일에만 적용
                self.film_generators['0'] = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels[0], rgb_channels * 2, kernel_size=1)
                )

        self.d = self.n = self.shape = 0
        self.with_uncertainty = with_uncertainty
        self.add_rgb = add_rgb

    def prep(self, d):
        self.d = sparsify_depth(d)
        self.shape = d.shape
        self.n = 0

    def forward(self, x=None):
        # 1) MinkConv 단계
        unc, self.d = self.mconvs[self.n](self.d)
        current_scale = self.n
        self.n += 1

        # 2) depth -> dense features
        if self.with_uncertainty:
            out = densify_add_features_unc(x, unc * self.d, unc, self.shape)
        else:
            out = densify_features(self.d, self.shape)

        # 3) RGB fusion
        if self.add_rgb:
            self.d = map_add_features(x, self.d)

        # 🆕 4) 현재 스케일에 해당하는 FiLM parameter 생성
        if self.rgb_channels is not None and str(current_scale) in self.film_generators:
            # out: [B, C_depth, H, W]
            params = self.film_generators[str(current_scale)](out)  # [B, 2*rgb_channels, 1, 1]
            gamma, beta = params.chunk(2, dim=1)  # each: [B, rgb_channels, 1, 1]
            return out, gamma, beta

        return out
