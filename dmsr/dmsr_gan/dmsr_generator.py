#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:18:31 2024

@author: brennan
"""

import torch
import torch.nn as nn

from torch import randn
from torch.nn.functional import interpolate
from ..field_operations.resize import crop


class DMSRGenerator(nn.Module):
    
    def __init__(self,
                 grid_size,
                 channels, 
                 scale_factor=8,
                 **kwargs
        ):
        super().__init__()
        
        self.grid_size = grid_size
        self.scale_factor = scale_factor

        self.block0 = nn.Sequential(
            nn.Conv3d(3, channels, 3),
            nn.PReLU(),
        )
        
        scale = 1
        curr_chan = channels
        next_chan = curr_chan // 2
        N = grid_size - 2
        noise_shapes = []
        
        self.blocks = nn.ModuleList()
        while scale < scale_factor:
            self.blocks.append(
                HBlock(curr_chan, next_chan)
            )
            
            scale *= 2
            curr_chan = next_chan
            next_chan = curr_chan // 2
            noise_shapes.append((N, 2 * N - 2))
            N = 2 * N - 4
            
        self.output_size = N
        self.noise_shapes = noise_shapes


    def forward(self, x, z):
        y = crop(x, 1)
        x = self.block0(x)

        for block, noise in zip(self.blocks, z):
            x, y = block(x, y, noise)
        
        return y
    
    
    def sample_latent_space(self, batch_size, device):
        
        latent_variable = [None] * len(self.noise_shapes)
        for i, (shape_A, shape_B) in enumerate(self.noise_shapes):
            shape_A = (batch_size, 1) + 3 * (shape_A,)
            shape_B = (batch_size, 1) + 3 * (shape_B,)
            noise = randn(shape_A).to(device), randn(shape_B).to(device)
            latent_variable[i] = noise
            
        return latent_variable


class HBlock(nn.Module):
    """
    The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n
    cat_noise: concatenate noise if True, otherwise add noise

    Notes
    -----
    next_size = 2 * prev_size - 4
    """

    def __init__(self, curr_chan, next_chan):
        super().__init__()

        self.conv_A = nn.Sequential(
            nn.Conv3d(curr_chan + 1, next_chan, 3),
            nn.PReLU()
        )
        
        self.conv_B = nn.Sequential(
            nn.Conv3d(next_chan + 1, next_chan, 3),
            nn.PReLU()
        )

        # Projection to xyz channels
        self.proj = nn.Sequential(
            nn.Conv3d(next_chan, 3, 1),
            nn.PReLU()
        )


    def forward(self, x, y, noise):
        noise_A, noise_B = noise
        
        x = torch.cat([x, noise_A], dim=1)
        x = interpolate(x, scale_factor=2, mode='trilinear')
        x = self.conv_A(x)
        x = torch.cat([x, noise_B], dim=1)
        x = self.conv_B(x)

        y = interpolate(y, scale_factor=2, mode='trilinear')
        y = crop(y, 2)
        y = y + self.proj(x)
        
        return x, y
