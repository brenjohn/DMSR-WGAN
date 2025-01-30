#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:06:27 2025

@author: brennan
"""

import torch
import torch.nn as nn

from torch.nn.functional import interpolate
from ..field_operations.resize import crop



class HBlock(nn.Module):
    """The H-block used in the generator model.
    
    The H-block take four inputs, (primary, auxiliary, noise A, noise B), and
    produces two outputs, (primary output, auxiliary output). The H-block has 
    the following structure:

                    (auxiliary input)               (primary input)
                           |                               |
                           |                               |
    (noise A)-----> concatenate noise               linear upsample
                           |                               |
                     linear upsample                     crop
                           |                               |
                      convolution                          |
                           |                               |
    (noise B)-----> concatenate noise                      |
                           |                               |
                      convolution                          |
                           |                               |
                           >-------- projection ---------->+
                           |        convolution            |
                           |                               |
                    (auxiliary output)              (primary output)

    Note, the spatial size of both outputs is:
        next_size = 2 * prev_size - 4
        
    Parameters:
        curr_chan : number of channels of auxiliary input
        next_chan : number of channels of auxiliary output
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



class ResidualBlock(nn.Module):
    """The Residual block used in the critic model.
    
    Residual convolution blocks have the following structure:
    
              (input)
                 |
                 |-------------------->|
                 |                     |
          Convolution (3x3x3)  Convolution (1x1x1)
                 |                     |
          Convolution (3x3x3)       Crop 2
                 |                     |
                 + <-------------------|
                 |
          Linear Downsample
                 |
             (output)
    
    Note, The output tensor has size `next_size = (prev_size - 4) / 2`, where 
    odd sizes are rounded down by the downsampling method.
    """
    
    def __init__(self, channels_curr, channels_next):
        super().__init__()
        self.skip = nn.Conv3d(channels_curr, channels_next, 1)
        self.convs = nn.Sequential(
            nn.Conv3d(channels_curr, channels_curr, 3),
            nn.PReLU(),
            nn.Conv3d(channels_curr, channels_next, 3),
            nn.PReLU(),
        )
            

    def forward(self, x):
        # Skip connection
        y = x
        y = self.skip(y)
        y = crop(y, 2)
        
        x = self.convs(x)
        x = x + y
        x = interpolate(x, scale_factor=0.5, mode='trilinear')
        return x