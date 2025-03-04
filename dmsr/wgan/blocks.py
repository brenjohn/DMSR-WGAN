#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:06:27 2025

@author: brennan
"""

import torch
import torch.nn as nn

from .conv import DMSRConv, DMSRStyleConv
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
        prim_chan : number of channels of primary input
    """

    def __init__(self, curr_chan, next_chan, prim_chan, style_size=None):
        super().__init__()
        Conv = DMSRStyleConv if style_size is not None else DMSRConv

        self.conv_A = Conv(curr_chan + 1, next_chan, 3, style_size)
        self.relu_A = nn.PReLU()
        
        self.conv_B = Conv(next_chan + 1, next_chan, 3, style_size)
        self.relu_B = nn.PReLU()

        # Projection to xyz channels
        self.proj_conv = Conv(next_chan, prim_chan, 1, style_size)
        self.proj_relu = nn.PReLU()


    def forward(self, x, y, noise, style=None):
        noise_A, noise_B = noise
        
        x = torch.cat([x, noise_A], dim=1)
        x = interpolate(x, scale_factor=2, mode='trilinear')
        x = self.conv_A(x, style)
        x = self.relu_A(x)
        
        x = torch.cat([x, noise_B], dim=1)
        x = self.conv_B(x, style)
        x = self.relu_B(x)

        y = interpolate(y, scale_factor=2, mode='trilinear')
        y = crop(y, 2)
        p = self.proj_conv(x, style)
        p = self.proj_relu(p)
        y = y + p
        
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
    
    def __init__(self, channels_in, channels_out, style_size=None):
        super().__init__()
        Conv = DMSRStyleConv if style_size is not None else DMSRConv
        
        self.skip   = Conv(channels_in, channels_out, 1, style_size)
        self.conv_A = Conv(channels_in, channels_in, 3, style_size)
        self.relu_A = nn.PReLU()
        self.conv_B = Conv(channels_in, channels_out, 3, style_size)
        self.relu_B = nn.PReLU()
            

    def forward(self, x, style=None):
        # Skip connection
        y = x
        y = self.skip(y, style)
        y = crop(y, 2)
        
        x = self.conv_A(x, style)
        x = self.relu_A(x)
        x = self.conv_B(x, style)
        x = self.relu_B(x)
        
        x = x + y
        x = interpolate(x, scale_factor=0.5, mode='trilinear')
        return x