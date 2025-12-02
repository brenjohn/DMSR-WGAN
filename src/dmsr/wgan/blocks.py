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
from ..data_tools import crop



class HBlock(nn.Module):
    """The H-block used in the generator model.
    
    The H-block take four inputs, (auxiliary, primary, noise A, noise B), and
    produces two outputs, (auxiliary output, primary output). The H-block has 
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
        self.INTERPOLATION_ARGS = {'scale_factor' : 2, 'mode' : 'trilinear'}
        Conv = DMSRStyleConv if style_size is not None else DMSRConv

        # First convolution operation in auxilary branch.
        self.conv_A = Conv(curr_chan + 1, next_chan, 3, style_size)
        self.relu_A = nn.PReLU()
        
        # Second convolution operation in auxilary branch.
        self.conv_B = Conv(next_chan + 1, next_chan, 3, style_size)
        self.relu_B = nn.PReLU()

        # Projection to primary xyz channels
        self.proj_conv = Conv(next_chan, prim_chan, 1, style_size)
        self.proj_relu = nn.PReLU()


    def forward(self, aux, primary, noise, style=None):
        # See HBlock docstring for dataflow explaination.
        noise_A, noise_B = noise
        
        aux = torch.cat([aux, noise_A], dim=1)
        aux = interpolate(aux, **self.INTERPOLATION_ARGS)
        aux = self.conv_A(aux, style)
        aux = self.relu_A(aux)
        
        aux = torch.cat([aux, noise_B], dim=1)
        aux = self.conv_B(aux, style)
        aux = self.relu_B(aux)

        primary = interpolate(primary, **self.INTERPOLATION_ARGS)
        primary = crop(primary, 2)
        projection = self.proj_conv(aux, style)
        projection = self.proj_relu(projection)
        primary = primary + projection
        
        return aux, primary



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
        self.INTERPOLATION_ARGS = {'scale_factor' : 0.5, 'mode' : 'trilinear'}
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
        x = interpolate(x, **self.INTERPOLATION_ARGS)
        return x