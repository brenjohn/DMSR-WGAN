#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:19:20 2024

@author: brennan
"""

import torch
import torch.nn as nn

from torch.nn.functional import interpolate
from ..field_operations.resize import crop


class DMSRCritic(nn.Module):
    
    def __init__(self, 
                 grid_size, 
                 channels,
                 **kwargs):
        super().__init__()

        self.input_size = grid_size
        self.block0 = nn.Sequential(
            nn.Conv3d(8, channels, 1),
            nn.PReLU(),
        )

        self.blocks = nn.ModuleList()
        size = grid_size
        channels_curr = channels
        channels_next = channels_curr * 2
        while size >= 10:
            self.blocks.append(ResidualBlock(channels_curr, channels_next))
            size = size // 2
            channels_curr = channels_next
            channels_next = channels_curr * 2
        
        
        self.block9 = nn.Sequential(
            nn.Conv3d(channels_curr, channels_next, 1),
            nn.PReLU(),
        )
        
        self.block10 = nn.Conv3d(channels_next, 1, 1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))


    def forward(self, x):
        
        x = self.block0(x)
        
        for block in self.blocks:
            x = block(x)

        x = self.block9(x)
        x = self.block10(x)
        x = self.pool(x)

        return x.flatten()
    
    

class ResidualBlock(nn.Module):
    """Residual convolution blocks of the form specified by `seq`.
    Input, via a skip connection, is added to the residual followed by an
    optional activation.

    The skip connection is identity if `out_chan` is omitted, otherwise it uses
    a size 1 "convolution", i.e. one can trigger the latter by setting
    `out_chan` even if it equals `in_chan`.

    A trailing `'A'` in seq can either operate before or after the addition,
    depending on the boolean value of `last_act`, defaulting to `seq[-1] == 'A'`

    See `ConvStyledBlock` for `seq` types.
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