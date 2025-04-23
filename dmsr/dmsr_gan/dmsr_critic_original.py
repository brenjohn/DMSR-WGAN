#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:22:51 2025

@author: brennan
"""

import torch
import torch.nn as nn

from torch.nn.functional import interpolate
from ..field_operations.resize import crop


class DMSRCriticOriginal(nn.Module):
    """The original critic model for the DMSR-WGAN.
    """
    
    def __init__(
            self,
            input_size,
            base_channels,
            **kwargs
        ):
        super().__init__()
        self.input_size = input_size
        self.base_channels = base_channels
        
        self.build_critic_components()
        
        
    def layer_channels_and_sizes(self):
        """Compute input and output channels of each layer.
        
        The output sizes of residual blocks are also computed.
        
        Returns two lists for the main and density branches of the model
        containing (channel_in, channel_out, out_size) tuples for each residual
        block in each branch.
        """
        # Main residual blocks
        size = self.input_size
        channels_curr = self.base_channels
        channels_next = channels_curr * 2
        
        blocks = []
        while size >= 10:
            size = (size - 4) // 2
            blocks.append(
                (channels_curr, channels_next, size)
            )
            channels_curr = channels_next
            channels_next = channels_curr * 2
        
        return blocks
        
        
    def build_critic_components(self):
        """Creates the neural network components of the critic model.
        
        The model consists of a sequence of residual blocks that downscales
        the given data and computes a score for it. An initial block is used to 
        create the initial channels. Finally, an aggregation block is used to 
        reduce the output to a single number.
                    
                        (SR_data, LR_data)           <--- Input
                                 |
                           Initial Block
                                 |
                          Residual Block
                                 |
                          Residual Block
                                 :
                                 :
                         Aggregation Block
                                 |
                           (Critic score)            <---- Output
        """
        residual_layers = self.layer_channels_and_sizes()
        
        self.initial_block = nn.Sequential(
            nn.Conv3d(8, self.base_channels, 1),
            nn.PReLU(),
        )
        
        self.residual_blocks = nn.ModuleList()
        for channel_in, channel_out, size in residual_layers:
            self.residual_blocks.append(
                ResidualBlock(channel_in, channel_out)
            )
        
        self.aggregate_block = nn.Sequential(
            nn.Conv3d(channel_out, channel_out, 1),
            nn.PReLU(),
            nn.Conv3d(channel_out, 1, 1),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )


    def forward(self, x):
        
        x = self.initial_block(x)
        for block in self.residual_blocks:
            x = block(x)

        x = self.aggregate_block(x)

        return x.flatten()
    
    

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