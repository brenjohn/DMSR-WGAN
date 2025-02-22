#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:19:20 2024

@author: brennan

This file defines the critic model used by the DMSR-WGAN model.
"""

import torch
import torch.nn as nn

from torch.nn.functional import interpolate
from ..field_operations.resize import crop


class DMSRCritic(nn.Module):
    """The critic model for the DMSR-WGAN.
    """
    
    def __init__(
            self,
            density_size,
            displacement_size,
            density_channels,
            main_channels,
            **kwargs
        ):
        super().__init__()
        self.density_size = density_size
        self.displacement_size = displacement_size
        self.density_channels = density_channels
        self.main_channels = main_channels
        
        self.build_critic_components()
        
        
    def layer_channels_and_sizes(self):
        """Compute the input and output channels of each layer.
        
        The output sizes of residual blocks are also computed.
        
        Returns two lists for the main and denisty branches of the model
        containing (channel_in, channel_out, out_size) tuples for each residual
        block in each branch.
        """
        
        # Density residual blocks
        density_size = self.density_size
        displacement_size = self.displacement_size
        channels_curr = self.density_channels
        channels_next = channels_curr * 2
        
        density_blocks = []
        while density_size > displacement_size:
            density_size = (density_size - 4)//2
            density_blocks.append(
                (channels_curr, channels_next, density_size)
            )
            
            channels_curr = channels_next
            channels_next = channels_curr * 2
            
            if density_size < displacement_size:
                raise('Density size incompatiable with displacement size')
        
        
        # Main residual blocks
        size = displacement_size
        channels_curr = self.main_channels
        channels_next = channels_curr * 2
        
        main_blocks = []
        while size >= 10:
            size = (size - 4) // 2
            main_blocks.append(
                (channels_curr, channels_next, size)
            )
            channels_curr = channels_next
            channels_next = channels_curr * 2
        
        
        return density_blocks, main_blocks
        
        
    def build_critic_components(self):
        """Creates the neural network components of the critic model.
        
        The model consists of of two branchs; a density branch that downscales
        the given density data to the same resolution as the displacement data,
        and a main branch for computing the score for the given denisty and
        displacement data. Initial blocks are used to create the initial 
        channels for the density and displacement data. Finally, an aggregation 
        block is used to reduce the output to a single number.
    
        (SR_density, LR_density)                     <--- Density Input
                   |
             Density Block
                   |
            Residual Block
                   |
                   |    (SR_data, LR_data)           <--- Displacement Input
                   |             |
                   |---------> concat
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
        density_layers, main_layers = self.layer_channels_and_sizes()
        
        self.density_initial_block = nn.Sequential(
            nn.Conv3d(2, self.density_channels, 1),
            nn.PReLU(),
        )
        
        # Note: channel_out = density_channels if density_channels is empty.
        channel_out = self.density_channels
        self.density_blocks = nn.ModuleList()
        for channel_in, channel_out, size in density_layers:
            self.density_blocks.append(
                ResidualBlock(channel_in, channel_out)
            )
        
        self.main_initial_block = nn.Sequential(
            nn.Conv3d(6 + channel_out, self.main_channels, 1),
            nn.PReLU(),
        )

        self.main_blocks = nn.ModuleList()
        for channel_in, channel_out, size in main_layers:
            self.main_blocks.append(
                ResidualBlock(channel_in, channel_out)
            )
        
        self.aggregate_block = nn.Sequential(
            nn.Conv3d(channel_out, channel_out, 1),
            nn.PReLU(),
            nn.Conv3d(channel_out, 1, 1),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )


    def forward(self, displacements, densities):
        
        y = self.density_initial_block(densities)
        for block in self.density_blocks:
            y = block(y)
        
        x = torch.cat([displacements, y], dim=1)
        x = self.main_initial_block(x)
        for block in self.main_blocks:
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