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
    """The DMSR critic model for the DMSR WGAN.
    """
    
    def __init__(
            self,
            density_size,
            displacement_size,
            density_channels,
            displacement_channels,
            **kwargs
        ):
        super().__init__()
        self.density_size = density_size
        self.displacement_size = displacement_size
        self.density_channels = density_channels
        self.displacement_channels = displacement_channels
        
        self.build_critic_components()
        
        
    def layer_channels_and_sizes(self):
        """Compute the input and output channels, along with the output sizes 
        of the residual blocks to be used in the critic model.
        
        Returns two lists for the denisty and main branches of the model
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
        channels_curr = self.displacement_channels + channels_curr
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
        """Create the neural network components of the dmsr critic model.
        
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
                   |
                   |    (SR_data, LR_data)           <--- Displacement Input
                   |             |
                   |       Initial Block
                   |             |
                   |-----------concat
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
        density_channels, main_channels = self.layer_channels_and_sizes()
        
        density_initial_channels = self.density_channels
        displacement_initial_channels = self.displacement_channels
        
        self.density_initial_block = nn.Sequential(
            nn.Conv3d(2, density_initial_channels, 1),
            nn.PReLU(),
        )
        
        self.density_blocks = nn.ModuleList()
        for channel_in, channel_out, size in density_channels:
            self.density_blocks.append(
                ResidualBlock(channel_in, channel_out)
            )
        
        self.displacement_initial_block = nn.Sequential(
            nn.Conv3d(6, displacement_initial_channels, 1),
            nn.PReLU(),
        )

        self.main_blocks = nn.ModuleList()
        for channel_in, channel_out, size in main_channels:
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
        
        x = self.displacement_initial_block(displacements)
        x = torch.cat([x, y], dim=1)
        for block in self.main_blocks:
            x = block(x)

        x = self.aggregate_block(x)

        return x.flatten()
    
    

class ResidualBlock(nn.Module):
    """Residual convolution blocks have the following structure:
    
                 x
                 |-------------------->|
                 |                     |
          Convolution (3x3x3)  Convolution (1x1x1)
                 |                     |
          Convolution (3x3x3)          |
                 |                     |
                 + <-------------------|
                 |
             Downsample
                 |
                 x
    
    Note, The output tensor has size given by:
        next_size = (prev_size - 4) / 2
    where odd sizes are rounded down by the downsampling method.
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