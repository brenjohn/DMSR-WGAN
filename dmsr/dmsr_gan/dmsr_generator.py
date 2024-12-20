#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:18:31 2024

@author: brennan

This file defines the generator model used by the DMSR-WGAN model.
"""

import torch
import torch.nn as nn

from torch import randn
from torch.nn.functional import interpolate
from ..field_operations.resize import crop


class DMSRGenerator(nn.Module):
    """The generator model for the DMSR-WGAN.
    """
    
    def __init__(self,
                 grid_size,
                 channels,
                 crop_size=0,
                 scale_factor=8,
                 **kwargs
        ):
        super().__init__()
        self.grid_size    = grid_size
        self.channels     = channels
        self.scale_factor = scale_factor
        self.crop_size    = crop_size
        
        self.build_generator_components()


    def build_generator_components(self):
        """Create the neural network components of the dmsr generator model.
        
        The model consists of an initial block to increase the number of input
        channels and a sequence of H-blocks to upscale the data. The output
        of the final H-block is cropped by a specified amount before being
        returned.
        
                            (LR-data)
                                |-----------|
                        (Initial-block)     | skip
                                |    -------|
                                |    |
                               (H-Block)
                                |    |
                               (H-Block)
                                |    |
                                   :
                                |    |
                               (H-Block)
                                |    |
                                -    |
                                   (crop)
                                     |
                                  (output)
        """
        self.initial_block = nn.Sequential(
            nn.Conv3d(3, self.channels, 1),
            nn.PReLU(),
        )
        
        scale = 1
        curr_chan = self.channels
        next_chan = curr_chan // 2
        N = self.grid_size
        noise_shapes = []
        
        self.blocks = nn.ModuleList()
        while scale < self.scale_factor:
            self.blocks.append(
                HBlock(curr_chan, next_chan)
            )
            
            scale *= 2
            curr_chan = next_chan
            next_chan = curr_chan // 2
            noise_shapes.append((N, 2 * N - 2))
            N = 2 * N - 4
        
        self.output_size = N - 2 * self.crop_size
        self.noise_shapes = noise_shapes


    def forward(self, x, z):
        y = x
        x = self.initial_block(x)
        for block, noise in zip(self.blocks, z):
            x, y = block(x, y, noise)
        
        if self.crop_size:
            y = crop(y, self.crop_size)
        
        return y
    
    
    def sample_latent_space(self, batch_size, device):
        """Returns a sample from the generator's latent space.
        """
        latent_variable = [None] * len(self.noise_shapes)
        for i, (shape_A, shape_B) in enumerate(self.noise_shapes):
            shape_A = (batch_size, 1) + 3 * (shape_A,)
            shape_B = (batch_size, 1) + 3 * (shape_B,)
            noise = randn(shape_A).to(device), randn(shape_B).to(device)
            latent_variable[i] = noise
            
        return latent_variable



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
