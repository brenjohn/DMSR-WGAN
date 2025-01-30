#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:18:31 2024

@author: brennan

This file defines the generator model used by the DMSR-WGAN model.
"""

import torch.nn as nn

from torch import randn
from ..field_operations.resize import crop
from .blocks import HBlock


class DMSRGenerator(nn.Module):
    """The generator model for the DMSR-WGAN.
    """
    
    def __init__(self,
                 grid_size,
                 input_channels,
                 base_channels,
                 crop_size=0,
                 scale_factor=8,
                 **kwargs
        ):
        super().__init__()
        self.grid_size      = grid_size
        self.input_channels = input_channels
        self.base_channels  = base_channels
        self.scale_factor   = scale_factor
        self.crop_size      = crop_size
        
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
            nn.Conv3d(self.input_channels, self.base_channels, 1),
            nn.PReLU(),
        )
        
        scale = 1
        curr_chan = self.base_channels
        next_chan = curr_chan // 2
        prim_chan = self.input_channels
        N = self.grid_size
        noise_shapes = []
        
        self.blocks = nn.ModuleList()
        while scale < self.scale_factor:
            self.blocks.append(
                HBlock(curr_chan, next_chan, prim_chan)
            )
            
            scale *= 2
            curr_chan = next_chan
            next_chan = curr_chan // 2
            noise_shapes.append((N, 2 * N - 2))
            N = 2 * N - 4
        
        self.output_size = N - 2 * self.crop_size
        self.noise_shapes = noise_shapes
        
        
    def compute_input_padding(self):
        """Computes the sizes of the input inner region and padding.
        
        The low resolution input to the generator is thought of as being made
        up of two regions: an inner region to be upscaled and an outer region 
        of padding. The output of the generator is thought of as an upscaled
        version of the inner region. This method computes the sizes of these 
        regions.
        """
        if self.output_size % self.scale_factor != 0:
            print('WARNING: inner region of generator input not an integer')
        self.inner_region = self.output_size // self.scale_factor
        
        if (self.grid_size - self.inner_region) % 2 != 0:
            print('WARNING: padding of generator input not an integer')
        self.padding = (self.grid_size - self.inner_region) // 2


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
