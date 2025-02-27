#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:19:20 2024

@author: brennan

This file defines the critic model used by the DMSR-WGAN model.
"""

import torch.nn as nn

from torch import concat, rand, autograd
from .blocks import ResidualBlock
from ..field_operations.conversion import cic_density_field
from ..field_operations.resize import pixel_unshuffle


class DMSRCritic(nn.Module):
    """Critic model for the DMSR-WGAN.
    
    This model evaluates the quality of high-resolution data by downscaling 
    it through a series of residual blocks and producing a critic score.
    """
    
    def __init__(
            self,
            input_size,
            input_channels,
            base_channels,
            density_scale_factor = None,
            **kwargs
        ):
        super().__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.build_critic_components()
        
        if density_scale_factor:
            self.density_scale = density_scale_factor
            self.density_size = density_scale_factor * input_size
            self.prepare_batch = self.prepare_batch_hr
        else:
            self.prepare_batch = self.prepare_batch_lr_hr
    
        
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
            nn.Conv3d(self.input_channels, self.base_channels, 1),
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
        """Forward pass of the critic.
        """
        x = self.initial_block(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.aggregate_block(x).flatten()
    
    
    def prepare_batch_lr_hr(self, hr_batch, lr_batch, box_size):
        """Prepare input data for the critic.
        
        Displacement coordinates are assumed to be contained in the first three
        channels of the given high- and low-resolution data.
        """
        lr_particles = lr_batch[:, :3, :, :, :]
        hr_particles = hr_batch[:, :3, :, :, :]
        lr_density = cic_density_field(lr_particles, box_size, self.input_size)
        hr_density = cic_density_field(hr_particles, box_size, self.input_size)
        return concat((hr_density, lr_density, hr_batch, lr_batch), dim=1),
    
    
    def prepare_batch_hr(self, hr_batch, lr_batch, box_size):
        """Prepare input data for the critic.
        
        Displacement coordinates are assumed to be contained in the first three
        channels of the given high- and low-resolution data.
        """
        hr_particles = hr_batch[:, :3, :, :, :]
        scale, density_size = self.density_scale, self.density_size
        hr_density = cic_density_field(hr_particles, box_size, density_size)
        hr_density = pixel_unshuffle(hr_density, scale)
        return concat((hr_density, hr_batch, lr_batch), dim=1),
    
    
    def gradient_penalty(
            self, 
            batch_size, 
            real_data, 
            fake_data, 
            device, 
            weight=10
        ):
        """Calculate the gradient penalty term for WGAN-GP.
        """
        # Create data by interpolating real and fake data and random amount.
        alpha = rand(batch_size, device=device).view(batch_size, 1, 1, 1, 1)
        mixed_data = real_data * alpha + fake_data * (1 - alpha)
        
        # Score the new data with the critic model and get its derivative.
        mixed_data.requires_grad_(True)
        score = self.forward(mixed_data).sum()
        grad, = autograd.grad(
            score,
            mixed_data,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )
        
        # Compute the gradient penalty term.
        grad = grad.flatten(start_dim=1)
        penalty = (weight * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
            + 0 * score
        )
        return penalty