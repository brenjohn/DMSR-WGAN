#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:46:51 2025

@author: brennan

This file defines a generalised critic model for the the DMSR-WGAN with an
additional branch for processing density field data.
"""

import torch
import torch.nn as nn

from torch import concat, rand, autograd
from ..field_operations.conversion import cic_density_field
from .blocks import ResidualBlock


class DMSRDensityCritic(nn.Module):
    """A generalized critic model for the DMSR-WGAN. The model is similar to
    the DMSRCritic model but with an additional density branch.
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
        
        Returns two lists for the main and density branches of the model
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
        """Forward pass of the critic.
        """
        
        y = self.density_initial_block(densities)
        for block in self.density_blocks:
            y = block(y)
        
        x = torch.cat([displacements, y], dim=1)
        x = self.main_initial_block(x)
        for block in self.main_blocks:
            x = block(x)

        return self.aggregate_block(x).flatten()
    
    
    def prepare_batch(self, hr_batch, lr_batch, box_size):
        """Prepare input data for the critic.
        """
        density_size = self.density_size
        lr_density = cic_density_field(lr_batch, box_size, density_size)
        hr_density = cic_density_field(hr_batch, box_size, density_size)
        displacement = concat((hr_batch, lr_batch), dim=1).detach()
        density = concat((hr_density, lr_density), dim=1).detach()
        return (displacement, density)
    
    
    def gradient_penalty(
            self, 
            batch_size, 
            real_displacements,
            real_density,
            fake_displacements,
            fake_density,
            device, 
            weight=10
        ):
        """Calculate the gradient penalty.
        """
        # Create data by interpolating real and fake data and random amount.
        alpha = rand(batch_size, device=device).view(batch_size, 1, 1, 1, 1)
        displacements  = real_displacements * alpha
        displacements += fake_displacements * (1 - alpha)
        density  = real_density * alpha
        density += fake_density * (1 - alpha)
        
        # Score the new data with the critic model and get its derivative.
        score = self.forward(
            displacements.requires_grad_(True), density.requires_grad_(True)
        )
        score = score.sum()
        displacement_grad, density_grad = autograd.grad(
            score,
            (displacements, density),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )
        
        # Compute the gradient penalty term.
        density_grad = density_grad.flatten(start_dim=1)
        displacement_grad = displacement_grad.flatten(start_dim=1)
        grad = concat((displacement_grad, density_grad), dim=1)
        penalty = (weight * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
            + 0 * score  # hack to trigger DDP allreduce hooks
        )
        return penalty