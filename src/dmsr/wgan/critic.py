#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:19:20 2024

@author: brennan

This file defines the critic model used by the DMSR-WGAN model.
"""

import torch.nn as nn

from torch import save, load
from torch import concat, rand, autograd
from pathlib import Path

from .conv import DMSRConv, DMSRStyleConv
from .blocks import ResidualBlock
from ..field_operations import cic_density_field, nn_distance_field
from ..data_tools import pixel_unshuffle, crop


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
            style_size = None,
            use_nn_distance_features = False,
            nn_smoothing = 0.01,
            nn_numerator = -1,
            **kwargs
        ):
        super().__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.density_scale = density_scale_factor
        self.style_size = style_size
        self.use_nn_distance_features = use_nn_distance_features
        self.build_critic_components()
        
        if density_scale_factor is not None:
            self.density_size = density_scale_factor * input_size
            self.prepare_batch = self.prepare_batch_hr
        elif use_nn_distance_features:
            self.nn_smoothing = nn_smoothing
            self.nn_numerator = nn_numerator
            self.prepare_batch = self.prepare_batch_nn
        else:
            self.prepare_batch = self.prepare_batch_lr_hr
            
            
    def get_arch_params(self):
        return {
            'input_size'               : self.input_size,
            'input_channels'           : self.input_channels,
            'base_channels'            : self.base_channels,
            'style_size'               : self.style_size,
            'density_scale_factor'     : self.density_scale,
            'use_nn_distance_features' : self.use_nn_distance_features
        }
    
        
    def layer_channels_and_sizes(self):
        """Compute input and output channels of each layer.
        
        The output sizes of residual blocks are also computed.
        
        Returns two lists for the main and density branches of the model
        containing (channel_in, channel_out, out_size) tuples for each residual
        block in each branch.
        """
        # Main residual blocks
        MAX_DATA_SIZE = 10
        data_size = self.input_size
        channels_curr = self.base_channels
        channels_next = channels_curr * 2
        
        # Each residual block doubles the number of channels in the data and 
        # reduces the size of its input to (N - 4) // 2, where N is the input 
        # size. The loop below adds residual blocks until the data size is less
        # than or equal to MAX_DATA_SIZE.
        blocks = []
        while data_size >= MAX_DATA_SIZE:
            data_size = (data_size - 4) // 2
            blocks.append(
                (channels_curr, channels_next, data_size)
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
        Conv = DMSRStyleConv if self.style_size is not None else DMSRConv
        
        self.initial_conv = Conv(
            self.input_channels, self.base_channels, 1, self.style_size
        )
        self.initial_relu = nn.PReLU()

        self.residual_blocks = nn.ModuleList()
        for channel_in, channel_out, size in residual_layers:
            self.residual_blocks.append(
                ResidualBlock(channel_in, channel_out, self.style_size)
            )
        
        # Aggregation components.
        self.agg_conv_1 = Conv(channel_out, channel_out, 1, self.style_size)
        self.agg_relu   = nn.PReLU()
        self.agg_conv_2 = Conv(channel_out, 1, 1, self.style_size)
        self.agg_pool   = nn.AdaptiveAvgPool3d((1, 1, 1))


    def forward(self, x, style=None):
        """Forward pass of the critic.
        """
        x = self.initial_conv(x, style)
        x = self.initial_relu(x)
        
        for block in self.residual_blocks:
            x = block(x, style)
        
        # Aggregation
        x = self.agg_conv_1(x, style)
        x = self.agg_relu(x)
        x = self.agg_conv_2(x, style)
        x = self.agg_pool(x).flatten()
        
        return x
    
    
    def prepare_batch_lr_hr(self, hr_batch, lr_batch, box_size):
        """Prepares input data for the critic by computing density fields for 
        both lr and hr data. 
        
        Returns a tensor with the following channels:
            [hr density, lr density, hr particle data, lr particle data]
        
        Displacement coordinates are assumed to be contained in the first three
        channels of the given high- and low-resolution data.
        """
        lr_particles = lr_batch[:, :3, :, :, :]
        hr_particles = hr_batch[:, :3, :, :, :]
        lr_density = cic_density_field(lr_particles, box_size, self.input_size)
        hr_density = cic_density_field(hr_particles, box_size, self.input_size)
        return concat((hr_density, lr_density, hr_batch, lr_batch), dim=1),
    
    
    def prepare_batch_hr(self, hr_batch, lr_batch, box_size):
        """Prepares input data for the critic by computing a density field for
        the hr data. The resolution of the density field is set by 
        `self.density_size`. This will be larger than the size/resolution of
        the particle data by a factor of `self.density_scale`. A pixel 
        unshuffle operation is used to reshape the density field into a
        multi-channel tensor with the same spatial resolution as the particle
        data. 
        
        A tensor with the following channels is then returned:
            [denisty field channels, hr particle data, lr particle data]
        
        Displacement coordinates are assumed to be contained in the first three
        channels of the given high- and low-resolution data.
        """
        hr_particles = hr_batch[:, :3, :, :, :]
        scale, density_size = self.density_scale, self.density_size
        hr_density = cic_density_field(hr_particles, box_size, density_size)
        hr_density = pixel_unshuffle(hr_density, scale)
        return concat((hr_density, hr_batch, lr_batch), dim=1),
    
    
    def prepare_batch_nn(self, hr_batch, lr_batch, box_size):
        """Prepare input data for the critic by computing a density field for
        the hr data. Feature maps encoding the distance between neighbouring
        particles are also computed. The feature maps are reminiscent of the
        gravitational potential between particles and uses a formula with the
        following form:
            f = numerator_const / (distance + smoothing_const)
        
        Returns a tenors with the following channels:
            [nn feature maps, density field, hr particles, lr particles]
        
        Displacement coordinates are assumed to be contained in the first three
        channels of the given high- and low-resolution data.
        """
        hr_particles = hr_batch[:, :3, :, :, :]
        hr_density = cic_density_field(
            hr_particles, box_size, self.input_size + 2
        )
        all_fields = concat((hr_density, hr_batch, lr_batch), dim=1)
        all_fields = crop(all_fields, 1)
        nn_distance = nn_distance_field(hr_particles, box_size)
        nn_features = self.nn_numerator / (nn_distance + self.nn_smoothing)
        return concat((nn_features, all_fields), dim=1),
    
    
    def gradient_penalty(
            self, 
            batch_size, 
            real_data, 
            fake_data,
            style,
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
        score = self.forward(mixed_data, style).sum()
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
    
    
    #=========================================================================#
    #                         Saving and Loading
    #=========================================================================#
    
    def save(self, model_dir=Path('./data/model/')):
        """Save the model state dictionary and architecture metadata.
        """
        model_dir.mkdir(parents=True, exist_ok=True)
        save(self.state_dict(), model_dir / 'critic.pth')
        crit_arch_metadata = self.get_arch_params()
        save(crit_arch_metadata, model_dir / 'crit_arch.pth')
    
    
    @classmethod
    def load(cls, model_dir, device):
        """Load a saved model
        """
        arch = load(
            model_dir / 'crit_arch.pth', 
            map_location=device, 
            weights_only=False
        )
        crit_state_dict = load(
            model_dir / 'critic.pth', 
            map_location=device, 
            weights_only=True
        )
        critic = DMSRCritic(**arch).to(device)
        critic.load_state_dict(crit_state_dict)
        return critic