#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:15:42 2025

@author: brennan
"""

import math
import torch
import torch.nn as nn

from torch.nn import init
from torch.nn.functional import conv3d


class DMSRStyleConv(nn.Module):
    """Convolution layer with modulation and demodulation as described in
    Karras et al. 2020 "Analyzing and improving the image quality of styleGAN.
    """

    def __init__(
            self,
            channels_in,
            channels_out,
            kernel_size,
            style_size
        ):
        super().__init__()
        
        self.channels_in  = channels_in
        self.channels_out = channels_out
        self.kernel_size  = (kernel_size,) * 3
        self.style_size   = style_size
        self.eps          = 1e-8
        
        # Convolution parameters.
        self.weight = nn.Parameter(
            torch.empty(channels_out, channels_in, *self.kernel_size)
        )
        self.bias = nn.Parameter(
            torch.zeros(1, channels_out, 1, 1, 1)
        )
        
        # Style modulation block.
        self.style_block = nn.Linear(
            in_features=style_size, out_features=channels_in
        )
        
        # Initialize parameters.
        self._initialize_parameters()
        
    
    def _initialize_parameters(self):
        """Applies Kaiming initialization to weights and bias."""
        init.kaiming_uniform_(self.weight, a=0.25)
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    
    def forward(self, x, style=None):
        """
        Perform a modulated 3D convolution with optional style conditioning.

        If a style tensor is provided, each sample in the batch modulates the 
        convolution weights, followed by a demodulation step. The convolution 
        is then performed using group convolution.
        
        A styled-convolution on batched data requires applying a differently
        styled convolutional kernel to each sample in the batch (since each
        sample comes with its own style parameter). However, the standard
        convolution applies the same kernel to all samples in a batch. To
        circumvent this, the batch and channel dimensions of the weight and 
        input tensors are merged and a grouped convolution is used. 
        
        More specifically, to merge the batch and channel dimensions, the 
        channel dimensions for each batch are stacked to transform batched 
        tensors of shape (batch_size, channels, ...) to a tensor holding a 
        single batch sample with shape (1, batch_size * channels, ...). Then,
        a grouped convolutional operation is used, with groups=batch_size, to
        apply independent convolutions, each with distinctly styled kernels,
        to each sample in the batch. Finally, the channels of the grouped 
        convolution output is unstacked to change its from shape
        (1, batch_size * channels_out, ...) to (batch_size, channels_out, ...). 
        """
        batch_size, channels_in, grid_size, _, _ = x.shape
        weight = self.weight
        
        # Use the style block to transform the batch of style parameters into a
        # tensor of scales for modulating the weights. The scale tensor should
        # have shape (batch_size, 1, channels_in, 1, 1, 1).
        style_scale = self.style_block(style)
        style_scale = style_scale.view(batch_size, 1, channels_in, 1, 1, 1)
        
        # Modulate the weights, with the above scales, and give them shape 
        # (batch_size, channels_out, channels_in, size, size, size)
        weight = weight.unsqueeze(0) * style_scale
        
        # Demodulate the weigths by normalising them.
        variance = weight.pow(2).sum(dim=(2, 3, 4, 5), keepdim=True)
        demodulation_factor = torch.rsqrt(variance + self.eps)
        weight = weight * demodulation_factor
        
        # Reshape weights and input for group convolution by merging batch and
        # input channels.
        weight = weight.view(
            batch_size * self.channels_out, channels_in, *self.kernel_size
        )
        x = x.view(
            1, batch_size * channels_in, grid_size, grid_size, grid_size
        )
        
        # Apply group convolution.
        x = conv3d(
            x, weight, bias=None, stride=1, padding=0, groups=batch_size
        )
        
        # Unmerge batch and output channels before returning
        new_size = x.shape[-1]
        x = x.view(batch_size, self.channels_out, new_size, new_size, new_size)
        return x + self.bias
    
    

class DMSRConv(nn.Module):
    """Wrapper class for a standard Conv3d which accepts and ignores a style
    parameter in the constructor and forward method.
    """

    def __init__(
            self,
            channels_in,
            channels_out,
            kernel_size,
            *args
        ):
        super().__init__()
        self.conv = nn.Conv3d(channels_in, channels_out, kernel_size)
    
    
    def forward(self, x, style=None):
        return self.conv(x)