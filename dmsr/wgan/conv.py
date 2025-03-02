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


class DMSRConv(nn.Module):
    """Convolution layer with modulation and demodulation as described in
    Karras et al. 2020 "Analyzing and improving the image quality of styleGAN.
    """

    def __init__(
            self,
            channels_in,
            channels_out,
            kernel_size,
            style_size=None
        ):
        super().__init__()
        
        self.channels_in  = channels_in
        self.channels_out = channels_out
        self.kernel_size  = (kernel_size,) * 3
        self.style_size   = style_size
        
        # Convolution parameters
        self.weight = nn.Parameter(
            torch.empty(channels_out, channels_in, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(channels_out))
        self.eps = 1e-8
        
        # Style modulation block (if applicable)
        self.style_block = None
        if style_size:
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
        weight = self.weight
        
        if style is not None:
            # Weight modulation and demodulation.
            scale = self.style_block(style).view(-1, self.channels_in, 1, 1, 1)
            weight = weight * scale
            variance = weight.pow(2).sum(dim=(1, 2, 3, 4), keepdim=True)
            weight = weight * torch.rsqrt(variance + self.eps)
        
        x = conv3d(x, weight, bias=self.bias, stride=1, padding=0)
        
        return x