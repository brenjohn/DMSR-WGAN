#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:28:55 2025

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import torch

from dmsr.wgan import DMSRStyleConv, DMSRConv
from dmsr.data_tools import generate_mock_data

x, _, = generate_mock_data(20, 32, 3, 2)

channels_in  = 3 
channels_out = 4
kernel_size  = 3
style_size   = 1


#%% Standard Convolution.
conv = DMSRConv(
    channels_in, channels_out, kernel_size
)

y = conv(x)
assert y.shape == (2, 4, 18, 18, 18)


#%% Standard Convolution that ignores style.
conv = DMSRConv(
    channels_in, channels_out, kernel_size, style_size
)

s = torch.rand(2, 1)
y = conv(x, s)
assert y.shape == (2, 4, 18, 18, 18)


#%% Styled Convolution.
style_conv = DMSRStyleConv(
    channels_in, channels_out, kernel_size, style_size
)

s = torch.rand(2, 1)
y = style_conv(x, s)
assert y.shape == (2, 4, 18, 18, 18)