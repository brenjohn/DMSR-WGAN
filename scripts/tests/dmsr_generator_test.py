#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:34:54 2024

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import time
import torch

from dmsr.wgan import DMSRGenerator

# Check if CUDA is available and set the device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%% Create the generator model
LR_grid_size   = 20
input_channels = 3
base_channels  = 3
crop_size      = 0
scale_factor   = 2

generator = DMSRGenerator(
    LR_grid_size, input_channels, base_channels, crop_size, scale_factor
)

generator.to(device)


#%% Foward Pass
batch_size = 2

lr_sample = torch.randn(
    (batch_size, 3, LR_grid_size, LR_grid_size, LR_grid_size)
).to(device)

z = generator.sample_latent_space(batch_size, device)

initial_time = time.time()
sr_sample = generator(lr_sample, z)
final_time = time.time()

print("Shape of generated output is", sr_sample.shape)
print("Prediction took", final_time-initial_time, "seconds")