#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:14:03 2024

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import time
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from dmsr.wgan import DMSRWGAN
from dmsr.wgan import DMSRDensityCritic
from dmsr.wgan import DMSRGenerator
from dmsr.data_tools import DMSRDataset

from dmsr.data_tools import generate_mock_data

# Check if CUDA is available and set the device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#=============================================================================#
#                      Generator and Critic Models
#=============================================================================#
LR_grid_size   = 20
input_channels = 3
base_channels  = 3
crop_size      = 2
scale_factor   = 2

generator = DMSRGenerator(
    LR_grid_size, input_channels, base_channels, crop_size, scale_factor
)

hr_grid_size      = generator.output_size
density_size      = 2 * hr_grid_size + 4
displacement_size = hr_grid_size
density_channels  = 4
main_channels     = 8

critic = DMSRDensityCritic(
    density_size, displacement_size, density_channels, main_channels
)

generator.to(device)
critic.to(device)


#=============================================================================#
#                              Optimizers
#=============================================================================#
b1 = 0.0
b2 = 0.99

lr_G = 0.00001
optimizer_g = optim.Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))

lr_C = 0.00002
optimizer_c = optim.Adam(critic.parameters(), lr=lr_C, betas=(b1, b2))


#=============================================================================#
#                           Training Dataset
#=============================================================================#
batch_size = 2
box_size = 1
lr_padding = 1

lr_data, hr_data = generate_mock_data(
    LR_grid_size, hr_grid_size, 3, batch_size
)

dataset = DMSRDataset(
    lr_data, hr_data, augment=True
)

dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


#=============================================================================#
#                              DMSR WGAN
#=============================================================================#
gan = DMSRWGAN(generator, critic, device)
gan.set_dataset(dataloader, batch_size, box_size)
gan.set_optimizer(optimizer_c, optimizer_g)


#%%
ti = time.time()
gan.train_step(lr_data, hr_data)
time_train_step = time.time() - ti
print('train step took :', time_train_step)


#%%
gan_dir = './gan_model/'
gan.save(gan_dir)
gan.load(gan_dir)