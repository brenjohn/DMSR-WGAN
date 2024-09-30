#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:35:32 2024

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from dmsr.dmsr_gan.dmsr_wgan import DMSRWGAN
from dmsr.dmsr_gan.dmsr_critic import DMSRCritic
from dmsr.dmsr_gan.dmsr_generator import DMSRGenerator

from dmsr.dmsr_gan.dmsr_dataset import DMSRDataset
from dmsr.dmsr_gan.dmsr_monitor import DMSRMonitor
from dmsr.swift_processing import load_numpy_dataset


# Check if CUDA is available and set the device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#=============================================================================#
#                      Generator and Critic Models
#=============================================================================#
LR_grid_size, generator_channels, scale_factor = 20, 256, 2
generator = DMSRGenerator(LR_grid_size, generator_channels, scale_factor)

critic_size = generator.output_size
critic_channels = 128
critic = DMSRCritic(critic_size, critic_channels)

generator.to(device)
critic.to(device)


#=============================================================================#
#                               Optimizers
#=============================================================================#
b1 = 0.0
b2 = 0.99

lr_G = 0.00001
optimizer_g = optim.Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))

lr_C = 0.00002
optimizer_c = optim.Adam(critic.parameters(), lr=lr_C, betas=(b1, b2))


#=============================================================================#
#                               Dataset
#=============================================================================#
data_directory = '../../data/dmsr_training/'
batch_size = 8

data = load_numpy_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data
lr_padding = (int(LR_grid_size) - int(HR_grid_size) // 2) // 2
# LR_data, HR_data = LR_data[:2, ...], HR_data[:2, ...]

dataset = DMSRDataset(LR_data.float(), HR_data.float(), augment=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#=============================================================================#
#                               Monitor
#=============================================================================#
n = 1
lr_sample = LR_data[n:n+1, ...].float()
hr_sample = HR_data[n:n+1, ...].float()

monitor = DMSRMonitor(
    generator, 16, lr_sample, 20*box_size/16, hr_sample, box_size, device
)


#=============================================================================#
#                              DMSR WGAN
#=============================================================================#
gan = DMSRWGAN(generator, critic, device)
gan.set_dataset(dataloader, batch_size, box_size, lr_padding, scale_factor)
gan.set_optimizer(optimizer_c, optimizer_g)
gan.set_monitor(monitor)


#=============================================================================#
#                         Supervised Training
#=============================================================================#
gan.train(5, train_step=gan.generator_supervised_step)
# gan.train(5, train_step=gan.critic_supervised_step)



#=============================================================================#
#                           WGAN Training
#=============================================================================#

num_epochs = 1024 * 6
gan.train(num_epochs)
