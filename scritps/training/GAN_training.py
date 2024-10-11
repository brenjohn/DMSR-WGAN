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
from dmsr.swift_processing import load_numpy_dataset


# Check if CUDA is available and set the device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#=============================================================================#
#                      Generator and Critic Models
#=============================================================================#
LR_grid_size, generator_channels, scale_factor = 20, 8, 4
generator = DMSRGenerator(LR_grid_size, generator_channels, scale_factor)

critic_size = generator.output_size
critic_channels = 8
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
#                           Training Dataset
#=============================================================================#
data_directory = '../../data/dmsr_training/'
batch_size = 4

data = load_numpy_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data

# TODO: The four below depends on scale factor. Maybe this should be read from
# dataset somehow or metadata.
lr_padding = 3
# LR_data, HR_data = LR_data[:2, ...], HR_data[:2, ...]

dataset = DMSRDataset(
    LR_data.float(), HR_data.float(), augment=True
)

dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


#=============================================================================#
#                           Validation Dataset
#=============================================================================#
data_directory = '../../data/dmsr_validation/'

data = load_numpy_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data
# LR_data, HR_data = LR_data[:2, ...], HR_data[:2, ...]

# valid_dataset = DMSRDataset(LR_data.float(), HR_data.float())
# valid_dataloader = DataLoader(valid_dataset, batch_size=1)


#=============================================================================#
#                              DMSR WGAN
#=============================================================================#
gan = DMSRWGAN(generator, critic, device)
gan.set_dataset(dataloader, batch_size, box_size, lr_padding, scale_factor)
gan.set_optimizer(optimizer_c, optimizer_g)


#=============================================================================#
#                               Monitor
#=============================================================================#
from dmsr.dmsr_gan.dmsr_monitor import MonitorManager, LossMonitor
from dmsr.dmsr_gan.dmsr_monitor import SamplesMonitor, CheckpointMonitor
from dmsr.dmsr_gan.dmsr_monitor import UpscaleMonitor

lr_sample = LR_data[1:2, ...].float()
hr_sample = HR_data[1:2, ...].float()
lr_box_size = 20 * box_size / 16
hr_box_size = box_size

monitors = {
    'loss_monitor' : LossMonitor(),
    
    'samples_monitor' : SamplesMonitor(
        generator, 
        lr_sample, hr_sample, 
        lr_box_size, hr_box_size, 
        device
    ),
    
    'checkpoint_monitor' : CheckpointMonitor(
        gan,
        checkpoint_dir = './data/checkpoints/'
    )
}

realisations = 1
upscaling_monitor = UpscaleMonitor(
    generator,
    realisations,
    device
)


particle_mass = 1
grid_size = int(HR_grid_size)
upscaling_monitor.set_data_set(
    LR_data.float(), 
    HR_data.float(), 
    particle_mass, 
    box_size, 
    grid_size
)
monitors['upscaling_monitor'] = upscaling_monitor

batch_report_rate = 16
monitor_manager = MonitorManager(batch_report_rate, device)
monitor_manager.set_monitors(monitors)
gan.set_monitor(monitor_manager)


#=============================================================================#
#                         Supervised Training
#=============================================================================#
# from dmsr.dmsr_gan.dmsr_monitor import SupervisedValidator

# supervised_epochs = 5
# validator = SupervisedValidator(generator, valid_dataloader, device)


#%%
# gan.train(
#     supervised_epochs, 
#     train_step = gan.generator_supervised_step, 
#     validator = validator
# )
# gan.train(5, train_step=gan.critic_supervised_step)

#%%
# import numpy as np
# import matplotlib.pyplot as plt

# plt.plot([np.sum(monitor.generator_loss[128*n:128*(n+1)])/128 for n in range(20)])
# plt.plot(validator.generator_valid_losses)



#=============================================================================#
#                           WGAN Training
#=============================================================================#

num_epochs = 2 # 1024 * 6
gan.train(num_epochs)


#%%
# gan.load('./data/checkpoints/current_model/')

#%%
dataloader = DataLoader(
    LR_data.float(), batch_size=1
)