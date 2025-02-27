#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:35:32 2024

@author: brennan
"""

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from dmsr.wgan import DMSRWGAN
from dmsr.wgan import DMSRCritic
from dmsr.wgan import DMSRGenerator

from dmsr.data_tools import DMSRDataset
from dmsr.data_tools import load_numpy_dataset
from dmsr.data_tools import generate_mock_data


# Check if CUDA is available and set the device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

output_dir = './test_run/'
os.makedirs(output_dir, exist_ok=True)


#=============================================================================#
#                      Generator and Critic Models
#=============================================================================#
lr_grid_size   = 20
input_channels = 3
base_channels  = 16 
crop_size      = 2
scale_factor   = 2

generator = DMSRGenerator(
    lr_grid_size, input_channels, base_channels, crop_size, scale_factor
)

hr_grid_size      = generator.output_size
critic_input_size = hr_grid_size
input_channels    = 8
base_channels     = 16

critic = DMSRCritic(
    critic_input_size, input_channels, base_channels
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
lr_padding = generator.padding

# data_directory = 'path/to/training/data/directory'
# data = load_numpy_dataset(data_directory)
data = generate_mock_data(lr_grid_size, hr_grid_size, channels=3, samples=4)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data

dataset = DMSRDataset(
    LR_data.float(), HR_data.float(), augment=True
)

dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


#=============================================================================#
#                           Validation Dataset
#=============================================================================#
# data_directory = 'path/to/validation/data/directory'
# data = load_numpy_dataset(data_directory)

data = generate_mock_data(lr_grid_size, hr_grid_size, channels=3, samples=8)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data


#=============================================================================#
#                              DMSR WGAN
#=============================================================================#
gan = DMSRWGAN(generator, critic, device)
gan.set_dataset(dataloader, batch_size, box_size)
gan.set_optimizer(optimizer_c, optimizer_g)

# gan.load('./level_0_run/checkpoints/current_model/')


#=============================================================================#
#                               Monitor
#=============================================================================#
from dmsr.monitors import MonitorManager, LossMonitor
from dmsr.monitors import SamplesMonitor, CheckpointMonitor
from dmsr.monitors import UpscaleMonitor

lr_sample = LR_data[2:3, ...].float()
hr_sample = HR_data[2:3, ...].float()
lr_box_size = 20 * box_size / 16
hr_box_size = box_size

checkpoint_dir = output_dir + 'checkpoints/'
samples_dir    = output_dir + 'samples/'

monitors = {
    'loss_monitor' : LossMonitor(output_dir),
    
    'samples_monitor' : SamplesMonitor(
        generator, 
        lr_sample, hr_sample,
        device,
        samples_dir = samples_dir
    ),
    
    'checkpoint_monitor' : CheckpointMonitor(
        gan,
        checkpoint_dir = checkpoint_dir
    )
}

realisations = 1
upscaling_monitor = UpscaleMonitor(
    gan,
    realisations,
    device,
    checkpoint_dir = checkpoint_dir
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

batch_report_rate = 1
monitor_manager = MonitorManager(batch_report_rate, device)
monitor_manager.set_monitors(monitors)
gan.set_monitor(monitor_manager)


#=============================================================================#
#                         Supervised Training
#=============================================================================#
# from dmsr.dmsr_gan import SupervisedMonitor

# valid_dataset = DMSRDataset(
#     LR_data.float(), HR_data.float(), augment=True
# )

# valid_dataloader = DataLoader(
#     valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True
# )

# validator = SupervisedMonitor(generator, valid_dataloader, device)

supervised_epochs = 1
gan.train(
    supervised_epochs, 
    train_step = gan.generator_supervised_step
)
gan.train(
    supervised_epochs, 
    train_step = gan.critic_supervised_step
)


#=============================================================================#
#                           WGAN Training
#=============================================================================#

num_epochs = 1
gan.train(num_epochs)