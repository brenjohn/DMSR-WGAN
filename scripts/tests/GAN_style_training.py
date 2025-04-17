#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:53:01 2025

@author: brennan
"""

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import numpy as np

from torch import optim
from torch.utils.data import DataLoader

from dmsr.wgan import DMSRWGAN
from dmsr.wgan import DMSRCritic
from dmsr.wgan import DMSRGenerator

from dmsr.data_tools import DMSRDataset
from dmsr.data_tools import load_numpy_tensor
from dmsr.data_tools import generate_mock_data


# Check if CUDA is available and set the device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

output_dir = './velocity_run/'
os.makedirs(output_dir, exist_ok=True)


#=============================================================================#
#                      Generator and Critic Models
#=============================================================================#
lr_grid_size   = 20
input_channels = 6
base_channels  = 64 
crop_size      = 2
scale_factor   = 2
style_size     = 1

generator = DMSRGenerator(
    lr_grid_size, 
    input_channels, 
    base_channels, 
    crop_size, 
    scale_factor, 
    style_size
)

hr_grid_size         = generator.output_size
critic_input_size    = hr_grid_size
input_channels       = 20
base_channels        = 64
density_scale_factor = 2

critic = DMSRCritic(
    critic_input_size, 
    input_channels, 
    base_channels, 
    density_scale_factor, 
    style_size
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
batch_size = 4
lr_padding = generator.padding

# data_directory = '../../data/dmsr_training/'
# data = load_numpy_tensor(data_directory)
data = generate_mock_data(lr_grid_size, hr_grid_size, channels=6, samples=4)
LR_data, HR_data = data
box_size = 1

scale_factors = torch.randn((batch_size, style_size)).to(device)

# Split data into displacements and velocities.
LR_disp = LR_data[:, :3, ...].float()
LR_vel  = LR_data[:, 3:, ...].float()
HR_disp = HR_data[:, :3, ...].float()
HR_vel  = HR_data[:, 3:, ...].float()

dataset = DMSRDataset(
    LR_disp, HR_disp, LR_vel, HR_vel, scale_factors, augment=True
)

noramalisation_params = dataset.normalise_dataset()
np.save(output_dir + 'normalisation.npy', noramalisation_params)

dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

lr_position_std = noramalisation_params['lr_position_std']
hr_position_std = noramalisation_params['hr_position_std']


#=============================================================================#
#                           Validation Dataset
#=============================================================================#
# data_directory = 'path/to/validation/data/directory'
# data = load_numpy_tensor(data_directory)
num_samples = 4
data = generate_mock_data(
    lr_grid_size, hr_grid_size, channels=6, samples=num_samples
)
LR_data, HR_data = data

LR_data[:, :3, ...] /= noramalisation_params['lr_position_std']
LR_data[:, 3:, ...] /= noramalisation_params['lr_velocity_std']
HR_data[:, :3, ...] /= noramalisation_params['hr_position_std']
HR_data[:, 3:, ...] /= noramalisation_params['hr_velocity_std']
styles = torch.randn((num_samples, style_size)).to(device)


#=============================================================================#
#                              DMSR WGAN
#=============================================================================#
gan = DMSRWGAN(generator, critic, device)
gan.set_dataset(
    dataloader, 
    batch_size, 
    box_size / hr_position_std
)
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
lr_box_size = 20 * box_size / 16 / lr_position_std
hr_box_size = box_size / hr_position_std
style_sample = styles[2:3, ...]

checkpoint_dir = output_dir + 'checkpoints/'
samples_dir    = output_dir + 'samples/'

monitors = {
    'loss_monitor' : LossMonitor(output_dir),
    
    'samples_monitor' : SamplesMonitor(
        generator, 
        lr_sample, hr_sample,
        device,
        style = style_sample,
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
grid_size = int(hr_grid_size)
upscaling_monitor.set_data_set(
    LR_data.float(), 
    HR_data.float(), 
    particle_mass, 
    box_size, 
    grid_size,
    styles = styles
)
monitors['upscaling_monitor'] = upscaling_monitor

batch_report_rate = 16
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