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
import numpy as np

from torch import optim
from torch.utils.data import DataLoader

from dmsr.wgan import DMSRWGAN
from dmsr.wgan import DMSRCritic
from dmsr.wgan import DMSRGenerator

from dmsr.data_tools import DMSRDataset
from dmsr.data_tools import load_numpy_dataset

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
base_channels  = 128
crop_size      = 2
upscale_factor = 2
style_size     = 1

generator = DMSRGenerator(
    lr_grid_size, 
    input_channels, 
    base_channels, 
    crop_size, 
    upscale_factor, 
    style_size
)

hr_grid_size         = generator.output_size
critic_input_size    = hr_grid_size
input_channels       = 20
base_channels        = 128
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
data_directory = '../../data/dmsr_style_train/'
batch_size = 4

metadata = load_numpy_dataset(data_directory + 'metadata.npy')
box_size        = metadata[0]
LR_patch_length = metadata[1]
HR_patch_length = metadata[2]
LR_patch_size   = metadata[3]
HR_patch_size   = metadata[4]
LR_inner_size   = metadata[5]
padding         = metadata[6]
LR_mass         = metadata[7]
HR_mass         = metadata[8]

LR_disp = load_numpy_dataset(data_directory + 'LR_disp_fields.npy')
LR_vel  = load_numpy_dataset(data_directory + 'LR_vel_fields.npy')
HR_disp = load_numpy_dataset(data_directory + 'HR_disp_fields.npy')
HR_vel  = load_numpy_dataset(data_directory + 'HR_vel_fields.npy')
scale_factors = load_numpy_dataset(data_directory + 'scale_factors.npy')

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
data_directory = '../../data/dmsr_style_valid/'

LR_disp = load_numpy_dataset(data_directory + 'LR_disp_fields.npy')[:16]
LR_vel  = load_numpy_dataset(data_directory + 'LR_vel_fields.npy')[:16]
HR_disp = load_numpy_dataset(data_directory + 'HR_disp_fields.npy')[:16]
HR_vel  = load_numpy_dataset(data_directory + 'HR_vel_fields.npy')[:16]
scale_factors = load_numpy_dataset(data_directory + 'scale_factors.npy')[:16]

LR_disp /= noramalisation_params['lr_position_std']
LR_vel  /= noramalisation_params['lr_velocity_std']
HR_disp /= noramalisation_params['hr_position_std']
HR_vel  /= noramalisation_params['hr_velocity_std']

LR_valid_data = torch.concat([LR_disp, LR_vel], axis=1)
HR_valid_data = torch.concat([HR_disp, HR_vel], axis=1)


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

# gan.load('./velocity_run/checkpoints/current_model/')


#=============================================================================#
#                               Monitor
#=============================================================================#
from dmsr.monitors import MonitorManager, LossMonitor
from dmsr.monitors import SamplesMonitor, CheckpointMonitor
from dmsr.monitors import UpscaleMonitor

lr_sample = LR_valid_data[2:3, ...]
hr_sample = HR_valid_data[2:3, ...]
lr_box_size = 20 * box_size / 16 / lr_position_std
hr_box_size = box_size / hr_position_std

style_sample = scale_factors[2:3, ...]

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


particle_mass = HR_mass
grid_size = int(hr_grid_size)
upscaling_monitor.set_data_set(
    LR_valid_data.float(), 
    HR_valid_data.float(), 
    particle_mass, 
    HR_patch_length, 
    grid_size,
    styles = scale_factors,
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

# supervised_epochs = 5
# gan.train(
#     supervised_epochs, 
#     train_step = gan.generator_supervised_step
# )
# gan.train(
#     supervised_epochs, 
#     train_step = gan.critic_supervised_step
# )


#=============================================================================#
#                           WGAN Training
#=============================================================================#

num_epochs = 2
gan.train(num_epochs)
