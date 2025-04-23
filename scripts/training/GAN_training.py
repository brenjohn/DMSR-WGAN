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
from swift_tools.data import load_numpy_dataset


# Check if CUDA is available and set the device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#=============================================================================#
#                      Generator and Critic Models
#=============================================================================#
lr_grid_size       = 20
generator_channels = 64 
crop_size          = 2
scale_factor       = 2

generator = DMSRGenerator(
    lr_grid_size, generator_channels, crop_size, scale_factor
)

hr_grid_size      = generator.output_size
density_size      = hr_grid_size
displacement_size = hr_grid_size
density_channels  = 16
main_channels     = 64

critic = DMSRCritic(
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
data_directory = '../../data/dmsr_training/'
batch_size = 4

data = load_numpy_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data

lr_padding = 2

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


#=============================================================================#
#                              DMSR WGAN
#=============================================================================#
gan = DMSRWGAN(generator, critic, device)
gan.set_dataset(dataloader, batch_size, box_size, lr_padding, scale_factor)
gan.set_optimizer(optimizer_c, optimizer_g)

# gan.load('./level_0_run/checkpoints/current_model/')


#=============================================================================#
#                               Monitor
#=============================================================================#
from dmsr.dmsr_gan.dmsr_monitor import MonitorManager, LossMonitor
from dmsr.dmsr_gan.dmsr_monitor import SamplesMonitor, CheckpointMonitor
from dmsr.dmsr_gan.dmsr_monitor import UpscaleMonitor

lr_sample = LR_data[2:3, ...].float()
hr_sample = HR_data[2:3, ...].float()
lr_box_size = 20 * box_size / 16
hr_box_size = box_size

output_dir     = './level_0_restart/'
checkpoint_dir = output_dir + 'checkpoints/'
samples_dir    = output_dir + 'samples/'

monitors = {
    'loss_monitor' : LossMonitor(output_dir),
    
    'samples_monitor' : SamplesMonitor(
        generator, 
        lr_sample, hr_sample, 
        lr_box_size, hr_box_size, 
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

# gan.train(
#     supervised_epochs, 
#     train_step = gan.generator_supervised_step, 
#     validator = validator
# )
# gan.train(5, train_step=gan.critic_supervised_step)


#=============================================================================#
#                           WGAN Training
#=============================================================================#

num_epochs = 2
gan.train(num_epochs)