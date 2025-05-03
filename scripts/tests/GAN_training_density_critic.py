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
import torch.optim as optim

from torch.utils.data import DataLoader

from dmsr.wgan import DMSRWGAN
from dmsr.wgan import DMSRDensityCritic
from dmsr.wgan import DMSRGenerator

from dmsr.data_tools import PatchDataSet
from dmsr.data_tools import load_numpy_tensor
from dmsr.data_tools import generate_mock_dataset


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
density_size      = hr_grid_size
displacement_size = hr_grid_size
density_channels  = 4
main_channels     = 16

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
data_directory = './data/test_train/'
generate_mock_dataset(
    data_dir           = data_directory, 
    num_patches        = 16,
    lr_grid_size       = lr_grid_size,
    hr_grid_size       = hr_grid_size,
    lr_padding         = generator.padding,
    include_velocities = False, 
    include_scales     = False,
    include_spectra    = False
)
batch_size = 2

metadata = load_numpy_tensor(data_directory + 'metadata.npy')
box_size        = metadata[0].item()
LR_patch_length = metadata[1].item()
HR_patch_length = metadata[2].item()
LR_patch_size   = metadata[3].int().item()
HR_patch_size   = metadata[4].int().item()
LR_inner_size   = metadata[5].int().item()
padding         = metadata[6].int().item()
LR_mass         = metadata[7].item()
HR_mass         = metadata[8].item()

training_summary_stats = np.load(
    data_directory + 'summary_stats.npy', allow_pickle=True
).item()
np.save(output_dir + 'normalisation.npy', training_summary_stats)

dataset = PatchDataSet(
    lr_position_dir   = data_directory + 'LR_disp_fields/',
    hr_position_dir   = data_directory + 'HR_disp_fields/',
    summary_stats     = training_summary_stats,
    augment=True
)

dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


#=============================================================================#
#                           Validation Dataset
#=============================================================================#
# data_directory = 'path/to/validation/data/directory'
# data = load_numpy_dataset(data_directory)

# data = generate_mock_data(lr_grid_size, hr_grid_size, channels=3, samples=8)
# LR_data, HR_data = data

from dmsr.data_tools import SpectraDataset

valid_data_directory = './data/test_valid/'
generate_mock_dataset(
    data_dir           = valid_data_directory, 
    num_patches        = 4,
    lr_grid_size       = LR_patch_size,
    hr_grid_size       = HR_patch_size,
    lr_padding         = padding,
    include_velocities = False, 
    include_scales     = False,
    include_spectra    = True
)

spectra_data = SpectraDataset(
    lr_position_dir   = valid_data_directory + 'LR_disp_fields/',
    hr_spectrum_dir   = valid_data_directory + 'HR_spectra/',
    summary_stats     = training_summary_stats,
)

#=============================================================================#
#                              DMSR WGAN
#=============================================================================#
gan = DMSRWGAN(generator, critic, device)
gan.set_dataset(
    dataloader, 
    batch_size, 
    box_size / training_summary_stats['HR_disp_fields_std']
)
gan.set_optimizer(optimizer_c, optimizer_g)

# gan.load('./level_0_run/checkpoints/current_model/')


#=============================================================================#
#                               Monitor
#=============================================================================#
from dmsr.monitors import MonitorManager, LossMonitor
from dmsr.monitors import SamplesMonitor, CheckpointMonitor
from dmsr.monitors import SpectrumMonitor

checkpoint_dir = output_dir + 'checkpoints/'

monitors = {
    'loss_monitor' : LossMonitor(output_dir),
    
    'samples_monitor' : SamplesMonitor(
        generator,
        valid_data_directory,
        patch_number     = 1,
        device           = device,
        include_velocity = False,
        include_style    = False,
        samples_dir      = output_dir + 'samples/'
    ),
    
    'checkpoint_monitor' : CheckpointMonitor(
        gan,
        checkpoint_dir = checkpoint_dir
    ),
    
    'spectrum_monitor' : SpectrumMonitor(
        gan,
        spectra_data,
        HR_patch_length,
        HR_patch_size,
        training_summary_stats,
        device,
        checkpoint_dir = checkpoint_dir
    )
}


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