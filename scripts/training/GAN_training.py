#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:35:32 2024

@author: brennan
"""

import torch
import numpy as np

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader

from dmsr.wgan import DMSRWGAN
from dmsr.wgan import DMSRCritic
from dmsr.wgan import DMSRGenerator

from dmsr.data_tools import PatchDataSet


# Check if CUDA is available and set the device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

output_dir = Path('./tmp_run/')
output_dir.mkdir(exist_ok=True)


#=============================================================================#
#                      Generator and Critic Models
#=============================================================================#
lr_grid_size   = 20
input_channels = 6
base_channels  = 128
crop_size      = 1
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
critic_input_size    = hr_grid_size - 2
input_channels       = 3 + 3 + 3 + 3 + 6 + 1
base_channels        = 128

critic = DMSRCritic(
    critic_input_size, 
    input_channels, 
    base_channels,
    style_size = style_size,
    use_nn_distance_features = True
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
data_directory = Path('../../data/dmsr_style_train/').resolve()
batch_size = 8

metadata = np.load(data_directory / 'metadata.npy', allow_pickle=True).item()

training_summary_stats = np.load(
    data_directory / 'summary_stats.npy', allow_pickle=True
).item()
np.save(output_dir / 'normalisation.npy', training_summary_stats)
np.save(output_dir / 'metadata.npy', metadata)

dataset = PatchDataSet(
    data_dir              = data_directory,
    include_velocities    = True,
    include_scale_factors = True,
    summary_stats         = training_summary_stats,
    augment               = True
)

dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True,
    num_workers=2,
    prefetch_factor=8
)


#=============================================================================#
#                           Validation Dataset
#=============================================================================#
from dmsr.data_tools import SpectraDataset

valid_data_directory = Path('../../data/dmsr_style_valid/').resolve()

spectra_data = SpectraDataset(
    data_dir              = valid_data_directory, 
    include_velocities    = True,
    include_scale_factors = True,
    summary_stats         = training_summary_stats,
)


#=============================================================================#
#                              DMSR WGAN
#=============================================================================#
gan = DMSRWGAN(generator, critic, device)
gan.set_dataset(
    dataloader, 
    batch_size, 
    metadata['box_size'] / training_summary_stats['HR_Coordinates_std']
)
gan.set_optimizer(optimizer_c, optimizer_g)

#=============================================================================#
#                              Optimizers
#=============================================================================#
# model_dir = Path('./nn_run_b/checkpoints/current_model/')
# gan.load(model_dir)

# lr_G = 0.000001
# for g in gan.optimizer_g.param_groups:
#     g['lr'] = lr_G
    
# lr_C = 0.000002
# for g in gan.optimizer_c.param_groups:
#     g['lr'] = lr_G


#=============================================================================#
#                               Monitors
#=============================================================================#
from dmsr.monitors import MonitorManager, LossMonitor
from dmsr.monitors import SamplesMonitor, CheckpointMonitor
from dmsr.monitors import SpectrumMonitor

checkpoint_dir = output_dir / 'checkpoints/'

monitors = {
    'loss_monitor' : LossMonitor(output_dir),
    
    'samples_monitor_1' : SamplesMonitor(
        gan,
        valid_data_directory,
        patch_number     = 1,
        velocities       = True,
        scale_factors    = True,
        summary_stats    = training_summary_stats,
        samples_dir      = output_dir / 'samples_1/'
    ),
    
    'samples_monitor_191' : SamplesMonitor(
        gan,
        valid_data_directory,
        patch_number     = 191,
        velocities       = True,
        scale_factors    = True,
        summary_stats    = training_summary_stats,
        samples_dir      = output_dir / 'samples_191/'
    ),
    
    'checkpoint_monitor' : CheckpointMonitor(
        gan,
        checkpoint_dir = checkpoint_dir
    ),
    
    'spectrum_monitor' : SpectrumMonitor(
        gan,
        spectra_data,
        metadata['HR_patch_length'],
        metadata['HR_patch_size'],
        metadata['HR_mass'],
        training_summary_stats,
        checkpoint_dir = checkpoint_dir
    )
}


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

num_epochs = 512
gan.train(num_epochs)
