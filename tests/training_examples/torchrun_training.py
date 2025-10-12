#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 13:05:16 2025

@author: brennan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 21:17:05 2025

@author: brennan
"""

import os
import numpy as np
import torch

import torch.distributed as dist

from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler

from dmsr.wgan import DMSRWGAN, DMSRCritic, DMSRGenerator
from dmsr.data_tools import PatchDataSet, SpectraDataset
from dmsr.data_tools import generate_mock_dataset
from dmsr.monitors import MonitorManager, LossMonitor, SamplesMonitor
from dmsr.monitors import CheckpointMonitor, SpectrumMonitor


def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")


def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


def main():
    #================================ DDP SETUP ==============================#
    setup_ddp()
    rank = int(os.environ["RANK"])
    device = torch.device(int(os.environ["LOCAL_RANK"]))
    is_main_process = (rank == 0)

    # --- Configuration ---
    # TODO: These could be moved to argparse for better control
    num_epochs = 2
    batch_size = 8
    output_dir = Path('./test_run/')
    
    # Only the main process should create directories
    if is_main_process:
        output_dir.mkdir(exist_ok=True)
    
    #=========================================================================#
    #                        Generator and Critic Models
    #=========================================================================#
    lr_grid_size   = 20
    input_channels = 6
    base_channels  = 2
    crop_size      = 1
    upscale_factor = 2
    style_size     = 1
    
    generator = DMSRGenerator(
        lr_grid_size, 
        input_channels, 
        base_channels, 
        crop_size, 
        upscale_factor, 
        style_size,
        nn_distance=True
    )
    
    hr_grid_size      = generator.output_size
    critic_input_size = hr_grid_size - 2
    input_channels    = 3 + 3 + 3 + 3 + 6 + 1
    base_channels     = 2
    
    critic = DMSRCritic(
        critic_input_size, 
        input_channels, 
        base_channels,
        style_size = style_size,
        use_nn_distance_features = True
    )

    # Move models to the correct device for this process
    generator.to(device)
    critic.to(device)


    #=========================================================================#
    #                          Training Dataset
    #=========================================================================#
    data_directory = Path('./data/test_train/')
    if is_main_process:
        generate_mock_dataset(
            data_dir           = data_directory, 
            num_patches        = 16,
            lr_grid_size       = lr_grid_size,
            hr_grid_size       = hr_grid_size,
            lr_padding         = generator.padding,
            hr_padding         = 0,
            include_velocities = True, 
            include_scales     = True,
            include_spectra    = False
        )
    dist.barrier()
    batch_size = 8
    
    metadata = np.load(
        data_directory / 'metadata.npy', allow_pickle=True
    ).item()
    
    training_summary_stats = np.load(
        data_directory / 'summary_stats.npy', allow_pickle=True
    ).item()
    
    if is_main_process:
        np.save(output_dir / 'normalisation.npy', training_summary_stats)
        np.save(output_dir / 'metadata.npy', metadata)
    
    dataset = PatchDataSet(
        data_dir              = data_directory,
        include_velocities    = True,
        include_scale_factors = True,
        summary_stats         = training_summary_stats,
        augment               = True
    )
    
    sampler = DistributedSampler(
        dataset, shuffle=True, drop_last=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=2,
        prefetch_factor=8,
        sampler=sampler
    )


    #=========================================================================#
    #                          DMSR WGAN Setup
    #=========================================================================#
    gan = DMSRWGAN(generator, critic, device, distributed = True)
    
    gan.set_dataset(
        dataloader, 
        batch_size, 
        metadata['box_size'] / training_summary_stats['HR_Coordinates_std']
    )
    
    gan.set_optimizers(
        lr_G = 0.00001,
        lr_C = 0.00002,
        b1   = 0.0,
        b2   = 0.99
    )


    #=========================================================================#
    #                   Monitors (Run only on main process)
    #=========================================================================#
    batch_report_rate = 16
    monitor_manager = MonitorManager(batch_report_rate, device)
    gan.set_monitor(monitor_manager)
    
    if is_main_process:
        valid_data_directory = Path('./data/test_valid/')
        generate_mock_dataset(
            data_dir           = valid_data_directory, 
            num_patches        = 4,
            lr_grid_size       = metadata['LR_patch_size'],
            hr_grid_size       = metadata['HR_patch_size'],
            lr_padding         = metadata['LR_padding'],
            hr_padding         = 0,
            include_velocities = True, 
            include_scales     = True,
            include_spectra    = True
        )
        spectra_data = SpectraDataset(
            data_dir              = valid_data_directory, 
            include_velocities    = True,
            include_scale_factors = True,
            summary_stats         = training_summary_stats,
        )
        
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
        
        monitor_manager.set_monitors(monitors)


    #=========================================================================#
    #                           WGAN Training
    #=========================================================================#
    gan.train(num_epochs)
    cleanup_ddp()
    
    # Additional test for the load method.
    if is_main_process:
        gan = DMSRWGAN.load(checkpoint_dir / 'current_model', device)


if __name__ == '__main__':
    main()