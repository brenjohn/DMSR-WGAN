#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 14:32:36 2025

@author: brennan
"""

import os
import numpy as np
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler

from dmsr.wgan import DMSRWGAN, DMSRCritic, DMSRGenerator
from dmsr.data_tools import PatchDataSet, SpectraDataset
from dmsr.monitors import MonitorManager, LossMonitor, SamplesMonitor
from dmsr.monitors import CheckpointMonitor, SpectrumMonitor


def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Starting DDP on rank {rank}/{world_size}.")


def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


def main(rank, world_size):
    #================================ DDP SETUP ==============================#
    setup_ddp(rank, world_size)
    device = torch.device(rank)
    is_main_process = (rank == 0)

    # --- Configuration ---
    # TODO: These could be moved to argparse for better control
    num_epochs = 2
    batch_size = 8
    output_dir = Path('./tmp_run_ddp/')
    
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
        style_size
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

    # Wrap models with DDP
    generator = DDP(generator, device_ids=[device])
    critic = DDP(critic, device_ids=[device])

    #=========================================================================#
    #                            Optimizers
    #=========================================================================#
    b1, b2 = 0.0, 0.99
    
    lr_G = 0.00001
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    
    lr_C = 0.00002
    optimizer_c = optim.Adam(critic.parameters(), lr=lr_C, betas=(b1, b2))


    #=========================================================================#
    #                          Training Dataset
    #=========================================================================#
    data_directory = Path('../../data/dmsr_style_train/').resolve()
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
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=2,
        prefetch_factor=8,
        sampler=sampler
    )

    #=========================================================================#
    #                          DMSR WGAN Setup
    #=========================================================================#
    gan = DMSRWGAN(generator, critic, device)
    gan.set_dataset(
        dataloader, 
        batch_size, 
        metadata['box_size'] / training_summary_stats['HR_Coordinates_std']
    )
    gan.set_optimizer(optimizer_c, optimizer_g)


    #=========================================================================#
    #                   Monitors (Run only on main process)
    #=========================================================================#
    batch_report_rate = 16
    monitor_manager = MonitorManager(batch_report_rate, device)
    gan.set_monitor(monitor_manager)
    
    if is_main_process:
        valid_data_directory = Path('../../data/dmsr_style_valid/').resolve()
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
        
        monitor_manager.set_monitors(monitors)


    #=========================================================================#
    #                           WGAN Training
    #=========================================================================#
    gan.train(num_epochs)
    cleanup_ddp()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        main, 
        args=(world_size,),
        nprocs=world_size
    )