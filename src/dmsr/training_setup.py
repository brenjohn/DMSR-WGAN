#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:09:00 2025

@author: brennan
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from dmsr.wgan import DMSRWGAN, DMSRCritic, DMSRGenerator
from dmsr.data_tools import PatchDataSet, SpectraDataset
from dmsr.monitors import MonitorManager, LossMonitor, SamplesMonitor
from dmsr.monitors import CheckpointMonitor, SpectrumMonitor


def setup_environment():
    """
    Initializes the PyTorch training environment for DDP or single-GPU mode.

    It detects DDP based on the 'LOCAL_RANK' environment variable. If in DDP
    mode, it initializes the NCCL process group and prints the world size.
    It falls back to CPU if no CUDA devices are available in single-GPU mode.

    Returns:
        tuple: (
            is_distributed (bool): True if running under torchrun/DDP.
            is_main_process (bool): True if this is rank 0.
            device (torch.device): The determined device (e.g., cuda:0 or cpu).
        )
    """
    is_distributed = 'LOCAL_RANK' in os.environ

    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        is_main_process = (rank == 0)
        torch.cuda.set_device(local_rank)
        device = torch.device(local_rank)

    else:
        rank = 0
        world_size = 1
        device_index = 0
        is_main_process = True
        if torch.cuda.is_available():
            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")
        else:
            device = torch.device("cpu")
            
    if is_main_process:
        print(f'World size = {world_size}')

    return is_distributed, is_main_process, device



def clean_up_environment(is_distributed):
    """
    Cleans up the training environment, primarily by destroying the DDP 
    process group if one was initialized.

    Args:
        is_distributed (bool): Flag indicating if DDP was initialized.
    """
    if is_distributed:
        dist.destroy_process_group()



def setup_wgan(params, device, is_distributed):
    """
    Initializes or loads the DMSR-WGAN model and sets up optimizers.

    The function checks the 'params' dictionary for a 'gan' key. If present, 
    it attempts to load a model from the specified checkpoint path. 
    Otherwise, it creates a new Generator and Critic based on 'params' 
    and initializes a new DMSRWGAN instance with optimizers.

    Args:
        params (dict): Dictionary of configuration parameters.
        device (torch.device): The device (GPU or CPU) to move the models to.
        is_distributed (bool): Flag indicating if DDP is active.

    Returns:
        dmsr.wgan.DMSRWGAN: The initialized or loaded WGAN model instance.
    """
    if 'gan' in params:
        gan = DMSRWGAN.load(Path(params['gan']), device, is_distributed)
                            
    else:
        generator = DMSRGenerator(**params["generator"])
        critic = DMSRCritic(**params["critic"])
        generator.to(device)
        critic.to(device)
        gan = DMSRWGAN(generator, critic, device, distributed = is_distributed)
        gan.set_optimizers(**params["optimizers"])
        
    return gan



def setup_dataloader(params, output_dir, is_main_process, is_distributed):
    """
    Loads training data and its summary statistics, and creates the DataLoader.

    The main process saves the loaded metadata and summary statistics to
    the output directory. It uses DistributedSampler if DDP is active.

    Args:
        params (dict): Dictionary of parameters.
        output_dir (pathlib.Path): Root directory for saving results.
        is_main_process (bool): True if this is the main process (rank 0).
        is_distributed (bool): True if running in DDP mode.

    Returns:
        tuple: (
            dataloader    : The configured training data loader.
            summary_stats : Dict of statistics (position std deviation etc).
            metadata      : Dict of metadata (box size, etc.).
        )
    """
    data_directory = Path(params["data"]["train_data_dir"])
    
    metadata = np.load(
        data_directory / 'metadata.npy', allow_pickle=True
    ).item()
    
    summary_stats = np.load(
        data_directory / 'summary_stats.npy', allow_pickle=True
    ).item()
    
    if is_main_process:
        np.save(output_dir / 'normalisation.npy', summary_stats)
        np.save(output_dir / 'metadata.npy', metadata)
    
    dataset = PatchDataSet(
        data_dir      = data_directory,
        summary_stats = summary_stats,
        **params["data"]
    )
    
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        shuffle_flag = False 
        
    else:
        sampler = None 
        shuffle_flag = True 
    
    batch_size = params["batch_size"]
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_flag,
        pin_memory=True,
        drop_last=False,
        num_workers=4,
        prefetch_factor=8,
        sampler=sampler
    )
    
    return dataloader, summary_stats, metadata



def setup_monitors(
        params, 
        gan, 
        summary_stats, 
        metadata, 
        is_main_process, 
        output_dir, 
        device
    ):
    """
    Initializes the MonitorManager and sets up all training monitors 
    (Loss, Samples, Checkpoint, Spectrum).

    Monitoring objects are only instantiated and configured on the 
    main process (rank 0).

    Args:
        params          : Dictionary of configuration parameters.
        gan             : The initialized WGAN model instance.
        summary_stats   : Training data summary statistics.
        metadata        : Dataset metadata required for SpectrumMonitor.
        is_main_process : True if this is the main process (rank 0).
        output_dir      : Root directory for saving monitor outputs.
        device          : The device used for monitoring operations.
    """
    batch_report_rate = params["monitoring"]["batch_report_rate"]
    monitor_manager = MonitorManager(batch_report_rate, device)
    gan.set_monitor(monitor_manager)
    
    if is_main_process:
        valid_data_directory = Path(params["data"]["valid_data_dir"])
        checkpoint_dir = output_dir / 'checkpoints/'
        
        monitors = {}
        for name, monitor_params in params["monitoring"]["monitors"].items():
            if name.startswith('samples'):
                monitors[name] = SamplesMonitor(
                    gan,
                    valid_data_directory,
                    summary_stats = summary_stats,
                    samples_dir   = output_dir / name,
                    **monitor_params
                )
                
            elif name.startswith('loss'):
                monitors[name] = LossMonitor(output_dir)
                
            elif name.startswith('checkpoint'):
                monitors[name] = CheckpointMonitor(gan, checkpoint_dir)
                
            elif name.startswith('spectrum'):
                spectra_data = SpectraDataset(
                    data_dir      = valid_data_directory,
                    summary_stats = summary_stats,
                    **params["data"]
                )
                
                monitors[name] = SpectrumMonitor(
                    gan,
                    spectra_data,
                    metadata['HR_patch_length'],
                    metadata['HR_patch_size'],
                    metadata['HR_mass'],
                    summary_stats,
                    checkpoint_dir = checkpoint_dir
                )
        
        monitor_manager.set_monitors(monitors)