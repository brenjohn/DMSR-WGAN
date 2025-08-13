#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:27:20 2025

@author: brennan
"""

import h5py
import torch
import numpy as np

from pathlib import Path
from .monitor import Monitor


class SamplesMonitor(Monitor):
    """A monitor class for saving super resolution samples created by the
    generator during training.
    """
    
    def __init__(
            self,
            gan,
            data_directory,
            patch_number,
            device,
            velocities    = False,
            scale_factors = False,
            summary_stats = None,
            samples_dir   = Path('./data/samples/'),
            seed          = 42
        ):
        
        self.device = device
        self.generator = gan.generator
        self.summary_stats = summary_stats
        
        lr_sample, hr_sample, style = self.get_sample(
            data_directory, patch_number, velocities, scale_factors
        )
        
        self.lr_sample = lr_sample
        self.hr_sample = hr_sample
        self.style = style
        
        torch_generator = torch.Generator(device).manual_seed(seed)
        z = self.generator.sample_latent_space(1, device, torch_generator)
        self.z = [(z0.cpu(), z1.cpu()) for z0, z1 in z]
        
        self.samples_dir = samples_dir
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.samples_dir / 'lr_sample.npy', lr_sample.numpy())
        np.save(self.samples_dir / 'hr_sample.npy', hr_sample.numpy())
        
        
    def get_sample(
            self, 
            data_dir, 
            patch_num, 
            velocities,
            scale_factors
        ):
        patch_name = data_dir / f'patches/patch_{patch_num}.h5'
        style = None
        
        with h5py.File(patch_name, 'r') as patch:
            lr_data = patch['LR_Coordinates'][()]
            hr_data = patch['HR_Coordinates'][()]
            lr_data = self.scale(lr_data, 'LR_Coordinates_std')
            hr_data = self.scale(hr_data, 'HR_Coordinates_std')
        
            if velocities:
                lr_velocity = patch['LR_Velocities'][()]
                hr_velocity = patch['HR_Velocities'][()]
                lr_velocity = self.scale(lr_velocity, 'LR_Velocities_std')
                hr_velocity = self.scale(hr_velocity, 'HR_Velocities_std')
            
                lr_data = torch.concat((lr_data, lr_velocity))
                hr_data = torch.concat((hr_data, hr_velocity))
                
            if scale_factors:
                style = patch.attrs['scale_factor']
                style = torch.tensor([[style]]).float()
        
        return lr_data[None, ...], hr_data[None, ...], style
        
        
    def post_epoch_processing(self, epoch):
        # Move data to the device and use the generator to create fake data.
        lr_sample = self.lr_sample.to(self.device)
        z = [(z0.to(self.device), z1.to(self.device)) for z0, z1 in self.z]
        style = None 
        if self.style is not None:
            style = self.style.to(self.device)
        sr_sample = self.generator(lr_sample, z, style)
        
        # Move the fake data to the cpu and save.
        sr_sample = sr_sample.detach().cpu()
        filename = self.samples_dir / f'sr_sample_{epoch:04}.npy'
        np.save(filename, sr_sample.numpy())
        
    
    def scale(self, data, std_name):
        data = torch.from_numpy(data).float()
        if self.summary_stats is not None:
            data = data / self.summary_stats[std_name]
        return data