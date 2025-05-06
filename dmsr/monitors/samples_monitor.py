#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:27:20 2025

@author: brennan
"""

import os
import torch
import numpy as np

from .monitor import Monitor
from ..data_tools import load_numpy_tensor


class SamplesMonitor(Monitor):
    """A monitor class for saving super resolution samples created by the
    generator during training.
    """
    
    def __init__(
            self,
            generator,
            data_directory,
            patch_number,
            device,
            include_velocity = True,
            include_style = True,
            summary_stats = None,
            samples_dir = './data/samples/'
        ):
        
        self.device = device
        self.generator = generator
        self.summary_stats = summary_stats
        
        lr_sample, hr_sample, style = self.get_sample(
            data_directory, patch_number, include_velocity, include_style
        )
        
        self.lr_sample = lr_sample
        self.hr_sample = hr_sample
        self.style = style
        
        z = generator.sample_latent_space(1, device)
        self.z = [(z0.cpu(), z1.cpu()) for z0, z1 in z]
        
        self.samples_dir = samples_dir
        os.makedirs(self.samples_dir, exist_ok=True)
        np.save(self.samples_dir + 'lr_sample.npy', lr_sample.numpy())
        np.save(self.samples_dir + 'hr_sample.npy', hr_sample.numpy())
        
        
    def get_sample(
            self, 
            data_dir, 
            patch_num, 
            include_velocity,
            include_style
        ):
        patch_name = f'patch_{patch_num}.npy'
        lr_data = load_numpy_tensor(data_dir + 'LR_disp_fields/' + patch_name)
        hr_data = load_numpy_tensor(data_dir + 'HR_disp_fields/' + patch_name)
        style = None
        
        if self.summary_stats is not None:
            lr_data /= self.summary_stats['LR_disp_fields_std']
            hr_data /= self.summary_stats['HR_disp_fields_std']
        
        if include_velocity:
            lr_velocity = load_numpy_tensor(
                data_dir + 'LR_vel_fields/' + patch_name
            )
            hr_velocity = load_numpy_tensor(
                data_dir + 'HR_vel_fields/' + patch_name
            )
            
            if self.summary_stats is not None:
                lr_velocity /= self.summary_stats['LR_vel_fields_std']
                hr_velocity /= self.summary_stats['HR_vel_fields_std']
            
            lr_data = torch.concat((lr_data, lr_velocity))
            hr_data = torch.concat((hr_data, hr_velocity))
            
        if include_style:
            styles = load_numpy_tensor(data_dir + 'scale_factors.npy')
            style = styles[patch_num][None, ...]
        
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
        filename = self.samples_dir + f'sr_sample_{epoch:04}.npy'
        np.save(filename, sr_sample.numpy())