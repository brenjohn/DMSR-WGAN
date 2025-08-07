#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 18:10:09 2025

@author: brennan
"""

import h5py
import torch

from torch.utils.data import Dataset


class SpectraDataset(Dataset):
    """A Dataset class for managing data for the upscale_monitor class.
    """
    
    def __init__(
            self, 
            data_dir, 
            include_velocities    = False,
            include_scale_factors = False,
            summary_stats     = None,
        ):
        self.data_dir      = data_dir
        self.velocities    = include_velocities
        self.scale_factors = include_scale_factors
        self.summary_stats = summary_stats
            
        self.num_patches = len(list(data_dir.iterdir()))
        
        
    def __len__(self):
        return self.num_patches
    
    
    def __getitem__(self, idx):
        patch_name = self.data_dir / f'patches/patch_{idx}.h5'
        style = None
        
        with h5py.File(patch_name, 'r') as patch:
            lr_data = patch['LR_Coordinates'][()]
            hr_data = patch['HR_Power_Spectrum'][()]
            lr_data = self.scale(lr_data, 'LR_disp_fields_std')
            
            if self.velocities:
                lr_velocity = patch['LR_Velocities'][()]
                lr_velocity = self.scale(lr_velocity, 'LR_vel_fields_std')
                lr_data = torch.concat((lr_data, lr_velocity))
                
            if self.scale_factors:
                style = patch.attrs['scale_factor']
                style = torch.tensor([[style]]).float()
        
        return lr_data, hr_data, style
    
    
    def scale(self, data, std_name):
        data = torch.from_numpy(data).float()
        return data / self.summary_stats[std_name]