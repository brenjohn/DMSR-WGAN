#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 18:10:09 2025

@author: brennan
"""

import os
import torch

from torch.utils.data import Dataset
from dmsr.data_tools import load_numpy_tensor


class SpectraDataset(Dataset):
    """A Dataset class for managing data for the upscale_monitor class.
    """
    
    def __init__(
            self, 
            lr_position_dir, 
            hr_spectrum_dir,
            lr_velocity_dir   = None,
            scale_factor_file = None,
            summary_stats     = None,
        ):
        self.lr_position   = lr_position_dir
        self.hr_spectrum   = hr_spectrum_dir
        self.lr_velocity   = lr_velocity_dir
        self.scale_factors = scale_factor_file
        self.summary_stats = summary_stats
        
        self.velocities_included = lr_velocity_dir is not None
        self.scale_factors_included = scale_factor_file is not None
        
        if self.scale_factors_included:
            self.scale_factors = load_numpy_tensor(scale_factor_file)
            
        self.num_patches = len(os.listdir(lr_position_dir))
        
        
    def __len__(self):
        return self.num_patches
    
    
    def __getitem__(self, idx):
        patch_name = f'patch_{idx}.npy'
        lr_data = load_numpy_tensor(self.lr_position + patch_name)
        hr_data = load_numpy_tensor(self.hr_spectrum + patch_name)
        lr_data = self.scale(lr_data, 'LR_disp_fields_std')
        
        if self.velocities_included:
            lr_velocity = load_numpy_tensor(self.lr_velocity + patch_name)
            lr_velocity = self.scale(lr_velocity, 'LR_vel_fields_std')
            lr_data = torch.concat((lr_data, lr_velocity))
          
        if self.scale_factors_included:
            style = self.scale_factors[idx]
            return lr_data, hr_data, style
        
        return lr_data, hr_data, None
    
    
    def scale(self, data, std_name):
        return data / self.summary_stats[std_name]