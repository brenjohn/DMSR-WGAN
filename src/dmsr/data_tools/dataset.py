#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:45:00 2024

@author: brennan

This file defines a DMSR-Dataset class for handling training data for a 
DMSR-WGAN model.
"""

import os
import torch

from torch.utils.data import Dataset
from .loaders import load_numpy_tensor
from .augmentation import permute_tensor


class PatchDataSet(Dataset):
    """A Dataset class for holding training data for the DMSR-WGAN.
    """
    
    def __init__(
            self, 
            lr_position_dir, 
            hr_position_dir,
            lr_velocity_dir   = None,
            hr_velocity_dir   = None,
            scale_factor_file = None,
            summary_stats     = None,
            augment           = True
        ):
        self.lr_position   = lr_position_dir
        self.hr_position   = hr_position_dir
        self.lr_velocity   = lr_velocity_dir
        self.hr_velocity   = hr_velocity_dir
        self.scale_factors = scale_factor_file
        self.summary_stats = summary_stats
        self.augment       = augment
        
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
        hr_data = load_numpy_tensor(self.hr_position + patch_name)
        lr_data = self.scale(lr_data, 'LR_disp_fields_std')
        hr_data = self.scale(hr_data, 'HR_disp_fields_std')

        # Apply augmentation (random flip/permutation) if specified
        if self.augment:
            random_perm = torch.randperm(3)
            lr_data = permute_tensor(lr_data, random_perm)
            hr_data = permute_tensor(hr_data, random_perm)
                
        if self.velocities_included:
            lr_velocity = load_numpy_tensor(self.lr_velocity + patch_name)
            hr_velocity = load_numpy_tensor(self.hr_velocity + patch_name)
            lr_velocity = self.scale(lr_velocity, 'LR_vel_fields_std')
            hr_velocity = self.scale(hr_velocity, 'HR_vel_fields_std')
            
            if self.augment:
                lr_velocity = permute_tensor(lr_velocity, random_perm)
                hr_velocity = permute_tensor(hr_velocity, random_perm)
            
            lr_data = torch.concat((lr_data, lr_velocity))
            hr_data = torch.concat((hr_data, hr_velocity))
          
        if self.scale_factors_included:
            style = self.scale_factors[idx]
            return lr_data, hr_data, style
        
        return lr_data, hr_data
    
    
    def scale(self, data, std_name):
        return data / self.summary_stats[std_name]