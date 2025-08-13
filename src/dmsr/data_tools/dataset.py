#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:45:00 2024

@author: brennan

This file defines a DMSR-Dataset class for handling training data for a 
DMSR-WGAN model.
"""

import h5py
import torch

from torch.utils.data import Dataset
from .augmentation import permute_tensor


class PatchDataSet(Dataset):
    """A Dataset class for holding training data for the DMSR-WGAN.
    """
    
    def __init__(
            self, 
            data_dir, 
            include_velocities    = False,
            include_scale_factors = False,
            summary_stats         = None,
            augment               = True
        ):
        self.data_dir      = data_dir / 'patches/'
        self.velocities    = include_velocities
        self.scale_factors = include_scale_factors
        self.summary_stats = summary_stats
        self.augment       = augment
            
        self.num_patches = len(list(self.data_dir.iterdir()))
        
        
    def __len__(self):
        return self.num_patches
    
    
    def __getitem__(self, idx):
        patch_name = self.data_dir / f'patch_{idx}.h5'
        
        with h5py.File(patch_name, 'r') as patch:
            lr_data = patch['LR_Coordinates'][()]
            hr_data = patch['HR_Coordinates'][()]
            lr_data = self.scale(lr_data, 'LR_Coordinates_std')
            hr_data = self.scale(hr_data, 'HR_Coordinates_std')
            
            # Apply augmentation (random flip/permutation) if specified
            if self.augment:
                random_perm = torch.randperm(3)
                lr_data = permute_tensor(lr_data, random_perm)
                hr_data = permute_tensor(hr_data, random_perm)
                
            if self.velocities:
                lr_velocity = patch['LR_Velocities'][()]
                hr_velocity = patch['HR_Velocities'][()]
                lr_velocity = self.scale(lr_velocity, 'LR_Velocities_std')
                hr_velocity = self.scale(hr_velocity, 'HR_Velocities_std')
            
                if self.augment:
                    lr_velocity = permute_tensor(lr_velocity, random_perm)
                    hr_velocity = permute_tensor(hr_velocity, random_perm)
                
                lr_data = torch.concat((lr_data, lr_velocity))
                hr_data = torch.concat((hr_data, hr_velocity))
          
            if self.scale_factors:
                style = patch.attrs['scale_factor']
                style = torch.tensor([[style]]).float()
            
        if self.scale_factors:
            return lr_data, hr_data, style
        
        return lr_data, hr_data
    
    
    def scale(self, data, std_name):
        data = torch.from_numpy(data).float()
        return data / self.summary_stats[std_name]