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
from .augmentation import permute_tensor, flip_tensor


class PatchDataSet(Dataset):
    """A Dataset class for holding training data for the DMSR-WGAN.
    """
    
    def __init__(
            self, 
            data_dir, 
            include_velocities    = False,
            include_scale_factors = False,
            summary_stats         = None,
            augment               = True,
            **kwargs
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
            lr_data = self.scale(lr_data, 'LR_Coordinates')
            hr_data = self.scale(hr_data, 'HR_Coordinates')
            
            # Apply augmentation (random flip/permutation) if specified
            if self.augment:
                random_flip = torch.randint(0, 2, (3,), dtype=torch.bool)
                lr_data = flip_tensor(lr_data, random_flip)
                hr_data = flip_tensor(hr_data, random_flip)
                
                random_perm = torch.randperm(3)
                lr_data = permute_tensor(lr_data, random_perm)
                hr_data = permute_tensor(hr_data, random_perm)
                
            if self.velocities:
                lr_velocity = patch['LR_Velocities'][()]
                hr_velocity = patch['HR_Velocities'][()]
                lr_velocity = self.scale(lr_velocity, 'LR_Velocities')
                hr_velocity = self.scale(hr_velocity, 'HR_Velocities')
            
                if self.augment:
                    lr_velocity = flip_tensor(lr_velocity, random_flip)
                    hr_velocity = flip_tensor(hr_velocity, random_flip)
                    lr_velocity = permute_tensor(lr_velocity, random_perm)
                    hr_velocity = permute_tensor(hr_velocity, random_perm)
                
                lr_data = torch.concat((lr_data, lr_velocity))
                hr_data = torch.concat((hr_data, hr_velocity))
          
            if self.scale_factors:
                style = patch.attrs['scale_factor']
                style = torch.tensor([[style]]).float()
            
        if self.scale_factors:
            return lr_data, hr_data, style
        
        return lr_data.detach(), hr_data.detach()
    
    
    def scale(self, data, base_name):
        data = torch.from_numpy(data).float()
        std = self.summary_stats[base_name + '_std']
        return data / std
