#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:45:00 2024

@author: brennan

This file defines a DMSR-Dataset class for handling training data for a 
DMSR-WGAN model.
"""

import torch

from torch.utils.data import Dataset
from ..field_operations.augmentation import permute_tensor


class DMSRDataset(Dataset):
    """A Dataset class for holding training data for the DMSR-WGAN.
    """
    
    def __init__(
            self, 
            lr_position, 
            hr_position,
            lr_velocity  = None,
            hr_velocity  = None,
            scale_factor = None,
            augment      = True
        ):
        self.lr_position  = lr_position
        self.hr_position  = hr_position
        self.lr_velocity  = lr_velocity
        self.hr_velocity  = hr_velocity
        self.scale_factor = scale_factor
        self.augment      = augment
        
        self.velocities_included = lr_velocity is not None
        self.scale_factors_included = scale_factor is not None
        
        
    
    def __len__(self):
        return self.lr_position.size(0)
    
    
    def __getitem__(self, idx):
        lr_data = self.lr_position[idx]
        hr_data = self.hr_position[idx]

        # Apply augmentation (random flip/permutation) if specified
        if self.augment:
            random_perm = torch.randperm(3)
            lr_data = permute_tensor(lr_data, random_perm)
            hr_data = permute_tensor(hr_data, random_perm)
                
        if self.velocities_included:
            lr_velocity = self.lr_velocity[idx]
            hr_velocity = self.hr_velocity[idx]
            
            if self.augment:
                lr_velocity = permute_tensor(lr_velocity, random_perm)
                hr_velocity = permute_tensor(hr_velocity, random_perm)
            
            lr_data = torch.concat((lr_data, lr_velocity))
            hr_data = torch.concat((hr_data, hr_velocity))
          
        if self.scale_factors_included:
            style = self.scale_factor[idx]
            return lr_data, hr_data, style
        
        return lr_data, hr_data
    
    
    def normalise_dataset(self):
        """Scales position and velocity data by dividing by their respective
        standard deviations. The standard deviations are also returned as a
        dictionary.
        """
        params = {}
        field_names = ["lr_position", "hr_position"]
        if self.velocities_included:
            field_names += ["lr_velocity", "hr_velocity"]
        
        for field in field_names:
            standard_deviation = vars(self)[field].std()
            params[field + "_std"] = standard_deviation.item()
            vars(self)[field] /= standard_deviation
        
        return params