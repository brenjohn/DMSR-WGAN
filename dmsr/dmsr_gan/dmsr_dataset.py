#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:45:00 2024

@author: brennan

This file define a DMSR-Dataset class for handling training data for a 
DMSR-WGAN model.
"""

from torch.utils.data import Dataset
from ..field_operations.augmentation import random_transformation


class DMSRDataset(Dataset):
    """A Dataset class for holding training data for the DMSR-WGAN.
    """
    
    def __init__(self, lr_data, hr_data, augment=True):
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.augment = augment

    def __len__(self):
        return self.lr_data.size(0)

    def __getitem__(self, idx):
        lr = self.lr_data[idx]
        hr = self.hr_data[idx]

        # Apply augmentation (random flip/permutation) if specified
        if self.augment:
            lr, hr = random_transformation(lr, hr)
            
        return lr, hr
