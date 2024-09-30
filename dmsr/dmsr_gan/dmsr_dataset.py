#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:45:00 2024

@author: brennan
"""

from torch.utils.data import Dataset
from ..field_operations.augmentation import random_transformation


class DMSRDataset(Dataset):
    
    def __init__(self, lr_data, hr_data, augment=True):
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.augment = augment

    def __len__(self):
        return self.lr_data.size(0)

    def __getitem__(self, idx):
        lr = self.lr_data[idx]
        hr = self.hr_data[idx]

        # Apply augmentation (manual flip) if specified
        if self.augment:
            lr, hr = random_transformation(lr, hr)
            
        return lr, hr