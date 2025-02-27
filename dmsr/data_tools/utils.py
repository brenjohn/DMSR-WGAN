#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:56:32 2025

@author: brennan
"""

import torch
import numpy as np

from os.path import exists


def load_numpy_dataset(data_directory):
    """Returns LR and HR data contained in numpy files saved in the given 
    directory.
    """
    LR_data = np.load(data_directory + 'LR_fields.npy')
    LR_data = torch.from_numpy(LR_data)
    
    HR_data = np.load(data_directory + 'HR_fields.npy')
    HR_data = torch.from_numpy(HR_data)
    
    meta_file = data_directory + 'metadata.npy'
    meta_data = np.load(meta_file)
    box_size, HR_patch_size, LR_size, HR_size, LR_mass, HR_mass = meta_data
    
    return LR_data, HR_data, HR_patch_size, LR_size, HR_size


def load_normalisation_parameters(param_file):
    """Reads the standard deviations from the given .npy file used to noramlise
    dmsr training data.
    """
    lr_pos_std = hr_pos_std = lr_vel_std = hr_vel_std = 1
    
    if exists(param_file):
        scale_params = np.load(param_file, allow_pickle=True).item()
        scale_params = {k : v.item() for k, v in scale_params.items()}
        lr_pos_std = scale_params.get('lr_position_std', 1)
        hr_pos_std = scale_params.get('hr_position_std', 1)
        lr_vel_std = scale_params.get('lr_velocity_std', 1)
        hr_vel_std = scale_params.get('hr_velocity_std', 1)
    
    return lr_pos_std, hr_pos_std, lr_vel_std, hr_vel_std


def generate_mock_data(lr_grid_size, hr_grid_size, channels, samples):
    """Create a mock training data set for testing.
    """
    box_size = 1
    shape = (samples, channels, lr_grid_size, lr_grid_size, lr_grid_size)
    LR_data = torch.rand(*shape)
    shape = (samples, channels, hr_grid_size, hr_grid_size, hr_grid_size)
    HR_data = torch.rand(*shape)
    return LR_data, HR_data, box_size, lr_grid_size, hr_grid_size