#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 12:49:26 2025

@author: brennan
"""

import os
import torch
import numpy as np

from numpy.random import randn, rand


def generate_mock_dataset(
        data_dir, 
        num_patches,
        lr_grid_size,
        hr_grid_size,
        lr_padding,
        include_velocities=True, 
        include_scales=True,
        include_spectra=False
    ):
    """Creates a mock dataset in the given data direcotry.
    """
    generate_metadata(data_dir, lr_grid_size, hr_grid_size, lr_padding)
    
    lr_disp_dir = data_dir + 'LR_disp_fields/'
    hr_disp_dir = data_dir + 'HR_disp_fields/'
    generate_mock_patches(lr_disp_dir, num_patches, 3, lr_grid_size)
    generate_mock_patches(hr_disp_dir, num_patches, 3, hr_grid_size)
    
    if include_velocities:
        lr_vel_dir = data_dir + 'LR_vel_fields/'
        hr_vel_dir = data_dir + 'HR_vel_fields/'
        generate_mock_patches(lr_vel_dir, num_patches, 3, lr_grid_size)
        generate_mock_patches(hr_vel_dir, num_patches, 3, hr_grid_size)
        
    if include_scales:
        np.save(data_dir + 'scale_factors.npy', rand(num_patches, 1))
        
    if include_spectra:
        hr_spectra_dir = data_dir + 'HR_spectra/'
        generate_mock_spectra(hr_spectra_dir, num_patches, hr_grid_size)
        

def generate_metadata(data_dir, lr_grid, hr_grid, padding):
    """Creates metadata and summary stats for a generated dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    box_size, LR_patch_length, HR_patch_length = 1, 1, 1
    LR_inner_size = (lr_grid - padding)//2
    LR_mass, HR_mass = 1, 1
    metadata = np.asarray([
        box_size, LR_patch_length, HR_patch_length, 
        lr_grid, hr_grid, LR_inner_size, padding, 
        LR_mass, HR_mass
    ])
    np.save(data_dir + 'metadata.npy', metadata)
    
    stats = {
        'LR_disp_fields_std'  : 1,
        'LR_disp_fields_mean' : 0,
        'HR_disp_fields_std'  : 1,
        'HR_disp_fields_mean' : 0,
        'LR_vel_fields_std'   : 1,
        'LR_vel_fields_mean'  : 0,
        'HR_vel_fields_std'   : 1,
        'HR_vel_fields_mean'  : 0,
    }
    np.save(data_dir + 'summary_stats.npy', stats)

 
def generate_mock_patches(patch_dir, num_patches, channels, grid_size):
    os.makedirs(patch_dir, exist_ok=True)
    
    for i in range(num_patches):
        patch = generate_mock_patch(channels, grid_size)
        np.save(patch_dir + f'patch_{i}.npy', patch)


def generate_mock_patch(channels, grid_size):
    return randn(channels, grid_size, grid_size, grid_size)


def generate_mock_spectra(data_dir, num_patches, grid_size):
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_patches):
        spectra = rand(grid_size//2 - 1)
        np.save(data_dir + f'patch_{i}.npy', spectra)