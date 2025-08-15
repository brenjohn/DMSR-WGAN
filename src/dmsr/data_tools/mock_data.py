#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 12:49:26 2025

@author: brennan
"""

import h5py
import numpy as np

from numpy.random import randn, rand


def generate_mock_dataset(
        data_dir, 
        num_patches,
        lr_grid_size,
        hr_grid_size,
        lr_padding,
        hr_padding,
        include_velocities=True, 
        include_scales=True,
        include_spectra=False
    ):
    """Creates a mock dataset in the given data direcotry.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    generate_metadata(
        data_dir, lr_grid_size, hr_grid_size, lr_padding, hr_padding
    )
    
    patch_dir = data_dir / "patches/"
    patch_dir.mkdir(parents=True, exist_ok=True)
    
    for patch_num in range(num_patches):
        patch_file = patch_dir / f"patch_{patch_num}.h5"
        
        with h5py.File(patch_file, 'w') as file:
            mock_data = generate_mock_patch(3, lr_grid_size)
            file.create_dataset('LR_Coordinates', data = mock_data)
            
            mock_data = generate_mock_patch(3, hr_grid_size)
            file.create_dataset('HR_Coordinates', data = mock_data)
            
            if include_velocities:
                mock_data = generate_mock_patch(3, lr_grid_size)
                file.create_dataset('LR_Velocities', data = mock_data)
                
                mock_data = generate_mock_patch(3, hr_grid_size)
                file.create_dataset('HR_Velocities', data = mock_data)
            
            if include_scales:
                file.attrs['scale_factor'] = 1
                
            if include_spectra:
                mock_data = generate_mock_spectra(hr_grid_size)
                file.create_dataset('HR_Power_Spectrum', data = mock_data)
        

def generate_metadata(data_dir, lr_grid, hr_grid, lr_padding, hr_padding):
    """Creates metadata and summary stats for a generated dataset.
    """
    
    box_size, lr_patch_length, hr_patch_length = 1, 1, 1
    lr_mass, hr_mass = 1, 1
    
    meta_file = data_dir / 'metadata.npy'
    np.save(meta_file, {
        'box_size'        : box_size,
        'LR_patch_length' : lr_patch_length * box_size / lr_grid,
        'HR_patch_length' : hr_patch_length * box_size / hr_grid,
        'LR_patch_size'   : lr_grid,
        'HR_patch_size'   : hr_grid,
        'LR_inner_size'   : (lr_grid - lr_padding)//2,
        'HR_inner_size'   : (hr_grid - hr_padding)//2,
        'LR_padding'      : lr_padding,
        'HR_padding'      : hr_padding,
        'LR_mass'         : lr_mass,
        'HR_mass'         : hr_mass,
        'hubble'          : 0.7
    })
    
    np.save(data_dir / 'summary_stats.npy', {
        'LR_Coordinates_std'  : np.float64(1.0),
        'LR_Coordinates_mean' : np.float64(0.0),
        'HR_Coordinates_std'  : np.float64(1.0),
        'HR_Coordinates_mean' : np.float64(0.0),
        'LR_Velocities_std'   : np.float64(1.0),
        'LR_Velocities_mean'  : np.float64(0.0),
        'HR_Velocities_std'   : np.float64(1.0),
        'HR_Velocities_mean'  : np.float64(0.0),
    })


def generate_mock_patch(channels, grid_size):
    return randn(channels, grid_size, grid_size, grid_size)


def generate_mock_spectra(grid_size):
    return rand(grid_size//2 - 1)