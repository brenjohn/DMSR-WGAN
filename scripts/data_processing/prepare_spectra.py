#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:46:00 2025

@author: brennan
"""

import h5py
import numpy as np

from pathlib import Path
from dm_analysis import power_spectrum
from swift_tools.fields import get_positions
from dm_analysis import cloud_in_cells


dataset_dir = Path('../../data/dmsr_style_valid/').resolve()

metadata = np.load(dataset_dir / 'metadata.npy', allow_pickle=True).item()
box_size        = metadata['box_size']
LR_patch_length = metadata['LR_patch_length']
HR_patch_length = metadata['HR_patch_length']
LR_patch_size   = metadata['LR_patch_size']
HR_patch_size   = metadata['HR_patch_size']
LR_inner_size   = metadata['LR_inner_size']
LR_padding      = metadata['LR_padding']
HR_padding      = metadata['HR_padding']
LR_mass         = metadata['LR_mass']
HR_mass         = metadata['HR_mass']

grid_size = int(HR_patch_size)
box_size  = HR_patch_length


#%%
patch_dir = dataset_dir / 'patches/'
num_patches = len(list(patch_dir.iterdir()))

for patch_num in range(num_patches):
    patch_file = patch_dir / f"patch_{patch_num:05d}.h5"
    
    with h5py.File(patch_file, 'a') as file:
        patch = file['HR_Coordinates'][()]
        
        positions = get_positions(
            patch, box_size, grid_size, periodic=False
        )
        density = HR_mass * cloud_in_cells(
            positions.T, grid_size, box_size, periodic=False
        )
        k_bins, spectrum, _ = power_spectrum(density, box_size, grid_size)
        
        file.create_dataset('HR_Power_Spectrum', data = spectrum)