#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:46:00 2025

@author: brennan
"""

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np

from dm_analysis import power_spectrum
from swift_tools.fields import get_positions
from dm_analysis import cloud_in_cells


dataset_dir = '../../data/dmsr_style_valid/'
displacement_field_dir = dataset_dir + 'HR_disp_fields/'

spectra_dir = dataset_dir + 'HR_spectra/'
os.makedirs(spectra_dir, exist_ok=True)

metadata = np.load(dataset_dir + 'metadata.npy')
box_size        = metadata[0]
LR_patch_length = metadata[1]
HR_patch_length = metadata[2]
LR_patch_size   = metadata[3]
HR_patch_size   = metadata[4]
LR_inner_size   = metadata[5]
padding         = metadata[6]
LR_mass         = metadata[7]
HR_mass         = metadata[8]

grid_size = int(HR_patch_size)
box_size  = HR_patch_length

#%%
patches = os.listdir(displacement_field_dir)
patches.sort(key = lambda s: (len(s), s))

for patch_name in patches:
    patch_file = displacement_field_dir + patch_name
    patch = np.load(patch_file)
    
    positions = get_positions(patch, box_size, grid_size, periodic=False)
    density = cloud_in_cells(positions.T, grid_size, box_size, periodic=False)
    density *= HR_mass
    k_bins, spectrum, _ = power_spectrum(density, box_size, grid_size)
    
    np.save(spectra_dir + patch_name, spectrum)