#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:02:21 2024

@author: brennan
"""

import sys
import glob
sys.path.append("..")
sys.path.append("../..")

import h5py as h5
import numpy as np

from dmsr.swift_processing import get_displacement_field
from dmsr.swift_processing import read_snapshot
from dmsr.field_operations.resize import cut_field


#%%
data_directory = '../../data/dmsr_runs/'

LR_snapshots = np.sort(glob.glob(data_directory + '*/064/snap_0002.hdf5'))
HR_snapshots = np.sort(glob.glob(data_directory + '*/128/snap_0002.hdf5'))


#%%
LR_fields, box_size, LR_grid_size, LR_mass = read_snapshot(LR_snapshots)
HR_fields, box_size, HR_grid_size, HR_mass = read_snapshot(HR_snapshots)

# # Normalise values so that box size is 1
# LR_fields /= box_size
# HR_fields /= box_size
# box_size = 1


#%%
padding = 2
LR_patch_size = 16
HR_patch_size = 32

LR_fields = cut_field(LR_fields, LR_patch_size, LR_patch_size, pad=padding)
HR_fields = cut_field(HR_fields, HR_patch_size, HR_patch_size)


#%%
LR_file = '../../data/dmsr_training/LR_fields.npy'
np.save(LR_file, LR_fields)

HR_file = '../../data/dmsr_training/HR_fields.npy'
np.save(HR_file, HR_fields)

meta_file = '../../data/dmsr_training/metadata.npy'
LR_size = LR_patch_size + 2 * padding
HR_size = HR_patch_size
np.save(meta_file, [
    box_size,
    # LR_size * box_size / LR_grid_size, # TODO: Add LR patch size to metadata.
    HR_size * box_size / HR_grid_size,
    LR_size,
    HR_size,
    LR_mass,
    HR_mass
])
