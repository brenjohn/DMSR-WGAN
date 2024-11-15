#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:31:25 2024

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import os
import time
import torch
import shutil
import h5py as h5
import numpy as np

from dmsr.swift_processing import get_displacement_field
from dmsr.field_operations.resize import cut_field

# Check if CUDA is available and set the device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%% Load the generator model
dmsr_model_dir = './dmsr_model/'
generator = torch.load(dmsr_model_dir + 'generator.pth')

input_grid_size = generator.grid_size
scale_factor = generator.scale_factor


#%%
data_dir = './swift_snapshots/'
lr_snapshot = data_dir + '064/snap_0002.hdf5'
sr_snapshot = lr_snapshot.replace('.hdf5', '_sr.hdf5')

os.remove(sr_snapshot)
shutil.copy(lr_snapshot, sr_snapshot)
  
# with h5.File(sr_snapshot, 'a') as sr_file:
sr_file = h5.File(sr_snapshot, 'a')

dm_data = sr_file['DMParticles']


# Update particle mass
old_mass = np.asarray(dm_data['Masses'])
new_mass = old_mass / scale_factor**3
new_mass = np.tile(new_mass, scale_factor**3)

del dm_data['Masses']
dm_data.create_dataset('Masses', data=new_mass)


# Update particle velocities
new_velocities = np.zeros_like(dm_data['Velocities'])
new_velocities = np.tile(new_velocities, (scale_factor**3, 1))

del dm_data['Velocities']
dm_data.create_dataset('Velocities', data=new_velocities)


# Update potentials
new_potentials = np.zeros_like(dm_data['Potentials'])
new_potentials = np.tile(new_potentials, scale_factor**3)

del dm_data['Potentials']
dm_data.create_dataset('Potentials', data=new_potentials)


# Update softenings
# TODO: Check correctness of this update rule
old_soft = np.asarray(dm_data['Softenings'])
new_soft = old_soft / scale_factor
new_soft = np.tile(new_soft, scale_factor**3)

del dm_data['Softenings']
dm_data.create_dataset('Softenings', data=new_mass)

# TODO: some code in swift_processing could probably be reused here with some 
# refactoring.
# Update particle coordinates and IDs
grid_size = sr_file['ICs_parameters'].attrs['Grid Resolution']
box_size  = sr_file['Header'].attrs['BoxSize'][0]
ids       = np.asarray(dm_data['ParticleIDs'])
positions = np.asarray(dm_data['Coordinates'])
positions = positions.transpose()
displacements = get_displacement_field(positions, ids, box_size, grid_size)

# TODO: move the padding attribute from the dmr gan to the generator.
# TODO: refactor cut_fields to have the option to return patches that cover the
# given field. At the moment if the cut_size doesn't evenly divide into the
# size of the field then some patches near the boundary are missing.
# TODO: A function for recombining patches into a full field would be useful.
cut_size = 16
stride = 16
pad = 2
field_patches = cut_field(displacements[None, ...], cut_size, stride, pad)