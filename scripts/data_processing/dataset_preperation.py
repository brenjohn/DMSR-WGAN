#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:02:21 2024

@author: brennan
"""

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import glob
import numpy as np

from swift_tools.data import read_metadata, read_particle_data
from swift_tools.fields import cut_field

from swift_tools.fields import get_displacement_field, get_velocity_field


def set_fields(
        fields, 
        particle_data_name, 
        get_field_data, 
        snapshots, 
        num_patches_per_snap, 
        patch_size, 
        padding
    ):
    """
    """
    for n, snapshot in enumerate(snapshots):
        grid_size, box_size, mass, h, a = read_metadata(snapshot)
        IDs = read_particle_data(snapshot, 'ParticleIDs')
        particle_data = read_particle_data(snapshot, particle_data_name)
        
        particle_data = particle_data.transpose()
        field_data = get_field_data(particle_data, IDs, box_size, grid_size)
        
        ni = n * num_patches_per_snap
        nf = (n + 1) * num_patches_per_snap
        fields[ni:nf, ...] = cut_field(
            field_data[None,...], patch_size, patch_size, pad=padding
        )


#%% Parameters
data_directory = '../../data/dmsr_runs/'

LR_snapshots = np.sort(glob.glob(data_directory + '*/064/snap_0002.hdf5'))
HR_snapshots = np.sort(glob.glob(data_directory + '*/128/snap_0002.hdf5'))

num_snaps = len(LR_snapshots)
num_patches_per_snap = 64

padding = 2
LR_inner_size = 16
LR_patch_size = LR_inner_size + 2 * padding
HR_patch_size = 32
patches_per_snapshot = (64 // 16)**3

output_dir = '../../data/dmsr_style_train/'
os.makedirs(output_dir, exist_ok=True)


#%% Metadata
LR_grid_size, box_size, LR_mass, h, a = read_metadata(LR_snapshots[0])
HR_grid_size, box_size, HR_mass, h, a = read_metadata(HR_snapshots[0])

meta_file = output_dir + 'metadata.npy' 
np.save(meta_file, [
    box_size,
    LR_patch_size * box_size / LR_grid_size, # LR patch length
    HR_patch_size * box_size / HR_grid_size, # LR patch length
    LR_patch_size,
    HR_patch_size,
    LR_inner_size,
    padding,
    LR_mass,
    HR_mass
])


#%% LR displacement
LR_displacment_fields = np.zeros(
    (num_snaps * num_patches_per_snap, 3, 
     LR_patch_size, LR_patch_size, LR_patch_size)
)

set_fields(
    LR_displacment_fields,
    'Coordinates',
    get_displacement_field,
    LR_snapshots,
    num_patches_per_snap,
    LR_inner_size,
    padding
)

LR_disp_file = output_dir + 'LR_disp_fields.npy'
np.save(LR_disp_file, LR_displacment_fields)
del LR_disp_file


#%% HR displacement
HR_size = HR_patch_size
HR_displacment_fields = np.zeros(
    (num_snaps * num_patches_per_snap, 3, HR_size, HR_size, HR_size)
)

set_fields(
    HR_displacment_fields,
    'Coordinates',
    get_displacement_field,
    HR_snapshots,
    num_patches_per_snap,
    HR_patch_size,
    padding = 0
)

HR_disp_file = output_dir + 'HR_disp_fields.npy'
np.save(HR_disp_file, HR_displacment_fields)
del HR_disp_file


#%% LR velocity
LR_velocity_fields = np.zeros(
    (num_snaps * num_patches_per_snap, 3, 
     LR_patch_size, LR_patch_size, LR_patch_size)
)

set_fields(
    LR_velocity_fields,
    'Velocities',
    get_velocity_field,
    LR_snapshots,
    num_patches_per_snap,
    LR_inner_size,
    padding
)

LR_vel_file = output_dir + 'LR_vel_fields.npy'
np.save(LR_vel_file, LR_velocity_fields)
del LR_vel_file


#%% HR velocity
HR_size = HR_patch_size
HR_velocity_fields = np.zeros(
    (num_snaps * num_patches_per_snap, 3, HR_size, HR_size, HR_size)
)

set_fields(
    HR_velocity_fields,
    'Velocities',
    get_velocity_field,
    HR_snapshots,
    num_patches_per_snap,
    HR_patch_size,
    padding = 0
)

HR_vel_file = output_dir + 'HR_vel_fields.npy'
np.save(HR_vel_file, HR_velocity_fields)
del HR_vel_file


#%% Scale factors
scale_factors = np.zeros((num_snaps * num_patches_per_snap, 1))

for n, snapshot in enumerate(LR_snapshots):
    grid_size, box_size, mass, h, a = read_metadata(snapshot)
    ni = n * num_patches_per_snap
    nf = (n + 1) * num_patches_per_snap
    scale_factors[ni:nf, :] = np.repeat(a, patches_per_snapshot)[:, None]

scale_factor_file = output_dir + 'scale_factors.npy'
np.save(scale_factor_file, scale_factors)