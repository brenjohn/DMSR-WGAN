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


def create_fields(
        field_dirname, 
        particle_data_name, 
        get_field_data, 
        snapshots,
        patch_size, 
        padding
    ):
    """
    """
    os.makedirs(field_dirname, exist_ok=True)
    patch_filename = field_dirname + 'patch_{}.npy'
    patch_num = 0
    for n, snapshot in enumerate(snapshots):
        grid_size, box_size, mass, h, a = read_metadata(snapshot)
        IDs = read_particle_data(snapshot, 'ParticleIDs')
        particle_data = read_particle_data(snapshot, particle_data_name)
        
        particle_data = particle_data.transpose()
        field_data = get_field_data(
            particle_data, IDs, box_size, grid_size
        )
        
        patches = cut_field(
            field_data[None,...], patch_size, patch_size, pad=padding
        )
        for patch in patches:
            np.save(patch_filename.format(patch_num), patch)
            patch_num += 1


#%% Parameters
# data_directory = '../../data/dmsr_runs/'
data_directory = '/media/brennan/Leavitt_data/data/DM_SR/swift-sims/dmsr_z_runs_1Mpc/'

LR_snapshots = np.sort(glob.glob(data_directory + 'run9/064/snap_*.hdf5'))
HR_snapshots = np.sort(glob.glob(data_directory + 'run9/128/snap_*.hdf5'))

num_snaps = len(LR_snapshots)

padding = 2
LR_inner_size = 16
LR_patch_size = LR_inner_size + 2 * padding
HR_patch_size = 32
patches_per_snapshot = (64 // 16)**3

output_dir = '../../data/dmsr_style_valid/'
os.makedirs(output_dir, exist_ok=True)


#%% Metadata
print('Creating metadata.')
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
print('Creating LR displacement patches.')
LR_disp_dir = output_dir + 'LR_disp_fields/'
create_fields(
    LR_disp_dir,
    'Coordinates',
    get_displacement_field,
    LR_snapshots,
    LR_inner_size,
    padding
)


#%% HR displacement
print('Creating HR displacement patches.')
HR_disp_dir = output_dir + 'HR_disp_fields/'
create_fields(
    HR_disp_dir,
    'Coordinates',
    get_displacement_field,
    HR_snapshots,
    HR_patch_size,
    padding = 0
)


#%% LR velocity
print('Creating LR velocity patches.')
LR_vel_dir = output_dir + 'LR_vel_fields/'
create_fields(
    LR_vel_dir,
    'Velocities',
    get_velocity_field,
    LR_snapshots,
    LR_inner_size,
    padding
)


#%% HR velocity
print('Creating HR velocity patches.')
HR_vel_dir = output_dir + 'HR_vel_fields/'
create_fields(
    HR_vel_dir,
    'Velocities',
    get_velocity_field,
    HR_snapshots,
    HR_patch_size,
    padding = 0
)


#%% Scale factors
print('Creating scale factors array.')
num_patches_per_snap = 64
scale_factors = np.zeros((num_snaps * num_patches_per_snap, 1))

for n, snapshot in enumerate(LR_snapshots):
    grid_size, box_size, mass, h, a = read_metadata(snapshot)
    ni = n * num_patches_per_snap
    nf = (n + 1) * num_patches_per_snap
    scale_factors[ni:nf, :] = np.repeat(a, patches_per_snapshot)[:, None]

scale_factor_file = output_dir + 'scale_factors.npy'
np.save(scale_factor_file, scale_factors)