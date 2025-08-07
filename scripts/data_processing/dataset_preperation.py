#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:02:21 2024

@author: brennan

This script will create a dataset for training a DMSR-WGAN from swift 
snapshots. It creates field patches from simulation snapshots and saves them as 
individual HDF5 files.

Each file contains represents a patch, and can include multiple datasets :
(LR displacement, LR velocity, HR displacement, HR velocity, scale factor)
"""

import h5py
import numpy as np

from pathlib import Path
from swift_tools.fields import cut_field
from swift_tools.data import read_metadata, read_particle_data
from swift_tools.fields import get_displacement_field, get_velocity_field


def create_fields(
        output_dir,
        prefix,
        particle_data_name, 
        get_field_data, 
        snapshots,
        patch_size, 
        padding
    ):
    """
    Creates field patches from simulation snapshots and saves them as 
    individual HDF5 files.
    """
    dataset_name = f'{prefix}_{particle_data_name}'
    patch_num = 0
    
    for snapshot in snapshots:
        grid_size, box_size, mass, h, a = read_metadata(snapshot)
        IDs = read_particle_data(snapshot, 'ParticleIDs')
        particle_data = read_particle_data(snapshot, particle_data_name)
        
        particle_data = particle_data.transpose()
        field_data = get_field_data(particle_data, IDs, box_size, grid_size)
        
        patches = cut_field(
            field_data[None,...], patch_size, patch_size, pad=padding
        )
            
        for patch in patches:
            patch_file = output_dir / f"patch_{patch_num}.h5"
            
            with h5py.File(patch_file, 'a') as file:
                file.create_dataset(dataset_name, data = patch)
                file.attrs['scale_factor'] = a

            patch_num += 1


#%% Parameters
data_dir = Path('/media/brennan/Leavitt_data/data/DM_SR/')
data_dir /= 'swift-sims/dmsr_z_runs_1Mpc/'

output_dir = Path('../../data/dmsr_style_train/').resolve()
output_dir.mkdir(parents=True, exist_ok=True)

LR_snapshots = sorted(data_dir.glob('run[1-8]/064/snap_*.hdf5'))
HR_snapshots = sorted(data_dir.glob('run[1-8]/128/snap_*.hdf5'))

num_snaps = len(LR_snapshots)

LR_padding = 2
LR_inner_size = 16
LR_patch_size = LR_inner_size + 2 * LR_padding

HR_padding = 1
HR_inner_size = 32
HR_patch_size = HR_inner_size + 2 * HR_padding

patches_per_snapshot = (64 // 16)**3


#%% Metadata
print('Creating metadata.')
LR_grid_size, box_size, LR_mass, h, a = read_metadata(LR_snapshots[0])
HR_grid_size, box_size, HR_mass, h, a = read_metadata(HR_snapshots[0])

meta_file = output_dir / 'metadata.npy'
np.save(meta_file, {
    'box_size'        : box_size,
    'LR_patch_length' : LR_patch_size * box_size / LR_grid_size,
    'HR_patch_length' : HR_patch_size * box_size / HR_grid_size,
    'LR_patch_size'   : LR_patch_size,
    'HR_patch_size'   : HR_patch_size,
    'LR_inner_size'   : LR_inner_size,
    'HR_inner_size'   : HR_inner_size,
    'LR_padding'      : LR_padding,
    'HR_padding'      : HR_padding,
    'LR_mass'         : LR_mass,
    'HR_mass'         : HR_mass,
    'hubble'          : h
})


#%% LR displacement
print('Creating LR displacement patches.')
create_fields(
    output_dir,
    'LR', 'Coordinates',
    get_displacement_field,
    LR_snapshots,
    LR_inner_size,
    LR_padding
)


#%% HR displacement
print('Creating HR displacement patches.')
create_fields(
    output_dir,
    'HR', 'Coordinates',
    get_displacement_field,
    HR_snapshots,
    HR_patch_size,
    HR_padding
)


#%% LR velocity
print('Creating LR velocity patches.')
create_fields(
    output_dir,
    'LR', 'Velocities',
    get_velocity_field,
    LR_snapshots,
    LR_inner_size,
    LR_padding
)


#%% HR velocity
print('Creating HR velocity patches.')
create_fields(
    output_dir,
    'HR', 'Velocities',
    get_velocity_field,
    HR_snapshots,
    HR_patch_size,
    HR_padding
)
