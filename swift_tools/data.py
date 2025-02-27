#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:54:40 2024

@author: brennan

This file has functions for reading some data stored in different file types.
"""

import torch
import h5py as h5
import numpy as np

from os.path import exists
from .fields import get_displacement_field, get_velocity_field


def read_snapshot(snapshot):
    """Reads the positions, grid_size, box_size, h and particle mass from the 
    given snapshot.
    """
    file       = h5.File(snapshot, 'r')
    h          = file['Cosmology'].attrs['h'][0]
    grid_size  = file['ICs_parameters'].attrs['Grid Resolution']
    box_size   = file['Header'].attrs['BoxSize'][0]
    dm_data    = file['DMParticles']
    IDs        = np.asarray(dm_data['ParticleIDs'])
    positions  = np.asarray(dm_data['Coordinates'])
    velocities = np.asarray(dm_data['Velocities'])
    mass       = np.asarray(dm_data['Masses'])[0]
    file.close()
    return IDs, positions, velocities, grid_size, box_size, h, mass


def read_snapshots(snapshots):
    """Returns displacement and velocity fields from the given list of swift 
    snapshots. The box_size, grid_size and particle mass are also returned.
    """
    displacement_fields = []
    velocity_fields = []
    
    for snapshot in snapshots:
        data = read_snapshot(snapshot)
        IDs, coordinates, velocities, grid_size, box_size, h, mass = data
        
        coordinates = coordinates.transpose()
        displacement_fields.append(
            get_displacement_field(coordinates, IDs, box_size, grid_size)
        )
        
        velocities = velocities.transpose()
        velocity_fields.append(
            get_velocity_field(velocities, IDs, box_size, grid_size)
        )
        
    displacement_fields = np.stack(displacement_fields)
    velocity_fields = np.stack(velocity_fields)
    
    return displacement_fields, velocity_fields, box_size, grid_size, mass


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