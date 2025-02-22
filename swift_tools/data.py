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

from .positions import get_displacement_field


def read_snapshot(snapshot):
    """Reads the positions, grid_size, box_size, h and particle mass from the 
    given snapshot.
    """
    file      = h5.File(snapshot, 'r')
    h         = file['Cosmology'].attrs['h'][0]
    grid_size = file['ICs_parameters'].attrs['Grid Resolution']
    box_size  = file['Header'].attrs['BoxSize'][0]
    dm_data   = file['DMParticles']
    positions = np.asarray(dm_data['Coordinates'])
    mass      = np.asarray(dm_data['Masses'])[0]
    file.close()
    return positions, grid_size, box_size, h, mass


def read_snapshots(snapshots):
    """Returns displacement fields from the given list of swift napshots.
    """
    displacement_fields = []
    
    for snap in snapshots:
        data = h5.File(snap, 'r')
        grid_size   = data['ICs_parameters'].attrs['Grid Resolution']
        box_size    = data['Header'].attrs['BoxSize'][0]
        mass        = data['DMParticles']['Masses'][0]
        IDs         = np.asarray(data['DMParticles']['ParticleIDs'])
        coordinates = np.asarray(data['DMParticles']['Coordinates'])
        coordinates = coordinates.transpose()
        displacement_fields.append(
            get_displacement_field(coordinates, IDs, box_size, grid_size)
        )
        data.close()
        
    return np.stack(displacement_fields), box_size, grid_size, mass


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