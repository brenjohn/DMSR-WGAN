#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:54:40 2024

@author: brennan

This file has functions for reading some data stored in different file types.
"""

import h5py as h5
import numpy as np

from .fields import get_displacement_field, get_velocity_field


def read_snapshot(snapshot):
    """Reads the positions, grid_size, box_size, h, particle mass and scale 
    factor from the given snapshot.
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
    a          = file['Cosmology'].attrs['Scale-factor']
    file.close()
    return IDs, positions, velocities, grid_size, box_size, h, mass, a


def read_snapshots(snapshots):
    """Returns displacement and velocity fields and scale factors from the 
    given list of swift snapshots. The box_size, grid_size and particle mass 
    are also returned.
    """
    displacement_fields = []
    velocity_fields     = []
    scale_factors       = []
    
    for snapshot in snapshots:
        data = read_snapshot(snapshot)
        IDs, coordinates, velocities, grid_size, box_size, h, mass, a = data
        
        coordinates = coordinates.transpose()
        displacement_fields.append(
            get_displacement_field(coordinates, IDs, box_size, grid_size)
        )
        
        velocities = velocities.transpose()
        velocity_fields.append(
            get_velocity_field(velocities, IDs, box_size, grid_size)
        )
        
        scale_factors.append(a)
        
    displacement_fields = np.stack(displacement_fields)
    velocity_fields     = np.stack(velocity_fields)
    scale_factors       = np.concatenate(scale_factors)
    
    return (
        displacement_fields, 
        velocity_fields, 
        scale_factors, 
        box_size, 
        grid_size, 
        mass
    )