#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:54:40 2024

@author: brennan
"""

import torch
import h5py as h5
import numpy as np

def get_displacement_field(positions, ids, box_size, grid_size):
    """Creates a displacement field from the given particle positions and
    particle IDs.
    
    Displacements of particles are relative to associated points on a regular
    grid. Particles are associated with grid points by the following relation
    between particle IDs and grid indices:
        
        ID = iz + grid_size * (iy + grid_size * ix)
    """
    # Use the particle IDs to compute particle grid indices (ix, iy, iz).
    ix = ids // (grid_size * grid_size)
    iy = (ids % (grid_size * grid_size)) // grid_size
    iz = ids % grid_size
    
    # Create an array containing the postions of grid points.
    cell_size = box_size / grid_size
    points = np.arange(cell_size/2, box_size, cell_size)
    grid_points = np.stack((points[ix], points[iy], points[iz]))
    
    # Compute the displacement of each particle from its associated grid point.
    # Periodic boundary conditions are taken into account by considering the
    # shortest distance between a particle and its grid point.
    d = positions - grid_points     # displacement from grid to particle.
    c = d - np.sign(d) * box_size   # complement displacement through boundary.
    displacements = np.where(np.abs(d) < np.abs(c), d, c)
    
    # Arrange displacements into a field and return it.
    displacement_field = np.zeros((3, grid_size, grid_size, grid_size))
    displacement_field[:, ix, iy, iz] = displacements
    return displacement_field


def get_positions(displacement_field, box_size, grid_size, periodic=True):
    """Creates an array containing the absolute coordinates of particles from 
    the given displacement field.
    """
    points = np.arange(0, box_size, box_size/grid_size)
    grid = np.stack(np.meshgrid(points, points, points, indexing='ij'))
    
    positions = (grid + displacement_field)
    positions = positions.reshape(3, -1)
    
    if periodic:
        positions %= box_size
        
    return positions


# TODO: in the interest of reuability, this should read a single snapshot.
def read_snapshot(snapshots):
    """Returns displacement fields from the given list of snapshots.
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
    """
    """
    LR_data = np.load(data_directory + 'LR_fields.npy')
    LR_data = torch.from_numpy(LR_data)
    
    HR_data = np.load(data_directory + 'HR_fields.npy')
    HR_data = torch.from_numpy(HR_data)
    
    meta_file = data_directory + 'metadata.npy'
    meta_data = np.load(meta_file)
    box_size, HR_patch_size, LR_size, HR_size, LR_mass, HR_mass = meta_data
    
    return LR_data, HR_data, HR_patch_size, LR_size, HR_size