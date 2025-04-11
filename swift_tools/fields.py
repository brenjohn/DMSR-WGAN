#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:57:50 2024

@author: brennan

This file defines functions for manipulating numpy tensors representing field 
data.
"""

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


def get_velocity_field(velocities, ids, box_size, grid_size):
    """Creates a velocity field from the given particle velocities and
    particle IDs.
    """
    # Use the particle IDs to compute particle grid indices (ix, iy, iz).
    ix = ids // (grid_size * grid_size)
    iy = (ids % (grid_size * grid_size)) // grid_size
    iz = ids % grid_size
    
    # Arrange displacements into a field and return it.
    velocity_field = np.zeros((3, grid_size, grid_size, grid_size))
    velocity_field[:, ix, iy, iz] = velocities
    return velocity_field


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


def get_particle_potential_field(potentials, ids, grid_size):
    """Creates a particle potential field from the given particle potentials 
    and particle IDs.
    """
    # Use the particle IDs to compute particle grid indices (ix, iy, iz).
    ix = ids // (grid_size * grid_size)
    iy = (ids % (grid_size * grid_size)) // grid_size
    iz = ids % grid_size
    
    # Arrange displacements into a field and return it.
    potential_field = np.zeros((1, grid_size, grid_size, grid_size))
    potential_field[:, ix, iy, iz] = potentials
    return potential_field


def cut_field(fields, cut_size, stride=0, pad=0):
    """Cuts the given field tensor into blocks of size `cut_size`.
    
    Arguments:
        - fields   : A numpy tensor of shape (batch_size, channels, N, N, N)
                     where N is the grid size of the fields.
        - cut_size : The base size of the blocks to cut the given fields into.
        
        - stride   : The number of cells to move in each direction before 
                     extracting the next block.
        - pad      : The number of cells to pad the base blocks on each side.
        
    Returns:
        A numpy tensor containing the blocks/subfields cut from the given
        fields tensor. The shape of the returned tensor is:
                 (number_of_cuts * batch_size, channels, n, n, n),
        where number_of_cuts is the number of subfields extracted from each 
        field and n is the grid size of each subfield (ie cut_size + 2 * pad).
    """
    grid_size = fields.shape[-1]
    if not stride:
        stride = cut_size
    
    cuts = []
    for i in range(0, grid_size, stride):
        for j in range(0, grid_size, stride):
            for k in range(0, grid_size, stride):
                
                slice_x = [n % grid_size for n in range(i-pad, i+cut_size+pad)]
                slice_y = [n % grid_size for n in range(j-pad, j+cut_size+pad)]
                slice_z = [n % grid_size for n in range(k-pad, k+cut_size+pad)]
                
                patch = np.take(fields, slice_x, axis=2)
                patch = np.take(patch, slice_y, axis=3)
                patch = np.take(patch, slice_z, axis=4)
                
                cuts.append(patch)
    
    return np.concatenate(cuts)


def stitch_fields(patches, patches_per_dim):
    """Combines or stitches the given collection of patches into a single
    tensor.
    
    This function can be thought of as performing the reverse operation
    performed by `cut_field`.
    """
    
    patch_size = patches[0].shape[-1]
    channels = patches[0].shape[-4]
    field_size = patch_size * patches_per_dim
    field = np.zeros((channels, field_size, field_size, field_size))
    
    for n, patch in enumerate(patches):
        i = n // patches_per_dim**2
        j = (n % patches_per_dim**2) // patches_per_dim
        k = n % patches_per_dim
        
        N = patch_size
        field[:, i*N:(i+1)*N, j*N:(j+1)*N, k*N:(k+1)*N] = patch
        
    return field