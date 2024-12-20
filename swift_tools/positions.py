#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:57:50 2024

@author: brennan

This file has functions for changing particle position data between displacment
and absolute position representations.
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