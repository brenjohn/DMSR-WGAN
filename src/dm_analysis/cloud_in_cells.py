#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:20:57 2025

@author: brennan

This file has an implementation of the cloud-in-cells method for computing
density fields from position data.
"""

import numpy as np
from numba import njit, prange


@njit(inline='always')
def add(grid, i, j, k, amount, grid_size, periodic):
    """Handles the boundary logic for a single neighbour update."""
    if periodic:
        grid[i % grid_size, j % grid_size, k % grid_size] += amount
    
    else:
        if 0 <= i < grid_size and 0 <= j < grid_size and 0 <= k < grid_size:
            grid[i, j, k] += amount



@njit(parallel=True, fastmath=True)
def cloud_in_cells(particles, grid_size, box_size, value=1.0, periodic=True):
    """Converts particle coordinates into a density field with shape 
    (grid_size, grid_size, grid_size) using Cloud-in-Cells (CIC) interpolation.

    Parameters:
        particles : Array of shape (N, 3) holding particle xyz coordinates.
        grid_size : Number of grid cells along each axis.
        box_size  : The size of the simulation box.
        value     : Real number to weight the particles by (eg. Mass).
        periodic  : Flag to turn periodic boundary conditions on or off.

    Returns:
        A numpy array containing the CIC density field.
    """
    density = np.zeros(
        (grid_size, grid_size, grid_size), dtype=np.float32
    )
    cell_size = box_size / grid_size
    inv_cell_size = 1.0 / cell_size
    num_particles = particles.shape[0]
    
    for i in prange(num_particles):
        # Get the coordinates of particle i, noramlised to [0, grid_size).
        xi = particles[i, 0] * inv_cell_size
        yi = particles[i, 1] * inv_cell_size
        zi = particles[i, 2] * inv_cell_size
        
        # Get the i, j, k indices of the cell containing the particle. 
        i0 = int(np.floor(xi)) 
        j0 = int(np.floor(yi))
        k0 = int(np.floor(zi))

        # Get the fractional part of the coordinates representing where in the
        # parent cell the particle is located.
        dx = xi - i0
        dy = yi - j0
        dz = zi - k0

        # Get indices for neighbouring cells.
        i1 = (i0 + 1) % grid_size if periodic else (i0 + 1)
        j1 = (j0 + 1) % grid_size if periodic else (j0 + 1)
        k1 = (k0 + 1) % grid_size if periodic else (k0 + 1)

        # Compute weights and accumulate density using trilinear interpolation.
        weight0 = value * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        weight1 = value *        dx  * (1.0 - dy) * (1.0 - dz)
        weight2 = value * (1.0 - dx) *        dy  * (1.0 - dz)
        weight3 = value *        dx  *        dy  * (1.0 - dz)
        weight4 = value * (1.0 - dx) * (1.0 - dy) *        dz
        weight5 = value *        dx  * (1.0 - dy) *        dz
        weight6 = value * (1.0 - dx) *        dy  *        dz
        weight7 = value *        dx  *        dy  *        dz
        
        add(density, i0, j0, k0, weight0, grid_size, periodic)
        add(density, i1, j0, k0, weight1, grid_size, periodic)
        add(density, i0, j1, k0, weight2, grid_size, periodic)
        add(density, i1, j1, k0, weight3, grid_size, periodic)
        add(density, i0, j0, k1, weight4, grid_size, periodic)
        add(density, i1, j0, k1, weight5, grid_size, periodic)
        add(density, i0, j1, k1, weight6, grid_size, periodic)
        add(density, i1, j1, k1, weight7, grid_size, periodic)

    # Normalize by cell volume.
    v_inv = 1.0 / (cell_size**3)
    return density * v_inv