#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:20:57 2025

@author: brennan

This file has an implementation of the cloud-in-cells method for computing
density fields from position data.
"""

import numpy as np


def cloud_in_cells(particles, grid_size, box_size, value=1, periodic=True):
    """Converts particle coordinates into a density field with shape 
    (grid_size, grid_size, grid_size) using Cloud-in-Cells (CIC) interpolation.

    Parameters:
        particles (numpy.ndarray): Array of shape (N, 3) containing the xyz 
        coordinates of particles.
        grid_size (int): Number of grid cells along each axis.
        box_size (float): The size of the simulation box.

    Returns:
        numpy.ndarray: Density field.
    """
    # Normalize particle coordinates to [0, grid_size) and get both the integer
    # and fractional parts of them (xi and dx respectively)
    cell_size = box_size / grid_size
    xs = particles / cell_size
    xi = np.floor(xs).astype(int)
    dx = xs - xi

    # Compute weights for trilinear interpolation
    weights = [
        (1 - dx[:, 0]) * (1 - dx[:, 1]) * (1 - dx[:, 2]),
             dx[:, 0]  * (1 - dx[:, 1]) * (1 - dx[:, 2]),
        (1 - dx[:, 0]) *      dx[:, 1]  * (1 - dx[:, 2]),
             dx[:, 0]  *      dx[:, 1]  * (1 - dx[:, 2]),
        (1 - dx[:, 0]) * (1 - dx[:, 1]) *      dx[:, 2],
             dx[:, 0]  * (1 - dx[:, 1]) *      dx[:, 2],
        (1 - dx[:, 0]) *      dx[:, 1]  *      dx[:, 2],
             dx[:, 0]  *      dx[:, 1]  *      dx[:, 2],
    ]

    # The offsets defineing the 8 neighbours of a cell.
    offsets = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ])

    # Accumulate density contributions
    density_field = np.zeros((grid_size, grid_size, grid_size))
    for offset, weight in zip(offsets, weights):
        neighbour = xi + offset[None, :]
        
        if periodic:
            # Wrap particles around box if they're outside of it.
            neighbour %= grid_size
            np.add.at(
                density_field, 
                (neighbour[:, 0], neighbour[:, 1], neighbour[:, 2]),
                value * weight
            )
            
        else:
            # Mask out particles contributing outside the grid
            valid = np.all((neighbour >= 0) & (neighbour < grid_size), axis=1)
            neighbour = neighbour[valid]
            np.add.at(
                density_field, 
                (neighbour[:, 0], neighbour[:, 1], neighbour[:, 2]), 
                (value * weight[valid])
            )

    # Normalize by cell volume to get density
    density_field /= cell_size**3
    return density_field