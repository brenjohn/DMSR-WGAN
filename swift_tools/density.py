#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:06:10 2024

@author: brennan

This file has a function that uses a cloud-in-cells method for computing
density fields from position data.
"""

import numpy as np


def cloud_in_cells(particles, grid_size, box_size, value=1):
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

    # Neighboring indices for each particle (including wrap-around 
    # using % grid_size)
    N = grid_size
    x, y, z = xi[:, 0], xi[:, 1], xi[:, 2]
    neighbors = [
              (x % N,       y % N,       z % N),
        ((x + 1) % N,       y % N,       z % N),
              (x % N, (y + 1) % N,       z % N),
        ((x + 1) % N, (y + 1) % N,       z % N),
              (x % N,       y % N, (z + 1) % N),
        ((x + 1) % N,       y % N, (z + 1) % N),
              (x % N, (y + 1) % N, (z + 1) % N),
        ((x + 1) % N, (y + 1) % N, (z + 1) % N),
    ]

    # Accumulate density contributions
    density_field = np.zeros((grid_size, grid_size, grid_size))
    for weight, (nx, ny, nz) in zip(weights, neighbors):
        np.add.at(density_field, (nx, ny, nz), value * weight)

    # Normalize by cell volume to get density
    density_field /= cell_size**3

    return density_field