#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:01:39 2024

@author: brennan
"""

import torch

from torch import arange


def displacements_to_positions(displacements, box_length):
    device = displacements.device
    grid_size = displacements.shape[-1]
    cell_size = box_length / grid_size
    
    # Add the grid positions to the given relative positions to get the
    # absolute positions of particles.
    r = torch.arange(cell_size/2, box_length, cell_size, device=device)
    X, Y, Z = torch.meshgrid(r, r, r, indexing='ij')
    grid_positions = torch.stack((X, Y, Z), axis=0)
    positions = displacements + grid_positions
    
    return positions


def cic_density_field(displacements, box_size, grid_size=None):
    displacements = displacements - displacements.mean((2, 3, 4), keepdims=True)
    
    batch_size = displacements.shape[0]
    grid_size = displacements.shape[-1] if grid_size is None else grid_size
    device = displacements.device
    cell_size = box_size / grid_size
    
    positions = displacements_to_positions(displacements, box_size)
    positions = positions / cell_size
    positions = torch.permute(positions, [0, 2, 3, 4, 1])
    
    # Compute the indices for the grid cells associated with each particle.
    # Note, grid indices here can be outside the boundary (0, grid_size).
    # these indices correspond to "ghost zones" that pad the main grid.
    grid_positions = torch.floor(positions)
    
    # Compute the weights (1 - dx) * (1 - dy) * (1 - dz)
    positions = torch.unsqueeze(positions, dim=-2)
    grid_positions = torch.unsqueeze(grid_positions, dim=-2)
    neighbours = arange(8, device=device)[:, None]
    neighbours = (neighbours >> arange(2, -1, -1, device=device)) & 1
    neighbour_positions = grid_positions + neighbours
    weights = 1 - torch.abs(positions - neighbour_positions)
    weights = torch.prod(weights, -1)
    weights = weights.reshape(batch_size, -1)    
    
    # Compute indices
    indices = grid_positions.int() + neighbours
    indices = indices.reshape(batch_size, -1, 3)
    
    # Compute the density
    density_shape = (batch_size, grid_size, grid_size, grid_size)
    density = torch.zeros(*density_shape, device=device)
    for n in range(batch_size):       
        index_x = indices[n, :, 0]
        index_y = indices[n, :, 1]
        index_z = indices[n, :, 2]
        index = (index_x * grid_size + index_y) * grid_size + index_z
        source = weights[n]
        
        mask = ((indices[n] >= 0) & (indices[n] < grid_size)).all(1)
        index = index[mask]
        source = source[mask]
        
        density[n].view(-1).index_add_(0, index, source.float())
    
    # density = density / grid_size**3
    return density[:, None, ...]