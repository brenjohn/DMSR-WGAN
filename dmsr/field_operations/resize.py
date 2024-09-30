#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:41:51 2024

@author: brennan
"""

import numpy as np

def crop(field, crop_size):
    """Crops the given tensor by size crop_size.
    
    The tensor `field` should have shape (batch_size, channels, Nx, Ny, Nz).
    """
    ind = (slice(None),) * 2 + (slice(crop_size, -crop_size),) * 3
    return field[ind]


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
    cuts = []
    if not stride:
        stride = cut_size
    
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