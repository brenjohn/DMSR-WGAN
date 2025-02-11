#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:41:51 2024

@author: brennan

This file defines functions for resizing tensors using various methods.
"""

import numpy as np


def crop(field, crop_size):
    """Crops the spatial dimensions of the given tensor by size crop_size.
    
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


def pixel_unshuffle(tensor, scale):
    """
    Reshapes the given a tensor of shape (B, C, D, H, W) to shape 
    (B, C * scale**3, D // scale, H // scale, W // scale).
    
    The reshaping procedure uses the pixel shuffle method of Shi et al 2016 -
    "Real-Time Single Image and Video Super-Resolution Using an Efficient 
    Sub-Pixel Convolutional Neural Network"
    """
    # Ensure tensor has the right shape
    batch_size, channels, depth, height, width = tensor.shape

    new_channels = channels * scale**3
    new_depth    = depth  // scale
    new_height   = height // scale
    new_width    = width  // scale

    # Reshape and permute to rearrange data
    tensor = tensor.contiguous().view(
        batch_size, channels,
        new_depth, scale,
        new_height, scale,
        new_width, scale
    )
    tensor = tensor.permute(0, 1, 3, 5, 7, 2, 4, 6)
    tensor = tensor.contiguous().view(
        batch_size, new_channels, new_depth, new_height, new_width
    )
    
    return tensor