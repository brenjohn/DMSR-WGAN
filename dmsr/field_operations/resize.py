#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:41:51 2024

@author: brennan

This file defines functions for resizing tensors.
"""


def crop(field, crop_size):
    """Crops the spatial dimensions of the given tensor by size crop_size.
    
    The tensor `field` should have shape (batch_size, channels, Nx, Ny, Nz).
    """
    ind = (slice(None),) * 2 + (slice(crop_size, -crop_size),) * 3
    return field[ind]


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