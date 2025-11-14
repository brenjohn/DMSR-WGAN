#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:17:12 2025

@author: brennan
"""

import torch

from .cloud_in_cells import displacements_to_positions


def nn_distance_field(displacements, box_size):
    """Returns a tensor with 6 channels, each containing the distance of each
    particle from one of its six neighbours.
    """
    positions = displacements_to_positions(displacements, box_size)
    
    dx = torch.diff(positions, axis=2)
    dx = torch.norm(dx, dim=1, keepdim=True)
    dx = dx[..., :, 1:-1, 1:-1]
    dx = torch.concat((dx[:, :, :-1, :, :], dx[:, :, 1:, :, :]), dim=1)
    
    dy = torch.diff(positions, axis=3)
    dy = torch.norm(dy, dim=1, keepdim=True)
    dy = dy[..., 1:-1, :, 1:-1]
    dy = torch.concat((dy[:, :, :, :-1, :], dy[:, :, :, 1:, :]), dim=1)
    
    dz = torch.diff(positions, axis=4)
    dz = torch.norm(dz, dim=1, keepdim=True)
    dz = dz[..., 1:-1, 1:-1, :]
    dz = torch.concat((dz[:, :, :, :, :-1], dz[:, :, :, :, 1:]), dim=1)
    
    return torch.concat((dx, dy, dz), dim=1)