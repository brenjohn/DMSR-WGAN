#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:50:35 2025

@author: john

This file defines functions that implement the friends of friends algorithm for
finding dark-matter halos. 
"""

import numpy as np
from scipy.spatial import cKDTree


def create_pbc_ghosts(positions, box_size, linking_length):
    """Create ghost particles for periodic boundary conditions.
    """
    offsets = [-1, 0, 1]  # Represents shifts of -1, 0, +1 box size
    shifts = np.array([
        (dx, dy, dz) for dx in offsets for dy in offsets for dz in offsets 
        if (dx, dy, dz) != (0, 0, 0)
    ])
    ghost_positions = []
    for shift in shifts:
        ghost_positions.append(positions + shift * box_size)
    
    return np.vstack([positions] + ghost_positions)


def friend_of_friends(positions, box_size, linking_length):
    """
    Use the friends of friends algorithm to build and return a list of halos
    in the given particle position data. Periodic boundary conditions are 
    assumed.
    """
    extended_positions = create_pbc_ghosts(positions, box_size, linking_length)
    tree = cKDTree(extended_positions)
    
    n_particles = len(positions)
    visited = np.zeros(len(extended_positions), dtype=bool)
    halos = []
    
    for i in range(n_particles):
        if not visited[i]:
            # Start a new halo
            halo = []
            stack = [i]
            
            while stack:
                idx = stack.pop()
                if not visited[idx]:
                    visited[idx] = True
                    halo.append(idx)
                    # Find neighbors
                    neighbors = tree.query_ball_point(
                        extended_positions[idx], linking_length
                    )
                    stack.extend(neighbors)
            halos.append(halo)
    
    return halos