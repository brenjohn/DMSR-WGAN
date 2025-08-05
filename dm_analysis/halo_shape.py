#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 23:31:43 2025

@author: john

This script was used to create figure 7 of Brennan et. al. 2025 "On the Use of
WGANs for Super Resolution in Dark-Matter Simulations".
"""

import numpy as np

from numpy import linalg as la


def compute_shape_tensor(positions):
    """Computes the shape tensor from the given particle positions.
    """
    S = np.zeros((3, 3))
    for r in positions:
        S += np.outer(r, r)
    
    num_particles = positions.shape[0]
    return S / num_particles


def compute_shape(halo, max_iterations=10):
    """Computes the shape of the give halo.
    
    Two ratios b/a and c/a are returned where a, b and c (a > b > c) are the 
    principle lengths of an ellipsoid that has been fit to the halo.
    
    The shape is computed using method E1 from Zemp et al (2011) "On
    determining the shape of matter distributions".
    """
    # Initialize teh principal semi-axes of the ellipsoid to be those of a 
    # sphere enclosing all particles.
    initial_radius = np.max(la.norm(halo.positions, axis=1))
    initial_principle_lengths = np.asarray([initial_radius] * 3)
    principle_lengths = initial_principle_lengths
    principal_axes = np.eye(3)
    
    for i in range(max_iterations):
        # Find particles inside current ellipsoid
        positions = np.matmul(halo.positions, principal_axes)
        scaled_elliptic_radii = la.norm(positions / principle_lengths, axis=1)
        inside_ellipsoid = scaled_elliptic_radii <= 1
        positions = positions[inside_ellipsoid, :]
        
        # If no particles are inside the volume, return none to signal the
        # method failed.
        if not np.any(inside_ellipsoid):
            print('Computing shape failed')
            return None
        
        # Compute the shape tensor for the particles inside the current 
        # ellipsoid
        shape_tensor = compute_shape_tensor(positions)
        
        # Compute principle axes and lengths of the new shape tensor
        eig_vals, principal_axes = la.eigh(shape_tensor)
        principle_ratios = np.sqrt(np.abs(eig_vals))
        principle_ratios /= principle_ratios[-1]
        principle_lengths = initial_principle_lengths * principle_ratios
    
    b_over_a = principle_lengths[1] / principle_lengths[2]
    c_over_a = principle_lengths[0] / principle_lengths[2]
    return b_over_a, c_over_a