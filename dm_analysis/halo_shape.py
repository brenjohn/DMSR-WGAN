#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:22:21 2025

@author: brennan
"""

import numpy as np

from numpy import linalg as la


class Halo:
    """A class to represent dark matter halos.
    """
    
    def __init__(self, particle_ids, positions, mass):
        self.particle_ids = particle_ids
        self.positions = positions
        self.num_particles = len(particle_ids)
        self.mass = mass * self.num_particles
        self.center()
        
    @classmethod
    def from_halo_dict(cls, halo_dict, mass):
        particle_ids = list(halo_dict.keys())
        positions = np.stack(list(halo_dict.values()))
        return cls(particle_ids, positions, mass)
    
    def center(self):
        """Removes the mean position from all particle positions. 
        """
        self.positions -= self.positions.mean(axis=0)
        
    def move_to_box_centre(self, box_size, particle_id=0):
        """Moves the given particle to the centre of the box. All other
        particles are moved with it. This can be useful when a halo sits on the
        boundary of a box with periodic bounary conditions so that members of
        the halo appear on opposite sides of the box.
        """
        centre = np.asarray([box_size, box_size, box_size]) / 2
        displacement = self.positions[particle_id] - centre
        self.positions -= displacement


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