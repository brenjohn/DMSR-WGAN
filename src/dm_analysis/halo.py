#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:34:35 2025

@author: brennan
"""

import numpy as np


class Halo:
    """A class to represent dark matter halos.
    """
    
    def __init__(self, halo_id, particle_ids):
        self.halo_id = halo_id
        self.particle_ids = np.asarray(particle_ids)
        self.num_particles = len(particle_ids)
        
        self.positions = None
        self.mass = None
        self.redshift = None
        
        self.descendant = None
        self.ancestors = None
    
    
    def set_particle_positions(self, positions):
        self.positions = positions[self.particle_ids]
    
    
    def set_mass(self, particle_mass):
        self.mass = particle_mass * self.num_particles
        
        
    def set_redshift(self, z):
        self.redshift = z
    
    
    def set_descendant(self, descendant_id):
        self.descendant = descendant_id
    
    
    def set_ancestor(self, ancestor_id):
        if self.ancestors is not None:
            self.ancestors.append(ancestor_id)
        else:
            self.ancestors = [ancestor_id]
    
    
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