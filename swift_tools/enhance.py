#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:50:49 2025

@author: brennan
"""

import os
import torch
import shutil
import h5py as h5
import numpy as np

from .positions import get_displacement_field, get_positions
from dmsr.field_operations.resize import cut_field, stitch_fields


def enhance(lr_snapshot, sr_snapshot, generator, device):
    """
    Use the given generator to enhance the `lr_snapshot` and save the result in
    `sr_snapshot`
    """
    
    if os.path.exists(sr_snapshot):
        os.remove(sr_snapshot)
    shutil.copy(lr_snapshot, sr_snapshot)
    scale_factor = generator.scale_factor
      
    with h5.File(sr_snapshot, 'a') as sr_file:
        dm_data = sr_file['DMParticles']
        
        update_particle_mass(dm_data, scale_factor)
        update_particle_velocities(dm_data, scale_factor)
        update_potentials(dm_data, scale_factor)
        update_softenings(dm_data, scale_factor)
        update_particle_data(sr_file, generator, device)
        
        grid_size = sr_file['ICs_parameters'].attrs['Grid Resolution']
        sr_grid_size = scale_factor * grid_size
        sr_file['ICs_parameters'].attrs['Grid Resolution'] = sr_grid_size
    
    
def update_particle_data(file, generator, device):
    """
    Use the given generator to upscale the particle data in the given file.
    """
    dm_data   = file['DMParticles']
    grid_size = file['ICs_parameters'].attrs['Grid Resolution']
    box_size  = file['Header'].attrs['BoxSize'][0]
    ids       = np.asarray(dm_data['ParticleIDs'])
    positions = np.asarray(dm_data['Coordinates'])
    positions = positions.transpose()
    
    generator.compute_input_padding()
    cut_size = generator.inner_region
    stride = cut_size
    pad = generator.padding
    z = generator.sample_latent_space(1, device)
    
    displacements = get_displacement_field(positions, ids, box_size, grid_size)
    field_patches = cut_field(displacements[None, ...], cut_size, stride, pad)

    sr_patches = []
    for patch in field_patches:
        patch = torch.from_numpy(patch).to(torch.float)
        sr_patch = generator(patch[None, ...], z)
        sr_patch = sr_patch.detach()
        sr_patches.append(sr_patch.numpy())
    
    scale_factor = generator.scale_factor
    sr_grid_size = scale_factor * grid_size
    displacement_field = stitch_fields(sr_patches, 4)
    sr_positions = get_positions(displacement_field, box_size, sr_grid_size)
    sr_positions = sr_positions.transpose()
    sr_ids = np.arange(sr_grid_size**3)

    del dm_data['Coordinates']
    dm_data.create_dataset('Coordinates', data=sr_positions)
    del dm_data['ParticleIDs']
    dm_data.create_dataset('ParticleIDs', data=sr_ids)


def update_particle_mass(dm_data, scale_factor):
    """Reduce the particle mass appropriately based on the scale factor.
    """
    old_mass = np.asarray(dm_data['Masses'])
    new_mass = old_mass / scale_factor**3
    new_mass = np.tile(new_mass, scale_factor**3)
    del dm_data['Masses']
    dm_data.create_dataset('Masses', data=new_mass)
    

def update_particle_velocities(dm_data, scale_factor):
    """Replaces velocity data with zeros.
    """
    new_velocities = np.zeros_like(dm_data['Velocities'])
    new_velocities = np.tile(new_velocities, (scale_factor**3, 1))
    del dm_data['Velocities']
    dm_data.create_dataset('Velocities', data=new_velocities)
    
    
def update_potentials(dm_data, scale_factor):
    """Replaces potential data with zeros.
    """
    new_potentials = np.zeros_like(dm_data['Potentials'])
    new_potentials = np.tile(new_potentials, scale_factor**3)
    del dm_data['Potentials']
    dm_data.create_dataset('Potentials', data=new_potentials)


def update_softenings(dm_data, scale_factor):
    """Reduce the softening length appropriately based on the scale factor.
    """
    old_soft = np.asarray(dm_data['Softenings'])
    new_soft = old_soft / scale_factor
    new_soft = np.tile(new_soft, scale_factor**3)
    del dm_data['Softenings']
    dm_data.create_dataset('Softenings', data=new_soft)