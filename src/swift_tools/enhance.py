#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:50:49 2025

@author: brennan

This file defines functions for enhancing the dark matter data in a swift 
snapshot using a dmsr generator model.
"""

import os
import torch
import shutil
import h5py as h5
import numpy as np

from .fields import get_positions
from .fields import get_displacement_field, get_velocity_field
from .fields import cut_field, stitch_fields
from dmsr.data_tools import crop

from torch.utils.data import DataLoader, TensorDataset


def enhance(lr_snapshot, sr_snapshot, generator, z, scale_params, device):
    """
    Use the given generator to enhance the `lr_snapshot` and save the result in
    `sr_snapshot`
    """
    
    if os.path.exists(sr_snapshot):
        os.remove(sr_snapshot)
    shutil.copy(lr_snapshot, sr_snapshot)
    scale_factor = generator.scale_factor
      
    with h5.File(sr_snapshot, 'a') as sr_file:
        cosmic_scale_factor = np.asarray(
            sr_file['Cosmology'].attrs['Scale-factor']
        )[None, ...]
        cosmic_scale_factor = torch.from_numpy(cosmic_scale_factor).float()
        if generator.style_size is None:
            cosmic_scale_factor = None
        
        dm_data = sr_file['DMParticles']
        
        update_particle_mass(dm_data, scale_factor)
        update_potentials(dm_data, scale_factor)
        update_softenings(dm_data, scale_factor)
        update_particle_data(
            sr_file, 
            generator, 
            z, 
            scale_params, 
            cosmic_scale_factor, 
            device
        )
        
        grid_size = sr_file['ICs_parameters'].attrs['Grid Resolution']
        sr_grid_size = scale_factor * grid_size
        sr_file['ICs_parameters'].attrs['Grid Resolution'] = sr_grid_size



def update_particle_data(
        file, 
        generator,
        z,
        scale_params,
        cosmic_scale_factor,
        device
    ):
    """
    Use the given generator to upscale the particle data in the given file.
    """
    cut_size           = generator.inner_region
    pad                = generator.padding
    scale_factor       = generator.scale_factor
    upscale_velocities = generator.input_channels == 6 # TODO: no magic numbers
    
    fields, grid_size, box_size = get_field_data(
        file, scale_params, upscale_velocities, scale_factor
    )
    
    fields = cut_field(fields[None, ...], cut_size, cut_size, pad)
    
    sr_patches = upscale_fields(
        fields, cosmic_scale_factor, z, generator, device
    )
    del fields
    
    sr_grid_size = scale_factor * grid_size
    write_enhanced_data(
        file, sr_patches, sr_grid_size, upscale_velocities, scale_params
    )
        


def get_field_data(file, scale_params, upscale_velocities, scale_factor):
    lr_position_std = scale_params.get('LR_disp_fields_std', 1)
    
    dm_data   = file['DMParticles']
    grid_size = file['ICs_parameters'].attrs['Grid Resolution']
    box_size  = file['Header'].attrs['BoxSize'][0]
    ids       = np.asarray(dm_data['ParticleIDs'])
    
    fields = np.asarray(dm_data['Coordinates'])
    fields = fields.transpose()
    fields = get_displacement_field(fields, ids, box_size, grid_size)
    fields /= lr_position_std
    
    if upscale_velocities:
        lr_velocity_std = scale_params.get('LR_vel_fields_std', 1)
        
        velocity = np.asarray(dm_data['Velocities'])
        velocity = velocity.transpose()
        velocity = get_velocity_field(velocity, ids, box_size, grid_size)
        velocity /= lr_velocity_std
        fields   = np.concatenate((fields, velocity))
        del velocity
    else:
        zero_particle_velocities(dm_data, scale_factor)
        
    return fields, grid_size, box_size



def upscale_fields(fields, cosmic_scale_factor, z, generator, device):
    batch_size = 1
    patches = torch.stack([
        torch.from_numpy(patch).float() for patch in fields
    ])
    del fields
    
    dataset = TensorDataset(patches)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    del patches
    
    cosmic_scale_factor = cosmic_scale_factor.repeat(batch_size, 1).to(device)
    z = generator.tile_latent_variable(z, batch_size, device)
    
    sr_patches = []
    with torch.no_grad():
        for batch, in data_loader:
            batch = batch.to(device, non_blocking=True)
            sr_batch = generator(batch, z, cosmic_scale_factor)
            sr_batch = sr_batch.detach().to('cpu', non_blocking=True)
            if generator.nn_distance:
                sr_batch = crop(sr_batch, 1)
            sr_patches.extend([patch.numpy() for patch in sr_batch.unbind(0)])
    
    del dataset
    del data_loader
    return sr_patches



def write_enhanced_data(
        file, sr_patches, sr_grid_size, upscale_velocities, scale_params
    ):
    dm_data   = file['DMParticles']
    box_size  = file['Header'].attrs['BoxSize'][0]
    
    output_size = sr_patches[0].shape[-1]
    patches_per_dim = sr_grid_size // output_size
    volume_covered = sr_grid_size == patches_per_dim * output_size
    assert volume_covered, 'Volume not covered by SR patches'
    sr_field = stitch_fields(sr_patches, patches_per_dim)
    
    hr_position_std = scale_params.get('HR_disp_fields_std', 1)
    
    if upscale_velocities:
        hr_velocity_std = scale_params.get('HR_vel_fields_std', 1)
        sr_displacement = sr_field[:3, ...] * hr_position_std
        sr_velocity     = sr_field[3:, ...] * hr_velocity_std
        sr_velocities = sr_velocity.reshape(3, -1)
        sr_velocities = sr_velocities.transpose()
    else:
        sr_displacement = sr_field * hr_position_std
    del sr_field
        
    sr_positions = get_positions(sr_displacement, box_size, sr_grid_size)
    sr_positions = sr_positions.transpose()
    sr_ids = np.arange(sr_grid_size**3)

    del dm_data['Coordinates']
    dm_data.create_dataset('Coordinates', data=sr_positions)
    del dm_data['ParticleIDs']
    dm_data.create_dataset('ParticleIDs', data=sr_ids)
    
    if upscale_velocities:
        del dm_data['Velocities']
        dm_data.create_dataset('Velocities', data=sr_velocities)
        
    

def zero_particle_velocities(dm_data, scale_factor):
    """Replaces velocity data with zeros.
    """
    new_velocities = np.zeros_like(dm_data['Velocities'])
    new_velocities = np.tile(new_velocities, (scale_factor**3, 1))
    del dm_data['Velocities']
    dm_data.create_dataset('Velocities', data=new_velocities)


def update_particle_mass(dm_data, scale_factor):
    """Reduce the particle mass appropriately based on the scale factor.
    """
    old_mass = np.asarray(dm_data['Masses'])
    new_mass = old_mass / scale_factor**3
    new_mass = np.tile(new_mass, scale_factor**3)
    del dm_data['Masses']
    dm_data.create_dataset('Masses', data=new_mass)
    
    
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