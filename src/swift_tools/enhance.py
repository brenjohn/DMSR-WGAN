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

from .fields import get_positions, particle_ids
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
    include_velocities = generator.input_channels == 6 # TODO: no magic numbers
    
    fields, grid_size, box_size = get_field_data(
        file, scale_params, include_velocities, scale_factor
    )
    
    patches, patch_inds = cut_field(
        fields[None, ...], cut_size, cut_size, pad, return_block_indices=True
    )
    
    upscale_patches(
        file,
        patches, 
        patch_inds, 
        cosmic_scale_factor, 
        z, 
        generator,
        scale_params,
        include_velocities,
        device
    )
        


def get_field_data(file, scale_params, include_velocities, scale_factor):
    lr_position_std = scale_params.get('LR_Coordinates_std', 1)
    
    dm_data   = file['DMParticles']
    grid_size = file['ICs_parameters'].attrs['Grid Resolution']
    box_size  = file['Header'].attrs['BoxSize'][0]
    ids       = np.asarray(dm_data['ParticleIDs'])
    
    fields = np.asarray(dm_data['Coordinates'])
    fields = fields.transpose()
    fields = get_displacement_field(fields, ids, box_size, grid_size)
    fields /= lr_position_std
    
    if include_velocities:
        lr_velocity_std = scale_params.get('LR_Velocities_std', 1)
        
        velocity = np.asarray(dm_data['Velocities'])
        velocity = velocity.transpose()
        velocity = get_velocity_field(velocity, ids, box_size, grid_size)
        velocity /= lr_velocity_std
        fields   = np.concatenate((fields, velocity))
        del velocity
    else:
        zero_particle_velocities(dm_data, scale_factor)
        
    return fields, grid_size, box_size



def upscale_patches(
        file,
        patches, 
        patch_inds, 
        scale_factor, 
        z, 
        generator, 
        scale_params,
        include_velocities,
        device
    ):
    grid_size = file['ICs_parameters'].attrs['Grid Resolution']
    box_size  = file['Header'].attrs['BoxSize'][0]
    enhancment_factor = generator.scale_factor
    grid_size = grid_size * enhancment_factor
    cell_size = box_size / grid_size
    
    dm_data = file['DMParticles']
    del dm_data['Coordinates']
    pos_dset = dm_data.create_dataset(
        'Coordinates', shape=(0, 3), maxshape=(grid_size**3, 3), dtype='f8'
    )
    
    del dm_data['ParticleIDs']
    ids_dset = dm_data.create_dataset(
        'ParticleIDs', shape=(0,), maxshape=(grid_size**3,), dtype='u8'
    )
    
    if include_velocities:
        del dm_data['Velocities']
        vel_dset = dm_data.create_dataset(
            'Velocities', shape=(0, 3), maxshape=(grid_size**3, 3), dtype='f4'
        )
    
    batch_size = 1
    patches = torch.from_numpy(patches).float()
    patch_inds = torch.from_numpy(patch_inds).float()
    
    dataset = TensorDataset(patches, patch_inds)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    
    scale_factor = scale_factor.repeat(batch_size, 1).to(device)
    z = generator.tile_latent_variable(z, batch_size, device)
    
    sr_patch_size = generator.output_size - 2 * generator.nn_distance
    r = np.arange(0, sr_patch_size, dtype=np.uint64)
    relative_grid_inds = np.stack(np.meshgrid(r, r, r, indexing='ij'))
    
    hr_position_std = scale_params.get('HR_Coordinates_std', 1)
    hr_velocity_std = scale_params.get('HR_Velocities_std', 1)
    
    with torch.no_grad():
        for (i, (batch, inds)) in enumerate(data_loader):
            print(f"Processing batch {i+1} of {len(data_loader)}")
            batch = batch.to(device)
            sr_batch = generator(batch, z, scale_factor)
            sr_batch = sr_batch.detach().to('cpu')
            if generator.nn_distance:
                sr_batch = crop(sr_batch, 1)
            
            sr_displacements = sr_batch[:, :3, ...].numpy() * hr_position_std
            inds = inds.numpy().astype(np.uint64) * enhancment_factor
            
            grid_indices = relative_grid_inds + inds.T[..., None, None]
            ids = particle_ids(grid_indices.reshape(3, -1), grid_size)
            # Write to disk
            ids_dset.resize(ids_dset.shape[0] + ids.shape[0], axis=0)
            ids_dset[-ids.shape[0]:] = ids
            
            positions = (grid_indices * cell_size + sr_displacements)
            positions = positions.reshape(3, -1).T
            positions %= box_size
            # Write to disk
            pos_dset.resize(pos_dset.shape[0] + positions.shape[0], axis=0)
            pos_dset[-positions.shape[0]:] = positions
            
            if include_velocities:
                sr_velocities = sr_batch[:, 3:, ...].numpy() * hr_velocity_std
                velocities = sr_velocities.reshape(3, -1).T
                # Write to disk
                vel_dset.resize(vel_dset.shape[0] + velocities.shape[0], axis=0)
                vel_dset[-velocities.shape[0]:] = velocities
        
    

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