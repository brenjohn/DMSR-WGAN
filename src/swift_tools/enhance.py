#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:50:49 2025

@author: brennan

This file defines functions for enhancing the dark matter data in a swift 
snapshot using a dmsr generator model.
"""

import os
import time
import torch
import shutil
import h5py as h5
import numpy as np

from .fields import particle_ids, cut_field
from .fields import get_displacement_field, get_velocity_field
from dmsr.data_tools import crop

from torch.utils.data import DataLoader, TensorDataset


class ChunkWriter:
    """The ChunkWriter class is used to collect enhanced patches of a SWIFT
    snapshot and save them in a hdf5 file. The patches are written to disk in 
    chunks to reduce IO overhead.
    """
    
    def __init__(self, file, include_velocities, enhancment_factor):
        self.file = file
        self.include_velocities = include_velocities
        self.enhancment_factor = enhancment_factor
        
        self.grid_size = file['ICs_parameters'].attrs['Grid Resolution']
        self.grid_size *= enhancment_factor
        self.box_size  = file['Header'].attrs['BoxSize'][0]
        self.cell_size = self.box_size / self.grid_size
        
        self.current_idx = 0
        self.batches_per_chunk = 4
        self.ids_buffer = []
        self.pos_buffer = []
        self.vel_buffer = []
        self.create_datasets()
        
        
    def create_datasets(self):
        chunk_dim = self.batches_per_chunk * 64**3
        pos_chunks = (chunk_dim, 3)
        ids_chunks = (chunk_dim,)
        vel_chunks = (chunk_dim, 3)
        
        dm_data = self.file['DMParticles']
        del dm_data['Coordinates']
        self.pos_dset = dm_data.create_dataset(
            'Coordinates', shape=(self.grid_size**3, 3), 
            dtype='f8', compression='gzip', compression_opts=4,
            chunks=pos_chunks
        )
        
        del dm_data['ParticleIDs']
        self.ids_dset = dm_data.create_dataset(
            'ParticleIDs', shape=(self.grid_size**3,), 
            dtype='u8', compression='gzip', compression_opts=4,
            chunks=ids_chunks
        )
        
        if self.include_velocities:
            del dm_data['Velocities']
            self.vel_dset = dm_data.create_dataset(
                'Velocities', shape=(self.grid_size**3, 3), 
                dtype='f4', compression='gzip', compression_opts=4,
                chunks=vel_chunks
            )
            
    def write(self, ids, pos, vel, batch, is_last_batch):
        
        self.ids_buffer.append(ids)
        self.pos_buffer.append(pos)
        self.vel_buffer.append(vel)
        
        if not ((batch + 1) % self.batches_per_chunk == 0 or is_last_batch):
            return
        
        ids_chunk = np.concatenate(self.ids_buffer)
        pos_chunk = np.concatenate(self.pos_buffer)
        
        num_new_rows = ids_chunk.shape[0]
        current_idx = self.current_idx
        
        self.ids_dset[current_idx : current_idx + num_new_rows] = ids_chunk
        self.pos_dset[current_idx : current_idx + num_new_rows] = pos_chunk
        
        if self.include_velocities:
            vel_chunk = np.concatenate(self.vel_buffer)
            self.vel_dset[current_idx : current_idx + num_new_rows] = vel_chunk
        
        self.ids_buffer.clear()
        self.pos_buffer.clear()
        self.vel_buffer.clear()
        self.current_idx += num_new_rows


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
    
    enhancment_factor = generator.scale_factor
    writer = ChunkWriter(file, include_velocities, enhancment_factor)
    grid_size = writer.grid_size
    box_size  = writer.box_size
    cell_size = writer.cell_size
    
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
    
    print("Starting batch processing...")
    
    with torch.no_grad():
        for (i, (batch, inds)) in enumerate(data_loader):
            print(f"Processing batch {i+1} of {len(data_loader)}")
            initial_time = time.time()
            
            batch = batch.to(device)
            sr_batch = generator(batch, z, scale_factor)
            sr_batch = sr_batch.detach().to('cpu')
            if generator.nn_distance:
                sr_batch = crop(sr_batch, 1)
                
            upscaling_time = time.time()
            
            sr_displacements = sr_batch[:, :3, ...].numpy() * hr_position_std
            inds = inds.numpy().astype(np.uint64) * enhancment_factor
            grid_indices = relative_grid_inds + inds.T[..., None, None]
            ids = particle_ids(grid_indices.reshape(3, -1), grid_size)
            
            conversion_time = time.time()
            
            positions = (grid_indices * cell_size + sr_displacements)
            positions = positions.reshape(3, -1).T
            positions %= box_size
            
            velocities = None
            if include_velocities:
                sr_velocities = sr_batch[:, 3:, ...].numpy() * hr_velocity_std
                velocities = sr_velocities.reshape(3, -1).T
                
            is_last_batch = (i + 1) == len(data_loader)
            writer.write(ids, positions, velocities, i, is_last_batch)
            
            disk_write_time = time.time()
            print(f'upscaling time: {upscaling_time - initial_time}')
            print(f'conversion time: {conversion_time - upscaling_time}')
            print(f'disk write time: {disk_write_time - conversion_time}')
        
    

def zero_particle_velocities(dm_data, scale_factor):
    """Replaces velocity data with zeros.
    """
    new_velocities = np.zeros_like(dm_data['Velocities'])
    new_velocities = np.tile(new_velocities, (scale_factor**3, 1))
    del dm_data['Velocities']
    dm_data.create_dataset(
        'Velocities', 
        data=new_velocities, 
        compression='gzip', 
        compression_opts=4
    )


def update_particle_mass(dm_data, scale_factor):
    """Reduce the particle mass appropriately based on the scale factor.
    """
    old_mass = np.asarray(dm_data['Masses'])
    new_mass = old_mass / scale_factor**3
    new_mass = np.tile(new_mass, scale_factor**3)
    del dm_data['Masses']
    dm_data.create_dataset(
        'Masses', 
        data=new_mass,
        compression='gzip',
        compression_opts=4
    )
    
    
def update_potentials(dm_data, scale_factor):
    """Replaces potential data with zeros.
    """
    new_potentials = np.zeros_like(dm_data['Potentials'])
    new_potentials = np.tile(new_potentials, scale_factor**3)
    del dm_data['Potentials']
    dm_data.create_dataset(
        'Potentials', 
        data=new_potentials,
        compression='gzip',
        compression_opts=4
    )


def update_softenings(dm_data, scale_factor):
    """Reduce the softening length appropriately based on the scale factor.
    """
    old_soft = np.asarray(dm_data['Softenings'])
    new_soft = old_soft / scale_factor
    new_soft = np.tile(new_soft, scale_factor**3)
    del dm_data['Softenings']
    dm_data.create_dataset(
        'Softenings', 
        data=new_soft,
        compression='gzip',
        compression_opts=4
    )