#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 13:19:45 2025

@author: brennan

This file defines functions and classes for handling IO related tasks that come
up while enhancing SWIFT snapshots. 
"""

import numpy as np
from .fields import get_displacement_field, get_velocity_field


def prepare_sr_snapshot(
        lr_file, sr_file, upscale_factor, include_velocities, chunk_size
    ):
    """Factory function to create a sr snapshot for the given lr snapshot and
    a writer object to handle writing enhanced data to it.
    """
    # Copy all data that won't be enhanced from the lr file to the sr file.
    copy_metadata(lr_file, sr_file, upscale_factor, include_velocities)
    
    # Create a chunk writer to write enhanced patches to the sr file.
    return ChunkWriter(sr_file, include_velocities, upscale_factor, chunk_size)



class ChunkWriter:
    """The ChunkWriter class is used to collect enhanced patches of a SWIFT
    snapshot and save them in a hdf5 file. The patches are written to disk in 
    chunks to reduce IO overhead.
    """
    
    def __init__(
            self, 
            file, 
            include_velocities, 
            upscale_factor,
            chunk_size,
            batches_per_chunk = 4
        ):
        self.file = file
        self.include_velocities = include_velocities
        self.upscale_factor = upscale_factor
        self.chunk_size = chunk_size
        self.batches_per_chunk = batches_per_chunk
        
        self.grid_size = file['ICs_parameters'].attrs['Grid Resolution']
        self.grid_size *= upscale_factor
        self.box_size  = file['Header'].attrs['BoxSize'][0]
        self.cell_size = self.box_size / self.grid_size
        
        self.current_idx = 0
        self.ids_buffer = []
        self.pos_buffer = []
        self.vel_buffer = []
        self.create_datasets()
        
        
    def create_datasets(self):
        """Creates datasets to write enhanced data to.
        """
        chunk_dim = self.chunk_size
        vector_chunks = (chunk_dim, 3)
        scalar_chunks = (chunk_dim,)
        
        dm_data = self.file['DMParticles']
        self.pos_dset = dm_data.create_dataset(
            'Coordinates', shape=(self.grid_size**3, 3), 
            dtype='f8', compression='gzip', compression_opts=4,
            chunks=vector_chunks
        )
        
        self.ids_dset = dm_data.create_dataset(
            'ParticleIDs', shape=(self.grid_size**3,), 
            dtype='u8', compression='gzip', compression_opts=4,
            chunks=scalar_chunks
        )
        
        if self.include_velocities:
            self.vel_dset = dm_data.create_dataset(
                'Velocities', shape=(self.grid_size**3, 3), 
                dtype='f4', compression='gzip', compression_opts=4,
                chunks=vector_chunks
            )
    
    
    def write(self, ids, pos, vel, batch, is_last_batch):
        """Writes the given batch of enhanced data to disk.
        """
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



def copy_metadata(lr_file, sr_file, upscale_factor, include_velocities):
    """Copies data from the given lr_file to the sr file. Only data that is not
    enhanced by the generator is copied.
    """
    # Copy attributes and groups of the lr_file to the sr_file.
    for attr_name, attr_value in lr_file.attrs.items():
        sr_file.attrs[attr_name] = attr_value
    
    for group in lr_file.keys():
        if group != 'DMParticles':
            lr_file.copy(group, sr_file)
    
    # Copy attributes and fields from the lr dm group to the sr dm group,
    # except the fields whose size needs to be increased.
    dm_sr = sr_file.create_group('DMParticles')
    dm_lr = lr_file['DMParticles']
    for attr_name, attr_value in dm_lr.attrs.items():
        dm_sr.attrs[attr_name] = attr_value
    
    fields_to_be_enlarged  = ['Masses',      'Potentials', 'Softenings']
    fields_to_be_enlarged += ['Coordinates', 'Velocities', 'ParticleIDs']
    for field_name in dm_lr.keys():
        if field_name not in fields_to_be_enlarged:
            dm_lr.copy(field_name, dm_sr)
      
    # Increase size of fields not enhanced by the generator.
    update_field(dm_sr, dm_lr, 'Masses',     upscale_factor, field_exponent=3)
    update_field(dm_sr, dm_lr, 'Softenings', upscale_factor, field_exponent=1)
    zero_field(dm_sr, dm_lr, 'Potentials', upscale_factor)
    if not include_velocities:
        zero_field(dm_sr, dm_lr, 'Velocities', upscale_factor)
    
    sr_file['ICs_parameters'].attrs['Grid Resolution'] *= upscale_factor



def update_field(dm_sr, dm_lr, field_name, upscale_factor, field_exponent):
    """Update the named field in the given dark matter dataset to scale it by
    `upscale_factor ** field_exponent` and tile it to increase the number of
    elements in the field by a factor of `upscale_factor**3`.
    """
    data = np.asarray(dm_lr[field_name]) / (upscale_factor**field_exponent)
    num_repeats = upscale_factor**3
    tile_shape = num_repeats if data.ndim == 1 else (num_repeats, 1)
    data = np.tile(data, tile_shape)
    
    # Note: If we're ever enhancing snapshots that are already large, writing
    # processed LR data to disk could be done in `num_repeats` chunks instead
    # of filling RAM first and writing it all at once.
    dm_sr.create_dataset(
        field_name, 
        data=data,
        compression='gzip',
        compression_opts=4
    )



def zero_field(dm_sr, dm_lr, field_name, upscale_factor):
    """Creates a dataset in `dm_sr` with the name `field_name` which is
    `upscale_factor**3` times larger than the same field in `dm_lr`. The
    created field is set to zero.
    """
    # Calculate target shape
    source_shape = dm_lr[field_name].shape
    new_rows = source_shape[0] * upscale_factor**3
    target_shape = (new_rows,)
    if not len(source_shape) == 1:
        target_shape += (source_shape[1],)
    
    dm_sr.create_dataset(
        field_name, 
        shape=target_shape, 
        dtype=dm_lr[field_name].dtype,
        fillvalue=0,
        compression='gzip',
        compression_opts=4
    )



def get_field_data(lr_file, scale_params, include_velocities):
    """Gathers the fields to be enhanced by the generator from the lr file. The
    total grid size and box size are also read and returned from the lr file.
    """
    lr_position_std = scale_params.get('LR_Coordinates_std', 1)
    
    dm_data   = lr_file['DMParticles']
    grid_size = lr_file['ICs_parameters'].attrs['Grid Resolution']
    box_size  = lr_file['Header'].attrs['BoxSize'][0]
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
        
    return fields, grid_size, box_size