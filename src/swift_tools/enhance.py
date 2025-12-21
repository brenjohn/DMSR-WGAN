#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:50:49 2025

@author: brennan

This file defines functions for enhancing the dark matter data in a swift 
snapshot using a dmsr generator model.
"""

import time
import torch
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



def enhance(lr_path, sr_path, generator, scale_params, device):
    """Use the given generator to enhance the snapshot at `lr_path` and save 
    the result in a snapshot file at `sr_path`
    """
    sr_path.unlink(missing_ok=True)
    
    with h5.File(sr_path, 'w') as sr_file, h5.File(lr_path, 'r') as lr_file:
        # Prepare style argument for the generator.
        prepare_generator(lr_file, generator, scale_params, device)
        upscale_factor     = generator.scale_factor
        include_velocities = generator.include_velocities
        chunk_size         = generator.output_size ** 3
        
        # Copy all data that won't be enhanced from the lr file to the sr file.
        copy_metadata(lr_file, sr_file, upscale_factor, include_velocities)
        
        # Create a dataloader to load patches from the lr file to be enhanced.
        lr_patches = prepare_lr_patches(lr_file, generator)
        
        # Create a chunk writer to write enhanced patches to the sr file.
        writer = ChunkWriter(sr_file, include_velocities, upscale_factor, chunk_size)
        
        # Enhance the lr patches into sr patches and write them to disk.
        upscale_patches(writer, lr_patches, generator)
        
 
        
def prepare_generator(lr_file, generator, scale_params, device):
    """Prepares the style and latent variable input arguments for the 
    generator. 
    """
    batch_size = 1
    include_velocities = generator.input_channels == 6 # TODO: no magic numbers
    
    generator.include_velocities = include_velocities
    generator.batch_size         = batch_size
    generator.scale_params       = scale_params
    generator.device             = device
    
    # Create a style parameter from the cosmological scale factor.
    cosmo_scale_factor = lr_file['Cosmology'].attrs['Scale-factor']
    style = None
    if generator.style_size is not None:
        style = torch.tensor(
            [cosmo_scale_factor], 
            dtype=torch.float32,
            device = device
        ).repeat(batch_size, 1).to(device)
    generator.style = style
    
    # Tile the latent variable so that it has periodic boundary conditions and 
    # the desired batch size.
    z = generator.z
    generator.z = generator.tile_latent_variable(z, batch_size, device)
        
     
        
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



def prepare_lr_patches(lr_file, generator):
    """Creates a dataloader to load patches of lr data that can be enhanced
    by the generator. The dataloader also loads tensors containing the grid
    indices of the lr particles.
    """
    cut_size           = generator.inner_region
    pad                = generator.padding
    upscale_factor       = generator.scale_factor
    include_velocities = generator.include_velocities
    batch_size         = generator.batch_size
    
    fields, grid_size, box_size = get_field_data(
        lr_file, generator.scale_params, include_velocities, upscale_factor
    )
    
    patches, patch_inds = cut_field(
        fields[None, ...], cut_size, cut_size, pad, return_block_indices=True
    )
    
    # Prepare LR data for upscaling.
    dataset = TensorDataset(
        torch.from_numpy(patches).float(), 
        torch.from_numpy(patch_inds).float()
    )
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True)
        


def get_field_data(lr_file, scale_params, include_velocities, upscale_factor):
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



def upscale_patches(
        writer,
        lr_patches,
        generator
    ):
    """Upscales LR patches to SR particles and maps them to the global 
    simulation box.
    """
    # Create grid indices for a regular grid with the same shape as SR batches.
    # This is used to convert SR displacements into positions.
    sr_patch_size = generator.output_size - 2 * generator.nn_distance
    r = np.arange(0, sr_patch_size, dtype=np.uint64)
    relative_grid_inds = np.stack(np.meshgrid(r, r, r, indexing='ij'))
    
    # Normalisation statistics.
    hr_position_std = generator.scale_params.get('HR_Coordinates_std', 1)
    hr_velocity_std = generator.scale_params.get('HR_Velocities_std', 1)
    
    # Process LR batches into SR batches and write them to disk.
    DIM_XYZ = slice(0, 3)  # displacement slice.
    DIM_UVW = slice(3, 6)  # velocity slice
    with torch.no_grad():
        for (i, (batch, inds)) in enumerate(lr_patches):
            initial_time = time.perf_counter()
            
            # STEP 1: Enhance LR data using the generator.
            batch = batch.to(generator.device)
            sr_batch = generator(batch, generator.z, generator.style)
            sr_batch = sr_batch.detach().to('cpu')
            if generator.nn_distance:
                sr_batch = crop(sr_batch, 1)
                
            upscaling_time = time.perf_counter()
            
            # STEP 2: Get Particle IDs and convert displacements to positions.
            sr_displacements = sr_batch[:, DIM_XYZ, ...].numpy()
            sr_displacements *= hr_position_std
            inds = inds.numpy().astype(np.uint64) * generator.scale_factor
            grid_indices = relative_grid_inds + inds.T[..., None, None]
            ids = particle_ids(grid_indices.reshape(3, -1), writer.grid_size)
            
            positions = (grid_indices * writer.cell_size + sr_displacements)
            positions = positions.reshape(3, -1).T
            positions %= writer.box_size
            conversion_time = time.perf_counter()
            
            # STEP 3: Process velocity field.
            velocities = None
            if generator.include_velocities:
                sr_velocities = sr_batch[:, DIM_UVW, ...].numpy()
                sr_velocities *= hr_velocity_std
                velocities = sr_velocities.reshape(3, -1).T
            
            # STEP 4: Write enhanced data to disk.
            is_last_batch = (i + 1) == len(lr_patches)
            writer.write(ids, positions, velocities, i, is_last_batch)
            
            disk_write_time = time.perf_counter()
            print(f"Batch {i+1} of {len(lr_patches)}")
            print(f'upscaling time: {upscaling_time - initial_time:.4f}')
            print(f'conversion time: {conversion_time - upscaling_time:.4f}')
            print(f'disk write time: {disk_write_time - conversion_time:.4f}')

    
    
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