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
        del dm_data['Coordinates']
        self.pos_dset = dm_data.create_dataset(
            'Coordinates', shape=(self.grid_size**3, 3), 
            dtype='f8', compression='gzip', compression_opts=4,
            chunks=vector_chunks
        )
        
        del dm_data['ParticleIDs']
        self.ids_dset = dm_data.create_dataset(
            'ParticleIDs', shape=(self.grid_size**3,), 
            dtype='u8', compression='gzip', compression_opts=4,
            chunks=scalar_chunks
        )
        
        if self.include_velocities:
            del dm_data['Velocities']
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


def enhance(lr_snapshot, sr_snapshot, generator, z, scale_params, device):
    """Use the given generator to enhance the `lr_snapshot` and save the result
    in `sr_snapshot`
    """
    sr_snapshot.unlink(missing_ok=True)
    shutil.copy(lr_snapshot, sr_snapshot)
    
    upscale_factor = generator.scale_factor
      
    with h5.File(sr_snapshot, 'a') as sr_file:
        # Get the cosmological scale factor and create a style parameter from
        # it if the generator requires one.
        cosmo_scale_factor = sr_file['Cosmology'].attrs['Scale-factor']
        style = None
        if generator.style_size is not None:
            style = torch.tensor(
                [cosmo_scale_factor], 
                dtype=torch.float32,
                device = device
            )
        
        # Get dark matter data and increase size of fields not enhanced by the
        # generator.
        dm_data = sr_file['DMParticles']
        update_field(dm_data, 'Masses',     upscale_factor, field_exponent=3)
        update_field(dm_data, 'Potentials', upscale_factor, 0, zero_field=True)
        update_field(dm_data, 'Softenings', upscale_factor, field_exponent=1)
        
        # Use the generator to enhance particle data.
        update_particle_data(
            sr_file, 
            generator, 
            z, 
            scale_params, 
            style, 
            device
        )
        
        # Increase grid size metadata.
        sr_file['ICs_parameters'].attrs['Grid Resolution'] *= upscale_factor



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
        update_field(dm_data, 'Velocities', scale_factor, 0, zero_field=True)
        
    return fields, grid_size, box_size



def upscale_patches(
        file,
        patches, 
        patch_inds, 
        cosmo_scale_factor, 
        z, 
        generator, 
        scale_params,
        include_velocities,
        device
    ):
    """Upscales LR patches to SR particles and maps them to the global 
    simulation box.
    """
    # Setup writer to write enhanced data to disk.
    upscale_factor = generator.scale_factor
    chunk_size = generator.output_size ** 3
    writer = ChunkWriter(file, include_velocities, upscale_factor, chunk_size)
    grid_size = writer.grid_size
    box_size  = writer.box_size
    cell_size = writer.cell_size
    
    # Prepare LR data for upscaling.
    batch_size = 1
    dataset = TensorDataset(
        torch.from_numpy(patches).float(), 
        torch.from_numpy(patch_inds).float()
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    
    # Prepare generator inputs.
    style = cosmo_scale_factor.repeat(batch_size, 1).to(device)
    z = generator.tile_latent_variable(z, batch_size, device)
    
    # Create grid indices for a regular grid with the same shape as SR batches.
    # This is used to convert SR displacements into positions.
    sr_patch_size = generator.output_size - 2 * generator.nn_distance
    r = np.arange(0, sr_patch_size, dtype=np.uint64)
    relative_grid_inds = np.stack(np.meshgrid(r, r, r, indexing='ij'))
    
    # Normalisation statistics.
    hr_position_std = scale_params.get('HR_Coordinates_std', 1)
    hr_velocity_std = scale_params.get('HR_Velocities_std', 1)
    
    # Process LR batches into SR batches and write them to disk.
    DIM_XYZ = slice(0, 3)  # displacement slice.
    DIM_UVW = slice(3, 6)  # velocity slice
    with torch.no_grad():
        for (i, (batch, inds)) in enumerate(data_loader):
            initial_time = time.perf_counter()
            
            # STEP 1: Enhance LR data using the generator.
            batch = batch.to(device)
            sr_batch = generator(batch, z, style)
            sr_batch = sr_batch.detach().to('cpu')
            if generator.nn_distance:
                sr_batch = crop(sr_batch, 1)
                
            upscaling_time = time.perf_counter()
            
            # STEP 2: Get Particle IDs and convert displacements to positions.
            sr_displacements = sr_batch[:, DIM_XYZ, ...].numpy()
            sr_displacements *= hr_position_std
            inds = inds.numpy().astype(np.uint64) * upscale_factor
            grid_indices = relative_grid_inds + inds.T[..., None, None]
            ids = particle_ids(grid_indices.reshape(3, -1), grid_size)
            
            positions = (grid_indices * cell_size + sr_displacements)
            positions = positions.reshape(3, -1).T
            positions %= box_size
            conversion_time = time.perf_counter()
            
            # STEP 3: Process velocity field.
            velocities = None
            if include_velocities:
                sr_velocities = sr_batch[:, DIM_UVW, ...].numpy()
                sr_velocities *= hr_velocity_std
                velocities = sr_velocities.reshape(3, -1).T
            
            # STEP 4: Write enhanced data to disk.
            is_last_batch = (i + 1) == len(data_loader)
            writer.write(ids, positions, velocities, i, is_last_batch)
            
            disk_write_time = time.perf_counter()
            print(f"Batch {i+1} of {len(data_loader)}")
            print(f'upscaling time: {upscaling_time - initial_time:.4f}')
            print(f'conversion time: {conversion_time - upscaling_time:.4f}')
            print(f'disk write time: {disk_write_time - conversion_time:.4f}')

    
    
def update_field(
        dm_data, field_name, upscale_factor, field_exponent, zero_field=False
    ):
    """Update the named field in the given dark matter dataset to scale it by
    `upscale_factor ** field_exponent` and tile it to increase the number of
    elements in the field by a factor of `upscale_factor**3`.
    
    If `zero_field` is True, the field data is set to zero instead of being
    rescaled.
    """
    # Get the field data and zero it if specified. Otherwise scale it according
    # to the given field exponent.
    field = np.asarray(dm_data[field_name])
    if zero_field:
        field = np.zeros_like(field)
    else:
        field = field / (upscale_factor**field_exponent)
    
    # Tile the field to cover the enhanced region.
    num_repeats = upscale_factor**3
    tile_shape = num_repeats if field.ndim == 1 else (num_repeats, 1)
    field = np.tile(field, tile_shape)
    
    # Create the new dataset.
    del dm_data[field_name]
    dm_data.create_dataset(
        field_name, 
        data=field,
        compression='gzip',
        compression_opts=4
    )