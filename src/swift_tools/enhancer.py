#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:50:49 2025

@author: brennan

This file defines the DMSR Enhancer class for enhancing the dark matter data in
a swift snapshot using a dmsr generator model.
"""

import time
import torch
import numpy as np
from h5py import File

from .fields import particle_ids, cut_field
from .enhance_io import prepare_sr_snapshot, get_field_data
from dmsr.data_tools import crop
from dmsr.wgan import DMSRGenerator

from torch.utils.data import DataLoader, TensorDataset


class DMSREnhancer:
    """DMSR Enhancer objects can be used to load a trained DMSR generator model
    and use it to enhance the particle resolution of SWIFT snapshots.
    """
    
    def __init__(self, model_dir, device):
        # Load the generator model and draw a sample from its latent space to
        # enhance all snapshots.
        generator = DMSRGenerator.load(model_dir, device).eval()
        self.generator = generator
        
        # Load any scaling parameters if they exist.
        scale_path = model_dir / "normalisation.npy"
        scale_params = None
        if scale_path.exists():
            scale_params = np.load(scale_path, allow_pickle=True).item()
            scale_params = {k : v.item() for k, v in scale_params.items()}
        self.scale_params = scale_params
        
        velocities = generator.input_channels == 6 #TODO: no magic numbers
        self.include_velocities = velocities
        self.batch_size         = 1
        self.model_dir          = model_dir
        self.device             = device
        self.upscale_factor     = generator.scale_factor
        self.chunk_size         = generator.output_size ** 3
        self.cut_size           = generator.inner_region
        self.pad                = generator.padding
        self.sr_patch_size = generator.output_size - 2 * generator.nn_distance
        
        # 
        self.sample_latent_space()
        
    
    def sample_latent_space(self):
        """Draws a sample from the generator's latent space and stores it as
        an attribute. It will be used as an argument for the generator model
        to enhance data.
        """
        self.z = self.generator.sample_latent_space(1, self.device)
        
    
    def prepare_sr_snapshot(self, lr_file, sr_file):
        """Factory function to create a sr snapshot for the given lr snapshot 
        and a writer object to handle writing enhanced data to it.
        """
        return prepare_sr_snapshot(
            lr_file,
            sr_file,
            self.upscale_factor,
            self.include_velocities,
            self.chunk_size
        )
        
    
    def prepare_generator_args(self, lr_file):
        """Prepares the style and latent variable input arguments for the 
        generator. 
        """
        # Create a style parameter from the cosmological scale factor.
        cosmo_scale_factor = lr_file['Cosmology'].attrs['Scale-factor']
        style = None
        if self.generator.style_size is not None:
            style = torch.tensor(
                [cosmo_scale_factor], 
                dtype=torch.float32,
                device = self.device
            ).repeat(self.batch_size, 1).to(self.device)
        self.style = style
        
        # Tile the latent variable so that it has periodic boundary conditions 
        # and the desired batch size.
        self.tiled_z = self.generator.tile_latent_variable(
            self.z, self.batch_size, self.device
        )
    
    
    def prepare_lr_patches(self, lr_file):
        """Creates a dataloader to load patches of lr data that can be enhanced
        by the generator. The dataloader also loads tensors containing the grid
        indices of the lr particles.
        """
        fields, grid_size, box_size = get_field_data(
            lr_file, self.scale_params, self.include_velocities
        )
        
        patches, patch_inds = cut_field(
            fields   = fields[None, ...], 
            cut_size = self.cut_size, 
            stride   = self.cut_size, 
            pad      = self.pad, 
            return_block_indices = True
        )
        
        # Prepare LR data for upscaling.
        dataset = TensorDataset(
            torch.from_numpy(patches).float(), 
            torch.from_numpy(patch_inds).float()
        )
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True)

    
    def enhance(self, lr_path, sr_path):
        """Use the generator to enhance the snapshot at `lr_path` and save the 
        result in a snapshot file at `sr_path`
        """
        sr_path.unlink(missing_ok=True)
        
        with File(sr_path, 'w') as sr_file, File(lr_path, 'r') as lr_file:
            # Prepare style and latent variable arguments for the generator.
            self.prepare_generator_args(lr_file)
            
            # Create a dataloader to load patches from the lr file.
            lr_patches = self.prepare_lr_patches(lr_file)
            
            # Create the sr snapshot to write enhanced data to and get a writer
            # object to handle writing data to it.
            writer = self.prepare_sr_snapshot(lr_file, sr_file)
            
            # Enhance the lr patches into sr patches and write them to disk.
            self.upscale_patches(writer, lr_patches)
    
    
    def upscale_patches(self, writer, lr_patches):
        """Upscales LR patches to SR particles and maps them to the global 
        simulation box.
        """
        # Create grid indices for a regular grid with the same shape as SR 
        # batches. This is used to convert SR displacements into positions.
        r = np.arange(0, self.sr_patch_size, dtype=np.uint64)
        relative_grid_inds = np.stack(np.meshgrid(r, r, r, indexing='ij'))
        
        # Normalisation statistics.
        hr_position_std = self.scale_params.get('HR_Coordinates_std', 1)
        hr_velocity_std = self.scale_params.get('HR_Velocities_std', 1)
        
        # Process LR batches into SR batches and write them to disk.
        DIM_XYZ = slice(0, 3)  # displacement slice.
        DIM_UVW = slice(3, 6)  # velocity slice
        with torch.no_grad():
            for (i, (batch, inds)) in enumerate(lr_patches):
                initial_time = time.perf_counter()
                
                # STEP 1: Enhance LR data using the generator.
                batch = batch.to(self.device)
                sr_batch = self.generator(batch, self.tiled_z, self.style)
                sr_batch = sr_batch.detach().to('cpu')
                if self.generator.nn_distance:
                    sr_batch = crop(sr_batch, 1)
                    
                upscaling_time = time.perf_counter()
                
                # STEP 2: Get Particle IDs and convert displacements to 
                # positions.
                sr_displacements = sr_batch[:, DIM_XYZ, ...].numpy()
                sr_displacements *= hr_position_std
                inds = inds.numpy().astype(np.uint64) * self.upscale_factor
                grid_inds = relative_grid_inds + inds.T[..., None, None]
                ids = particle_ids(grid_inds.reshape(3, -1), writer.grid_size)
                
                positions = (grid_inds * writer.cell_size + sr_displacements)
                positions = positions.reshape(3, -1).T
                positions %= writer.box_size
                conversion_time = time.perf_counter()
                
                # STEP 3: Process velocity field.
                velocities = None
                if self.include_velocities:
                    sr_velocities = sr_batch[:, DIM_UVW, ...].numpy()
                    sr_velocities *= hr_velocity_std
                    velocities = sr_velocities.reshape(3, -1).T
                
                # STEP 4: Write enhanced data to disk.
                is_last_batch = (i + 1) == len(lr_patches)
                writer.write(ids, positions, velocities, i, is_last_batch)
                
                disk_write_time = time.perf_counter()
                upscaling_seconds = upscaling_time - initial_time
                conversion_seconds = conversion_time - upscaling_time
                disk_write_seconds = disk_write_time - conversion_time
                print(f"Batch {i+1} of {len(lr_patches)}")
                print(f'upscaling time: {upscaling_seconds:.4f}')
                print(f'conversion time: {conversion_seconds:.4f}')
                print(f'disk write time: {disk_write_seconds:.4f}')