#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:02:21 2024

@author: brennan

This script will create a dataset for training a DMSR-WGAN from swift 
snapshots. It creates field patches from simulation snapshots and saves them as 
individual HDF5 files.

Each file represents a patch, and can include multiple datasets :
(LR displacement, LR velocity, HR displacement, HR velocity, scale factor)
"""

import json
import h5py
import argparse
import numpy as np
import multiprocessing as mp

from pathlib import Path
from swift_tools.fields import cut_field
from swift_tools.data import read_metadata, read_particle_data
from swift_tools.fields import get_displacement_field, get_velocity_field

FIELD_METHODS = {
    'Coordinates' : get_displacement_field,
    'Velocities'  : get_velocity_field,
}

def create_fields(patch_args):
    """
    Reads particle data, converts it into a continuous field (displacement or 
    velocity), and cuts the field into patches. Each patch is saved as a 
    dataset within a single HDF5 file, using append mode.
    """
    snapshot = patch_args['snapshot']
    particle_data_name = patch_args['particle_data_name']
    dataset_name = f'{patch_args["prefix"]}_{particle_data_name}'
    
    grid_size, box_size, mass, h, a = read_metadata(snapshot)
    IDs = read_particle_data(snapshot, 'ParticleIDs')
    particle_data = read_particle_data(snapshot, particle_data_name)
    
    particle_data = particle_data.transpose()
    field_data = FIELD_METHODS[particle_data_name](
        particle_data, IDs, box_size, grid_size
    )
    
    patches = cut_field(
        field_data[None,...], 
        patch_args['inner_size'],
        stride = patch_args['stride'] * patch_args['inner_size'],
        pad    = patch_args['padding']
    )
        
    for num, patch in enumerate(patches):
        patch_num = patch_args['snap_num'] * patch_args['patches_per_snapshot']
        patch_num += num
        patch_file = patch_args['output_dir'] / f"patch_{patch_num}.h5"
        
        with h5py.File(patch_file, 'a') as file:
            file.create_dataset(dataset_name, data=patch, compression="gzip")
            file.attrs['scale_factor'] = a
            
            
def create_patches(patch_args):
    """Manages the parallel processing of multiple simulation snapshots by 
    distributing `create_fields` tasks across a multiprocessing pool.
    """
    print(
        'Creating', 
        patch_args["prefix"], 
        patch_args["particle_data_name"],
        'patches.'
    )
    with mp.Pool(patch_args['num_procs']) as pool:
        tasks = []
        for num, snapshot in enumerate(patch_args['snapshots']):
            task_Args = patch_args | {'snapshot' : snapshot, 'snap_num' : num}
            tasks.append(task_Args)
        pool.starmap(create_fields, tasks)


def create_metadata(LR_args, HR_args):
    """
    Reads metadata from the first Low-Resolution (LR) and High-Resolution (HR) 
    snapshot, calculates derived patch properties (physical length, size), 
    and saves all configuration and metadata to a numpy file.

    It also updates the `LR_args` and `HR_args` dictionaries with the total 
    number of patches expected per snapshot.
    """
    print('Creating metadata.')
    LR_metadata = read_metadata(LR_args['snapshots'][0])
    HR_metadata = read_metadata(HR_args['snapshots'][0])
    LR_grid_size, box_size, LR_mass, h, a = LR_metadata
    HR_grid_size, box_size, HR_mass, h, a = HR_metadata
    
    LR_patch_size = LR_args['inner_size'] + 2 * LR_args['padding']
    HR_patch_size = HR_args['inner_size'] + 2 * HR_args['padding']
    
    meta_file = LR_args.output_dir / 'metadata.npy'
    np.save(meta_file, {
        'box_size'        : box_size,
        'LR_patch_length' : LR_patch_size * box_size / LR_grid_size,
        'HR_patch_length' : HR_patch_size * box_size / HR_grid_size,
        'LR_patch_size'   : LR_patch_size,
        'HR_patch_size'   : HR_patch_size,
        'LR_inner_size'   : LR_args['inner_size'],
        'HR_inner_size'   : HR_args['inner_size'],
        'LR_padding'      : LR_args['padding'],
        'HR_padding'      : HR_args['padding'],
        'LR_mass'         : LR_mass,
        'HR_mass'         : HR_mass,
        'hubble'          : h
    })
    
    patches_per_snapshot = (LR_grid_size // LR_args['inner_size'])**3
    LR_args['patches_per_snapshot'] = patches_per_snapshot
    HR_args['patches_per_snapshot'] = patches_per_snapshot


def read_args(args):
    """
    Loads configuration settings from the specified JSON file, combines them 
    with command-line arguments (like num_procs), and resolves file paths 
    and glob patterns.
    """
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        
    base_config = config['base']
    LR_config = config['LR_patch_args']
    HR_config = config['HR_patch_args']
    
    output_dir = Path(base_config['output_dir'])
    data_dir = Path(base_config['data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LR_snapshots = sorted(data_dir.glob(LR_config['snapshot_glob']))
    HR_snapshots = sorted(data_dir.glob(HR_config['snapshot_glob']))
    
    base_args = {
        'output_dir'         : output_dir,
        'stride'             : base_config['stride'],
        'include_velocities' : base_config['include_velocities'],
        'num_procs'          : args.num_procs
    }

    HR_patch_args = base_args | {
        'prefix'    :'HR',
        'snapshots' : HR_snapshots, 
        **HR_config
    }
    
    LR_patch_args = base_args | {
        'prefix'    :'LR',
        'snapshots' : LR_snapshots, 
        **LR_config
    }
    
    return LR_patch_args, HR_patch_args
    

def main(args):
    """Main entry point for the patch creation script.
    
    Handles setup, configuration loading, metadata creation, and orchestrates 
    the parallel patch creation for both Low and High-Resolution data streams.
    """
    # Read the arguments and configuration file.
    LR_patch_args, HR_patch_args = read_args(args)
    
    # Get metadata for the dataset and write it to the output dir. 
    create_metadata(LR_patch_args, HR_patch_args)
    
    # Define fields to process.
    fields = ('Coordinates',)
    if LR_patch_args['include_velocities']:
        fields += ('Velocities',)
    
    # Create patch files.
    for patch_args in (LR_patch_args, HR_patch_args):
        for particle_data_name in fields:
            patch_args['particle_data_name'] = particle_data_name
            create_patches(patch_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a dataset."
    )
    
    parser.add_argument(
        '--config_file',
        type=Path,
        default='config.json',
        help="Path to the JSON configuration file."
    )
    
    parser.add_argument(
        '--num_procs', 
        type=int, 
        default=1,
        help="Number of parallel processes to use."
    )
    
    args = parser.parse_args()
    main(args)
