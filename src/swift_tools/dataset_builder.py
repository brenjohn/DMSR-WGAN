#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:29:07 2026

@author: brennan
"""

import tomllib
import h5py
import json
import multiprocessing as mp

from pathlib import Path
from .fields import cut_field
from .data import read_metadata, read_particle_data
from .fields import get_displacement_field, get_velocity_field

from dm_analysis import power_spectrum
from swift_tools.fields import get_positions
from dm_analysis import cloud_in_cells


FIELD_METHODS = {
    'Coordinates' : get_displacement_field,
    'Velocities'  : get_velocity_field,
}


class DatasetBuilder:
    
    def __init__(self, args):
        with open(args.config_file, 'rb') as f:
            self.config = tomllib.load(f)
        
        self.num_procs     = args.num_procs
        self.velocities    = self.config['base']['include_velocities']
        self.spectra       = self.config['base']['include_spectra']
        self.LR_patch_args = self.get_patch_args('LR')
        self.HR_patch_args = self.get_patch_args('HR')
        self.read_metadata()
    
    
    def get_patch_args(self, prefix):
        """Returns configuration dict for creating LR or HR patches.
        """
        base_config = self.config['base']
        config      = self.config[prefix + '_patch_args']

        output_dir = Path(base_config['output_dir'])
        data_dir   = Path(base_config['data_dir'])
        patch_dir  = output_dir / 'patches'
        patch_dir.mkdir(parents=True, exist_ok=True)
        
        snapshots = sorted(data_dir.glob(config['snapshot_glob']))
        return {
            'output_dir'         : output_dir,
            'patch_dir'          : patch_dir,
            'stride'             : base_config['stride'],
            'include_velocities' : base_config['include_velocities'],
            'num_procs'          : self.num_procs,
            'prefix'             : prefix,
            'snapshots'          : snapshots, 
            **config
        }
    
    
    def read_metadata(self):
        """
        Read metadata from the first LR and HR snapshots and saves it to a
        metadata dict attribute.

        It also updates the `LR_patch_args` and `HR_patch_args` attributes with 
        the total number of patches expected per snapshot.
        """
        LR_args, HR_args = self.LR_patch_args, self.HR_patch_args
        LR_metadata = read_metadata(LR_args['snapshots'][0])
        HR_metadata = read_metadata(HR_args['snapshots'][0])
        LR_grid_size, box_size, LR_mass, h, a = LR_metadata
        HR_grid_size, box_size, HR_mass, h, a = HR_metadata
        
        LR_patch_size = LR_args['inner_size'] + 2 * LR_args['padding']
        HR_patch_size = HR_args['inner_size'] + 2 * HR_args['padding']
        
        self.metadata = {
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
        }
        
        patches_per_snapshot = (LR_grid_size // LR_args['inner_size'])**3
        self.LR_patch_args['patches_per_snapshot'] = patches_per_snapshot
        self.HR_patch_args['patches_per_snapshot'] = patches_per_snapshot
    
    
    def save_metadata(self):
        """Saves metadata to a json file.
        """
        serializable_meta = {
            k: (v.item() if hasattr(v, 'item') else v) 
            for k, v in self.metadata.items()
        }
        meta_file = self.LR_patch_args['output_dir'] / 'metadata.json'
        with open(meta_file, 'w') as file:
            json.dump(serializable_meta, file, indent=2)
    
    
    def create_dataset(self):
        """Creates dataset patches consiting of LR and HR data. Metadata is
        also updated with summary statistics of the data.
        """
        self.metadata |= create_patches(self.LR_patch_args, 'Coordinates')
        self.metadata |= create_patches(self.HR_patch_args, 'Coordinates')
        
        if self.velocities:
            self.metadata |= create_patches(self.LR_patch_args, 'Velocities')
            self.metadata |= create_patches(self.HR_patch_args, 'Velocities')
            
        if self.spectra:
            print("Computing power spectra")
            self.add_power_spectra()
    
    
    def add_power_spectra(self):
        """Adds the power spectrum for the HR displacement field to every patch
        file in the dataset.
        """
        patch_dir = self.HR_patch_args['patch_dir']
        
        HR_patch_length = self.metadata['HR_patch_length']
        HR_patch_size   = self.metadata['HR_patch_size']
        HR_mass         = self.metadata['HR_mass']
        grid_size = int(HR_patch_size)
        box_size  = HR_patch_length
        
        for patch_file in patch_dir.iterdir():
            with h5py.File(patch_file, 'a') as file:
                patch = file['HR_Coordinates'][()]
                
                positions = get_positions(
                    patch, box_size, grid_size, periodic=False
                )
                density = HR_mass * cloud_in_cells(
                    positions.T, grid_size, box_size, periodic=False
                )
                k_bins, spectrum, _ = power_spectrum(
                    density, box_size, grid_size
                )
                
                file.create_dataset('HR_Power_Spectrum', data = spectrum)



def create_patches(patch_args, particle_data_name):
    """Creates patches using the config from 'patch_args' and from data with 
    the name 'particle_data_name'.
    
    Snapshots are proccessed in parallel and summary statistics of the data are
    collected from each process in a list. These stats are reduced to a single 
    dict using Welford algorithm.
    """
    patch_args['particle_data_name'] = particle_data_name
    data_name = patch_args["prefix"] + '_' + particle_data_name
    print('Creating', data_name, 'patches.')
    
    with mp.Pool(patch_args['num_procs']) as pool:
        tasks = []
        for num, snapshot in enumerate(patch_args['snapshots']):
            task_args = patch_args | {'snapshot' : snapshot, 'snap_num' : num}
            tasks.append(task_args)
        stats = pool.map(convert_field_to_patches, tasks)
        
    return welford_reduction(stats, data_name)



def welford_reduction(stats, data_name):
    """
    Use Welford's algorithm to reduce a list of summary statistics to a single
    dict. The stats list is assumed to contain (mean, variance, num) tuples.
    The given data name is used to create keys for the returned dict.
    """
    count, mean, M2 = 0, 0.0, 0.0
    for (next_mean, next_var, next_n) in stats:
        delta = next_mean - mean
        total = count + next_n
        
        mean += delta * next_n / total
        M2   += next_var * next_n + delta**2 * count * next_n / total
        count = total
    
    variance = M2 / count
    return {
        data_name + '_mean' : mean, 
        data_name + '_std'  : variance**0.5
    }



def convert_field_to_patches(snapshot):
    """
    Reads particle data, converts it into a continuous field (displacement or 
    velocity), and cuts it into patches. Each field patch is saved to a patch
    hdf5 file. The mean, variance and sample size are then returned.
    """
    snap_path = snapshot['snapshot']
    particle_data_name = snapshot['particle_data_name']
    dataset_name = f'{snapshot["prefix"]}_{particle_data_name}'
    
    grid_size, box_size, mass, h, a = read_metadata(snap_path)
    IDs = read_particle_data(snap_path, 'ParticleIDs')
    particle_data = read_particle_data(snap_path, particle_data_name)
    
    particle_data = particle_data.transpose()
    field_data = FIELD_METHODS[particle_data_name](
        particle_data, IDs, box_size, grid_size
    )
    
    patches = cut_field(
        field_data[None,...], 
        snapshot['inner_size'],
        stride = snapshot['stride'] * snapshot['inner_size'],
        pad    = snapshot['padding']
    )
        
    patch_num = snapshot['snap_num'] * snapshot['patches_per_snapshot']
    for patch in patches:
        patch_file = snapshot['patch_dir'] / f'patch_{patch_num}.h5'
        patch_num += 1
        
        with h5py.File(patch_file, 'a') as file:
            file.create_dataset(dataset_name, data=patch, compression="gzip")
            file.attrs['scale_factor'] = a
    
    mean = field_data.mean()
    var  = field_data.var()
    size = field_data.size
    return mean, var, size