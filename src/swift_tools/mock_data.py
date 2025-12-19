#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 18:10:40 2025

@author: brennan
"""

import h5py as h5
import numpy as np


def generate_mock_snapshots(
        snapshot_dir, 
        num_mock_runs, 
        num_snapshots, 
        LR_size, 
        HR_size
    ):
    
    for run in range(num_mock_runs):
        LR_snapshot_dir = (snapshot_dir / f'run{run}') / 'LR_snapshots'
        HR_snapshot_dir = (snapshot_dir / f'run{run}') / 'HR_snapshots'
        
        LR_snapshot_dir.mkdir(parents=True, exist_ok=True)
        HR_snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        for snap in range(num_snapshots):
            generate_mock_snapshot(snap, LR_snapshot_dir, LR_size)
            generate_mock_snapshot(snap, HR_snapshot_dir, HR_size)


def generate_mock_snapshot(name, snapshot_dir, size):
    snapshot = snapshot_dir / f'snap_{name}.hdf5'
    
    h = 0.7
    a = 1
    mass = 1.0
    box_size = 1.0
    chunk_dim = size**3
    grid_size = size
    vector_chunks = (chunk_dim, 3)
    scalar_chunks = (chunk_dim,)
    
    with h5.File(snapshot, 'w') as file:
        
        cosmology = file.create_group('Cosmology')
        cosmology.attrs['h'] = np.array([h], dtype='f8')
        cosmology.attrs['Scale-factor'] = np.array([a], dtype='f8')
        
        ics = file.create_group('ICs_parameters')
        ics.attrs['Grid Resolution'] = grid_size

        header = file.create_group('Header')
        header.attrs['BoxSize'] = np.array([box_size], dtype='f8')

        dm_data = file.create_group('DMParticles')
        
        dm_data.create_dataset(
            'Coordinates', shape=(grid_size**3, 3), 
            dtype='f8', compression='gzip', compression_opts=4,
            chunks=vector_chunks, fillvalue=0
        )
        
        dm_data.create_dataset(
            'ParticleIDs', shape=(grid_size**3,), 
            dtype='u8', compression='gzip', compression_opts=4,
            chunks=scalar_chunks, fillvalue=0
        )
        
        dm_data.create_dataset(
            'Velocities', shape=(grid_size**3, 3), 
            dtype='f4', compression='gzip', compression_opts=4,
            chunks=vector_chunks, fillvalue=0
        )
        
        dm_data.create_dataset('Masses', data=np.array([mass], dtype='f8'))