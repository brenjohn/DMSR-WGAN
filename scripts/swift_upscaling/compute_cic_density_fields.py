#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:12:16 2025

@author: brennan

This script computes the CIC density field for swift snapshots specified by the
user and saves the result as a NumPy file. The computation is performed in 
chunks to manage memory usage for large files.
"""

import argparse
import h5py as h5
import numpy as np

from pathlib import Path
from dm_analysis import cloud_in_cells
from swift_tools.data import read_metadata


def read_positions(file, chunk, chunk_size):
    """Reads the positions, grid_size, box_size, h, particle mass and scale 
    factor from the given snapshot.
    """
    dm_data    = file['DMParticles']
    positions  = np.asarray(dm_data['Coordinates'][chunk:chunk+chunk_size])
    return positions


def compute_cic_density_fields(
        data_dir, 
        snapshot_pattern, 
        chunk_size, 
        cic_grid_size = None
    ):
    """Calculates the CIC density field for all snapshots and saves the result 
    as a NumPy file. The computation is performed in chunks to manage memory 
    usage for large files.
    """
    snapshots = np.sort(list(data_dir.glob(snapshot_pattern)))

    for snapshot_file in snapshots:
        print(f'\nComputing denisty for {snapshot_file.stem}')
        grid_size, box_size, h, mass, a = read_metadata(snapshot_file)
        density = 0
        num_particles = grid_size**3
        
        if cic_grid_size is None:
            cic_grid_size = grid_size
        
        with h5.File(snapshot_file, 'r') as snapshot:
            for chunk in range(0, num_particles, chunk_size):
                print(
                    f'Processing chunk {chunk} to {chunk+chunk_size}\r', 
                    end=''
                )
                positions = read_positions(snapshot, chunk, chunk_size)
                density += cloud_in_cells(positions, cic_grid_size, box_size)
            
            np.save(data_dir / (snapshot_file.stem + '_density.npy'), density)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate CIC density fields for SWIFT snapshots."
    )
    
    parser.add_argument(
        '--data_dir', 
        type=Path, 
        default='./swift_snapshots/meraxes_runs/256/',
        help='Path to the directory containing the SWIFT snapshot files.'
    )
    
    parser.add_argument(
        '--pattern', 
        type=str, 
        default='snap_*.hdf5',
        help='Glob pattern to match snapshot files (e.g., "snap_*.hdf5").'
    )
    
    parser.add_argument(
        '--chunk_size', 
        type=int, 
        default=256**3,
        help='The number of particles to read from disk at any one time.'
    )
    
    parser.add_argument(
        '--cic_grid_size', 
        type=int, 
        default=None,
        help='The grid size for the cic density field.'
    )
    
    args = parser.parse_args()
    compute_cic_density_fields(
        args.data_dir, 
        args.pattern, 
        args.chunk_size,
        args.cic_grid_size
    )