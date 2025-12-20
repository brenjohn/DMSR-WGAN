#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:46:00 2025

@author: brennan
"""

import h5py
import argparse
import numpy as np

from pathlib import Path
from dm_analysis import power_spectrum
from swift_tools.fields import get_positions
from dm_analysis import cloud_in_cells


def main(args):
    dataset_dir = args.dataset_dir
    patch_dir = dataset_dir / 'patches/'
    
    metadata = np.load(dataset_dir / 'metadata.npy', allow_pickle=True).item()
    HR_patch_length = metadata['HR_patch_length']
    HR_patch_size   = metadata['HR_patch_size']
    HR_mass         = metadata['HR_mass']
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
            k_bins, spectrum, _ = power_spectrum(density, box_size, grid_size)
            
            file.create_dataset('HR_Power_Spectrum', data = spectrum)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a dataset."
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        help="Path to the dataset."
    )
    
    args = parser.parse_args()
    main(args)