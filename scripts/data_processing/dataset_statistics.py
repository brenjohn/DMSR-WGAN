#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 21:31:22 2025

@author: brennan
"""

import h5py
import numpy as np
import multiprocessing as mp

from pathlib import Path


def compute_stats(patches, field):
    """
    Use Welford's algorithm to compute summary statistics of data from multiple
    patch files.
    """
    # Initialize counters
    count, mean, M2 = 0, 0.0, 0.0
    
    for patch_path in patches:
        with h5py.File(patch_path, 'r') as patch_file:
            patch = patch_file[field][()]
            patch = patch.ravel()
            
            n = patch.size
            new_mean = patch.mean()
            new_M2 = ((patch - new_mean) ** 2).sum()
        
            delta = new_mean - mean
            total = count + n
            
            mean += delta * n / total
            M2 += new_M2 + delta**2 * count * n / total
            count = total
    
    variance = M2 / count
    return mean, np.sqrt(variance)


def worker_compute_stats(task):
    """Worker function to compute stats for a single field."""
    field, patch_dir = task
    print(f'Computing stats for {field}\n')
    patches = list(patch_dir.iterdir())
    
    mean, standard_deviation = compute_stats(patches, field)
    return field, mean, standard_deviation


#%%
dataset_dir = Path('../../data/dmsr_style_train/').resolve()
patch_dir = dataset_dir / 'patches/'
fields = [
    'LR_Coordinates', 'HR_Coordinates', 'LR_Velocities', 'HR_Velocities'
]

patches = list(patch_dir.iterdir())
tasks = [(field, patch_dir) for field in fields]
stats = {}

with mp.Pool(4) as pool:
    results = pool.map(worker_compute_stats, tasks)
        
# Process the results to populate the stats dictionary
for field, mean, std_dev in results:
    stats[f'{field}_std'] = std_dev
    stats[f'{field}_mean'] = mean

print('stats computed:\n', stats)
np.save(dataset_dir / 'summary_stats.npy', stats)