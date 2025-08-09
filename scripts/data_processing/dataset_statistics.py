#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 21:31:22 2025

@author: brennan
"""

import h5py
import numpy as np

from pathlib import Path


def compute_stats(patches, field):
    """
    Use Welford's algorithm to compute summary statistics of data from multiple
    patch files.
    """
    # Initialize counters
    count, mean, M2 = 0, 0.0, 0.0
    
    for patch_path in patches:
        with h5py.open(patch_path, 'r') as patch_file:
            patch = patch_file[field]
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


dataset_dir = Path('../../data/dmsr_style_valid/').resolve()
fields = [
    'LR_Coordinates', 'HR_Coordinates', 'LR_Velocities', 'HR_Velocities'
]


#%%
patch_dir = dataset_dir / 'patches/'

patches = list(patch_dir.iterdir())
patches.sort(key = lambda s: (len(s), s))

stats = {}
for field in fields:
    print('Computing stats for', field)
    
    mean, standard_deviation = compute_stats(patches, field)
    stats[field + '_std'] = standard_deviation
    stats[field + '_mean'] = mean

print('stats computed:\n', stats)
np.save(dataset_dir + 'summary_stats.npy', stats)