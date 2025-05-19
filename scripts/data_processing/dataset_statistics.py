#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 21:31:22 2025

@author: brennan
"""

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np


def compute_stats(patches):
    """
    Use Welford's algorithm to compute summary statistics of data from multiple
    patch files.
    """
    # Initialize counters
    count = 0
    mean = 0.0
    M2 = 0.0
    
    for patch_file in patches:
        patch = np.load(patch_file)
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


dataset_dir = '../../data/dmsr_style_valid/'
fields = [
    'LR_disp_fields', 'HR_disp_fields', 'LR_vel_fields', 'HR_vel_fields'
]

scale_factor_file = dataset_dir + 'scale_factors.npy'
scale_factors = np.load(scale_factor_file)


#%%
field_dir = dataset_dir + fields[0] + '/'

patches = os.listdir(field_dir)
patches.sort(key = lambda s: (len(s), s))

stats = {}
for field in fields:
    print('Computing stats for', field)
    if not os.path.isdir(dataset_dir + field):
        continue
    
    field_patches = [
        dataset_dir + field + '/' + patch 
        for patch in patches
    ]
    
    mean, standard_deviation = compute_stats(field_patches)
    stats[field + '_std'] = standard_deviation
    stats[field + '_mean'] = mean

print('stats computed:\n', stats)
np.save(dataset_dir + 'summary_stats.npy', stats)